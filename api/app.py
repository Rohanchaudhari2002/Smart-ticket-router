"""
app.py - FastAPI REST API for Smart Incident Ticket Router
"""

import os
import sys
import time
import logging
import io
import csv
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import predict_department, get_supported_departments
from src.priority import detect_priority
from src.database import init_db, insert_ticket, get_tickets, get_stats, log_request
from src.utils import setup_logging, request_counter, sanitize_ticket, truncate_text
from src.preprocess import validate_ticket

# Setup logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Incident Ticket Router",
    description=(
        "Automatically classify IT support tickets and assign them "
        "to the correct department with priority level detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    ticket: str = Field(..., min_length=3, max_length=5000, example="Unable to login to VPN")

    @validator("ticket")
    def ticket_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Ticket text cannot be blank")
        return v.strip()


class TicketResponse(BaseModel):
    ticket_id: int
    ticket: str
    department: str
    priority: str
    confidence: float
    latency_ms: float
    timestamp: str


class BatchTicketRequest(BaseModel):
    tickets: List[str] = Field(..., min_items=1, max_items=500)


class BatchTicketResult(BaseModel):
    ticket: str
    department: str
    priority: str
    confidence: float


class BatchTicketResponse(BaseModel):
    total: int
    results: List[BatchTicketResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    departments: List[str]
    total_tickets_processed: int


# ─── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize DB and pre-load model on startup."""
    logger.info("Starting Smart Incident Ticket Router...")
    init_db()
    try:
        departments = get_supported_departments()
        logger.info(f"Model loaded. Supported departments: {departments}")
    except FileNotFoundError as e:
        logger.warning(f"Model not yet trained: {e}")
    logger.info("API ready.")
# Serve index.html at http://127.0.0.1:8000/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path) as f:
        return f.read()

# ─── Middleware: Request Logging ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000

    endpoint = str(request.url.path)
    request_counter.increment(endpoint)

    try:
        log_request(endpoint, response.status_code, round(latency_ms, 2))
    except Exception:
        pass  # Don't fail requests due to logging errors

    logger.info(
        f"{request.method} {endpoint} -> {response.status_code} ({latency_ms:.1f}ms)"
    )
    return response


# ─── Routes ─────────────────────────────────────────────────────────────────




@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint."""
    try:
        departments = get_supported_departments()
        model_loaded = True
    except Exception:
        departments = []
        model_loaded = False

    stats = get_stats()
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        departments=departments,
        total_tickets_processed=stats.get("total_tickets", 0),
    )


@app.post("/predict", response_model=TicketResponse, tags=["Prediction"])
def predict(request: TicketRequest):
    """
    Classify a single IT support ticket.

    - **ticket**: The support ticket text to classify

    Returns the predicted department, priority level, confidence score, and ticket ID.
    """
    ticket_text = sanitize_ticket(request.ticket)
    logger.info(f"Predicting for ticket: '{truncate_text(ticket_text, 100)}'")

    start_time = time.perf_counter()

    # Validate
    is_valid, error_msg = validate_ticket(ticket_text)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)

    try:
        # Predict department
        prediction = predict_department(ticket_text)
        department = prediction["department"]
        confidence = prediction["confidence"]

        # Detect priority
        priority = detect_priority(ticket_text)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Store in database
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        ticket_id = insert_ticket(
            ticket_text=ticket_text,
            department=department,
            priority=priority,
            confidence=confidence,
            latency_ms=latency_ms,
            source="api",
        )

        logger.info(
            f"Ticket #{ticket_id} -> Department: {department}, Priority: {priority}, "
            f"Confidence: {confidence:.2%}, Latency: {latency_ms}ms"
        )

        return TicketResponse(
            ticket_id=ticket_id,
            ticket=ticket_text,
            department=department,
            priority=priority,
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict/batch", response_model=BatchTicketResponse, tags=["Prediction"])
def predict_batch(request: BatchTicketRequest):
    """
    Classify multiple IT support tickets in one request.

    - **tickets**: List of ticket texts (max 500)
    """
    start_time = time.perf_counter()
    results = []

    for ticket_text in request.tickets:
        ticket_text = sanitize_ticket(ticket_text)
        try:
            is_valid, _ = validate_ticket(ticket_text)
            if not is_valid:
                department, priority, confidence = "Unknown", "Medium", 0.0
            else:
                prediction = predict_department(ticket_text)
                department = prediction["department"]
                confidence = prediction["confidence"]
                priority = detect_priority(ticket_text)

                insert_ticket(
                    ticket_text=ticket_text,
                    department=department,
                    priority=priority,
                    confidence=confidence,
                    source="batch_api",
                )
        except Exception as e:
            logger.warning(f"Failed to classify ticket '{truncate_text(ticket_text, 50)}': {e}")
            department, priority, confidence = "Unknown", "Medium", 0.0

        results.append(BatchTicketResult(
            ticket=ticket_text,
            department=department,
            priority=priority,
            confidence=confidence,
        ))

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"Batch prediction: {len(results)} tickets in {latency_ms}ms")

    return BatchTicketResponse(
        total=len(results),
        results=results,
        latency_ms=latency_ms,
    )


@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with a 'ticket' column to classify all tickets.

    Returns enriched CSV with department and priority columns.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if "ticket" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must contain a 'ticket' column")

    departments, priorities, confidences = [], [], []

    for ticket_text in df["ticket"].fillna("").astype(str):
        try:
            is_valid, _ = validate_ticket(ticket_text)
            if not is_valid:
                departments.append("Unknown")
                priorities.append("Medium")
                confidences.append(0.0)
                continue
            prediction = predict_department(ticket_text)
            departments.append(prediction["department"])
            priorities.append(detect_priority(ticket_text))
            confidences.append(prediction["confidence"])
            insert_ticket(ticket_text, departments[-1], priorities[-1], confidences[-1], source="csv_upload")
        except Exception:
            departments.append("Unknown")
            priorities.append("Medium")
            confidences.append(0.0)

    df["department"] = departments
    df["priority"] = priorities
    df["confidence"] = confidences

    output = df.to_csv(index=False)
    return JSONResponse(content={
        "total": len(df),
        "results": df.to_dict(orient="records"),
        "csv": output,
    })


@app.get("/tickets", tags=["History"])
def list_tickets(
    limit: int = Query(50, ge=1, le=500),
    department: Optional[str] = None,
    priority: Optional[str] = None,
):
    """Retrieve recent classified tickets with optional filters."""
    tickets = get_tickets(limit=limit, department=department, priority=priority)
    return {"total": len(tickets), "tickets": tickets}


@app.get("/stats", tags=["Monitoring"])
def statistics():
    """Return aggregate statistics and monitoring data."""
    db_stats = get_stats()
    request_stats = request_counter.summary()
    return {
        "database": db_stats,
        "requests": request_stats,
    }


@app.get("/departments", tags=["General"])
def list_departments():
    """List all departments the model can classify."""
    try:
        departments = get_supported_departments()
        return {"departments": departments}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet")
