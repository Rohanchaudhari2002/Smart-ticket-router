# 🎫 Smart Incident Ticket Router

Automatically classify IT support tickets and assign them to the correct department with priority detection — powered by NLP and machine learning.

---

## Architecture

```
smart-ticket-router/
│
├── data/
│   └── tickets.csv          # Labeled training dataset
│
├── models/
│   └── classifier.pkl       # Trained sklearn Pipeline (TF-IDF + LogReg)
│
├── src/
│   ├── preprocess.py        # Text cleaning & tokenization
│   ├── train_model.py       # Model training, evaluation, persistence
│   ├── classifier.py        # Load model & predict department
│   ├── priority.py          # Rule-based priority detection
│   ├── database.py          # SQLite storage layer
│   └── utils.py             # Logging, timing, request counter
│
├── api/
│   └── app.py               # FastAPI REST API (all endpoints)
│
├── tests/
│   └── test_api.py          # API + module validation tests
│
├── requirements.txt
├── Dockerfile
└── README.md
```

### Data Flow

```
Ticket Text
    │
    ▼
[preprocess.py]  ──── lowercase, remove punctuation, remove stopwords
    │
    ▼
[classifier.py]  ──── TF-IDF vectorize → Logistic Regression → Department
    │
    ▼
[priority.py]    ──── keyword matching → Priority (High / Medium / Low)
    │
    ▼
[database.py]    ──── INSERT into SQLite (ticket, department, priority, ts)
    │
    ▼
[api/app.py]     ──── return JSON response
```

---

## Supported Departments

| Department     | Example Tickets                                  |
|----------------|--------------------------------------------------|
| Network        | wifi not working, VPN connection failed          |
| Database       | database connection error, SQL timeout           |
| Email          | email not syncing, Outlook not working           |
| Authentication | unable to login, account locked, password reset  |
| Hardware       | laptop not turning on, screen flickering         |
| Software       | application crashing, installation failed        |

## Priority Levels

| Priority | Triggered By Keywords                                       |
|----------|-------------------------------------------------------------|
| **High** | error, failure, crash, down, corrupted, outage, breach ...  |
| **Medium**| unable, cannot, not responding, timeout, blocked ...       |
| **Low**  | slow, delay, minor, occasionally, request, how to ...       |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

This will:
- Load `data/tickets.csv`
- Preprocess tickets
- Train TF-IDF + Logistic Regression pipeline
- Print accuracy, classification report, and confusion matrix
- Save model to `models/classifier.pkl`

### 3. Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

Or with environment configuration:

```bash
MODEL_PATH=models/classifier.pkl DB_PATH=data/tickets.db uvicorn api.app:app --reload
```

### 4. View API docs

Open your browser at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### `POST /predict`
Classify a single ticket.

**Request:**
```json
{ "ticket": "Unable to login to VPN" }
```

**Response:**
```json
{
  "ticket_id": 42,
  "ticket": "Unable to login to VPN",
  "department": "Authentication",
  "priority": "Medium",
  "confidence": 0.7231,
  "latency_ms": 4.2,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### `POST /predict/batch`
Classify multiple tickets at once (max 500).

**Request:**
```json
{
  "tickets": [
    "wifi not working",
    "database connection error",
    "application crashing"
  ]
}
```

**Response:**
```json
{
  "total": 3,
  "results": [
    { "ticket": "wifi not working", "department": "Network", "priority": "High", "confidence": 0.82 },
    { "ticket": "database connection error", "department": "Database", "priority": "High", "confidence": 0.91 },
    { "ticket": "application crashing", "department": "Software", "priority": "High", "confidence": 0.75 }
  ],
  "latency_ms": 12.4
}
```

---

### `POST /predict/csv`
Upload a CSV file with a `ticket` column to classify all tickets at once.

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@my_tickets.csv"
```

CSV format:
```csv
ticket
wifi not working
database error
email not syncing
```

---

### `GET /tickets`
Retrieve classified ticket history.

```
GET /tickets?limit=10&department=Network&priority=High
```

---

### `GET /stats`
Monitoring dashboard data.

```json
{
  "database": {
    "total_tickets": 1024,
    "by_department": [{"department": "Network", "count": 312}, ...],
    "by_priority": [{"priority": "High", "count": 445}, ...],
    "avg_latency_ms": 5.3
  },
  "requests": {
    "total": 1089,
    "by_endpoint": { "/predict": 1024, "/stats": 65 }
  }
}
```

---

### `GET /departments`
List all supported departments.

### `GET /health`
Health check endpoint.

---

## Testing

### Run all tests
```bash
python tests/test_api.py
```

### Run with pytest
```bash
pytest tests/test_api.py -v
```

---

## Docker

### Build
```bash
docker build -t smart-ticket-router .
```

The Dockerfile trains the model automatically during the build step.

### Run
```bash
docker run -p 8000:8000 smart-ticket-router
```

### With persistent storage
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  smart-ticket-router
```

### Environment variables

| Variable     | Default                     | Description              |
|--------------|-----------------------------|--------------------------|
| `MODEL_PATH` | `models/classifier.pkl`     | Path to trained model    |
| `DB_PATH`    | `data/tickets.db`           | SQLite database path     |
| `DATA_PATH`  | `data/tickets.csv`          | Training data path       |
| `LOG_LEVEL`  | `INFO`                      | Logging verbosity        |

---

## Extending the Dataset

Add more rows to `data/tickets.csv` in this format:

```csv
ticket,department
cannot connect to sharepoint,Network
oracle db tablespace full,Database
gmail two factor not working,Authentication
```

Then retrain:
```bash
python src/train_model.py
```

---

## Future ML Improvements

The architecture is designed to be extended. Planned enhancements:

### BERT-Based Classification
Replace TF-IDF + LogReg with a transformer model for much higher accuracy:
```python
# In src/classifier.py — swap predict_department() implementation
from transformers import pipeline
classifier = pipeline("text-classification", model="bert-base-uncased")
```

### Anomaly Detection
Flag tickets that don't resemble any known department (OOD detection):
```python
from sklearn.covariance import EllipticEnvelope
detector = EllipticEnvelope(contamination=0.1)
```

### Feedback Learning System
Store human-corrected labels and retrain periodically:
```
POST /feedback  { "ticket_id": 42, "correct_department": "Network" }
```
Triggers scheduled retraining when feedback volume exceeds threshold.

---

## Project Tech Stack

| Layer        | Technology                     |
|--------------|--------------------------------|
| ML Pipeline  | scikit-learn (TF-IDF + LogReg) |
| Data         | pandas, CSV                    |
| API          | FastAPI + Uvicorn              |
| Database     | SQLite                         |
| Persistence  | pickle                         |
| Testing      | pytest + httpx                 |
| Deployment   | Docker                         |
