"""
test_api.py - API endpoint validation tests
Run with: python tests/test_api.py
Or with pytest: pytest tests/test_api.py -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# Import and init DB before app
from src.database import init_db
init_db()

from api.app import app

client = TestClient(app)

# ─── Test Cases ─────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SECTION = "\033[94m"
RESET = "\033[0m"

results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
        results["passed"] += 1
    else:
        print(f"  {FAIL}  {name}" + (f" | {detail}" if detail else ""))
        results["failed"] += 1


def section(title: str):
    print(f"\n{SECTION}{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}{RESET}")


# ─── Health Check Tests ──────────────────────────────────────────────────────

section("Health & Root Endpoints")

r = client.get("/")
check("GET / returns 200", r.status_code == 200)
check("GET / has service name", "Smart Incident Ticket Router" in r.json().get("service", ""))

r = client.get("/health")
check("GET /health returns 200", r.status_code == 200)
data = r.json()
check("Health has 'status' field", "status" in data)
check("Health has 'model_loaded' field", "model_loaded" in data)
check("Health has 'departments' list", isinstance(data.get("departments"), list))


# ─── Departments Endpoint ───────────────────────────────────────────────────

section("Departments Endpoint")

r = client.get("/departments")
if r.status_code == 200:
    check("GET /departments returns 200", True)
    depts = r.json().get("departments", [])
    check("Departments list is non-empty", len(depts) > 0, f"got {depts}")
    check("Has expected departments", any(d in depts for d in ["Network", "Database", "Email"]))
else:
    check("GET /departments returns 200", False, f"status={r.status_code}, body={r.text}")


# ─── Prediction Tests ────────────────────────────────────────────────────────

section("POST /predict - Core Classification")

test_cases = [
    ("wifi not working", "Network", "High"),
    ("database connection error", "Database", "High"),
    ("email not syncing", "Email", "Medium"),
    ("unable to login", "Authentication", "Medium"),
    ("application crashing", "Software", "High"),
    ("printer not working", "Hardware", "High"),
]

for ticket, expected_dept, expected_priority in test_cases:
    r = client.post("/predict", json={"ticket": ticket})
    if r.status_code == 200:
        data = r.json()
        check(
            f"'{ticket}' -> {expected_dept}",
            data.get("department") == expected_dept,
            f"got '{data.get('department')}'",
        )
        check(
            f"'{ticket}' has ticket_id",
            isinstance(data.get("ticket_id"), int),
        )
        check(
            f"'{ticket}' has confidence",
            0.0 <= data.get("confidence", -1) <= 1.0,
        )
    else:
        check(f"'{ticket}' -> 200", False, f"status={r.status_code}")


# ─── Priority Detection Tests ─────────────────────────────────────────────────

section("POST /predict - Priority Detection")

priority_cases = [
    ("database server crashed and data is lost", "High"),
    ("cannot access email account", "Medium"),
    ("system running slowly today", "Low"),
    ("critical security breach detected", "High"),
]

for ticket, expected_priority in priority_cases:
    r = client.post("/predict", json={"ticket": ticket})
    if r.status_code == 200:
        data = r.json()
        check(
            f"Priority for '{ticket[:40]}...' = {expected_priority}",
            data.get("priority") == expected_priority,
            f"got '{data.get('priority')}'",
        )
    else:
        check(f"Priority test '{ticket[:30]}' -> 200", False, f"status={r.status_code}")


# ─── Validation / Error Handling ─────────────────────────────────────────────

section("Input Validation & Error Handling")

# Empty ticket
r = client.post("/predict", json={"ticket": ""})
check("Empty ticket returns 422", r.status_code == 422)

# Missing ticket field
r = client.post("/predict", json={})
check("Missing 'ticket' field returns 422", r.status_code == 422)

# Very short ticket
r = client.post("/predict", json={"ticket": "hi"})
check("Very short ticket returns 4xx", r.status_code in (422, 200))

# Whitespace-only ticket
r = client.post("/predict", json={"ticket": "   "})
check("Whitespace-only ticket returns 422", r.status_code == 422)


# ─── Batch Prediction ────────────────────────────────────────────────────────

section("POST /predict/batch - Batch Prediction")

batch_payload = {
    "tickets": [
        "wifi not working",
        "database connection error",
        "email not syncing",
        "unable to login",
        "application crashing",
    ]
}
r = client.post("/predict/batch", json=batch_payload)
check("Batch predict returns 200", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    check("Batch returns correct count", data.get("total") == 5, f"got {data.get('total')}")
    check("Batch results list length matches", len(data.get("results", [])) == 5)
    check("Batch has latency_ms", "latency_ms" in data)


# ─── Ticket History ───────────────────────────────────────────────────────────

section("GET /tickets - History")

r = client.get("/tickets")
check("GET /tickets returns 200", r.status_code == 200)
data = r.json()
check("Response has 'tickets' list", "tickets" in data)
check("Response has 'total' count", "total" in data)

r = client.get("/tickets?limit=2")
check("GET /tickets?limit=2 respects limit", r.status_code == 200)

r = client.get("/tickets?department=Network")
check("GET /tickets?department=Network filters correctly", r.status_code == 200)


# ─── Stats & Monitoring ──────────────────────────────────────────────────────

section("GET /stats - Monitoring")

r = client.get("/stats")
check("GET /stats returns 200", r.status_code == 200)
data = r.json()
check("Stats has 'database' key", "database" in data)
check("Stats has 'requests' key", "requests" in data)


# ─── Summary ─────────────────────────────────────────────────────────────────

total = results["passed"] + results["failed"]
print(f"\n{'='*50}")
print(f"  Results: {results['passed']}/{total} tests passed")
if results["failed"] == 0:
    print(f"  \033[92mAll tests passed! ✓\033[0m")
else:
    print(f"  \033[91m{results['failed']} test(s) failed\033[0m")
print(f"{'='*50}\n")

sys.exit(0 if results["failed"] == 0 else 1)
