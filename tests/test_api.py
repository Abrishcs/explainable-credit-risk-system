"""
test_api.py
-----------
Tests the Flask API without needing curl or Postman.
Run this WHILE the Flask server is running in another terminal.

Tests:
  1. Health check
  2. Example endpoint
  3. High-risk applicant -> expect REJECT
  4. Low-risk applicant  -> expect APPROVE
  5. Bad input           -> expect 400 error
  6. Missing field       -> expect 400 error
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def show_response(resp):
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(json.dumps(data, indent=4))

# ─── Test 1: Health Check ──────────────────────────────────────────────────
section("TEST 1: Health Check")
resp = requests.get(f"{BASE_URL}/health")
show_response(resp)
assert resp.status_code == 200, "Health check failed!"
print("  PASSED")

# ─── Test 2: Example Endpoint ─────────────────────────────────────────────
section("TEST 2: Example Endpoint")
resp = requests.get(f"{BASE_URL}/example")
show_response(resp)
assert resp.status_code == 200
print("  PASSED")

# ─── Test 3: High-Risk Applicant (Expect REJECT) ──────────────────────────
section("TEST 3: High-Risk Applicant")
high_risk = {
    "Age":              22,
    "Sex":              "female",
    "Job":              0,
    "Housing":          "rent",
    "Saving accounts":  "none",
    "Checking account": "none",
    "Credit amount":    15000,
    "Duration":         60,
    "Purpose":          "education"
}
resp = requests.post(f"{BASE_URL}/predict", json=high_risk)
show_response(resp)
assert resp.status_code == 200
result = resp.json()
print(f"\n  Decision:      {result['decision']}")
print(f"  Probability:   {result['probability_of_default']}")
print(f"  Risk Level:    {result['risk_level']}")
print(f"  Explanation:   {result['explanation']}")
print("  PASSED")

# ─── Test 4: Low-Risk Applicant (Expect APPROVE) ──────────────────────────
section("TEST 4: Low-Risk Applicant")
low_risk = {
    "Age":              45,
    "Sex":              "male",
    "Job":              3,
    "Housing":          "own",
    "Saving accounts":  "rich",
    "Checking account": "rich",
    "Credit amount":    1500,
    "Duration":         12,
    "Purpose":          "car"
}
resp = requests.post(f"{BASE_URL}/predict", json=low_risk)
show_response(resp)
assert resp.status_code == 200
result = resp.json()
print(f"\n  Decision:      {result['decision']}")
print(f"  Probability:   {result['probability_of_default']}")
print(f"  Risk Level:    {result['risk_level']}")
print(f"  Explanation:   {result['explanation']}")
print("  PASSED")

# ─── Test 5: Invalid Value (Expect 400) ───────────────────────────────────
section("TEST 5: Invalid Field Value")
bad_input = {
    "Age":              35,
    "Sex":              "unknown",   # <-- invalid
    "Job":              2,
    "Housing":          "own",
    "Saving accounts":  "little",
    "Checking account": "little",
    "Credit amount":    5000,
    "Duration":         24,
    "Purpose":          "car"
}
resp = requests.post(f"{BASE_URL}/predict", json=bad_input)
show_response(resp)
assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
print("  PASSED (correctly rejected bad input)")

# ─── Test 6: Missing Field (Expect 400) ───────────────────────────────────
section("TEST 6: Missing Required Field")
missing_field = {
    "Age":    35,
    "Sex":    "male",
    # 'Job' is missing
}
resp = requests.post(f"{BASE_URL}/predict", json=missing_field)
show_response(resp)
assert resp.status_code == 400
print("  PASSED (correctly caught missing field)")

print(f"\n{'='*55}")
print("  ALL TESTS PASSED")
print(f"{'='*55}\n")
