import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()

def test_predict_purchase():
    response = client.get("/predict_purchase", params={"itemid": 356475, "hour": 14, "weekday": 3})
    assert response.status_code == 200
    assert "purchase_probability" in response.json()