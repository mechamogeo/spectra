from fastapi.testclient import TestClient
from src.core.app import app

client = TestClient(app)

def test_entrypoint():
    """Test the API root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to SPECTRA API, read the docs at /docs"}
