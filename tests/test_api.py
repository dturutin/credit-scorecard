from fastapi.testclient import TestClient

from api.main import app


def test_score_endpoint_known_applicant():
    with TestClient(app) as client:
        response = client.post(
            "/credit-scorecard/score",
            json={
                "checking_status": "A14",
                "credit_history": "A34",
                "purpose": "A43",
                "savings_status": "A65",
                "employment": "A75",
                "personal_status_sex": "A93",
                "other_parties": "A101",
                "property_magnitude": "A121",
                "other_payment_plans": "A143",
                "housing": "A152",
                "foreign_worker": "A201",
                "duration": 6,
                "credit_amount": 1000,
                "installment_rate": 1,
                "age": 50,
            },
        )
        assert response.status_code == 200
        assert response.json()["risk_band"] in ("low", "medium", "high")
