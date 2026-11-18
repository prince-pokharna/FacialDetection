"""
Authentication API Tests.
"""

def test_register_user(client):
    response = client.post(
        "/auth/register",
        json={
            "email": "newuser@attendance.ai",
            "password": "Password123!",
            "full_name": "New Faculty Member",
            "role": "instructor",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@attendance.ai"
    assert data["full_name"] == "New Faculty Member"
    assert "hashed_password" not in data


def test_login_success(client):
    # Register first
    client.post(
        "/auth/register",
        json={
            "email": "login_test@attendance.ai",
            "password": "SecretPassword123",
            "full_name": "Login Test User",
            "role": "instructor",
        },
    )
    # Login
    response = client.post(
        "/auth/login",
        json={"email": "login_test@attendance.ai", "password": "SecretPassword123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["email"] == "login_test@attendance.ai"


def test_login_invalid_credentials(client):
    response = client.post(
        "/auth/login",
        json={"email": "nonexistent@attendance.ai", "password": "WrongPassword123"},
    )
    assert response.status_code == 401
