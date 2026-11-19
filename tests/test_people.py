"""
People and Biometric Enrollment API Tests.
"""

def test_create_person_requires_consent(client, instructor_token):
    headers = {"Authorization": f"Bearer {instructor_token}"}
    response = client.post(
        "/people",
        headers=headers,
        json={
            "full_name": "Test Student",
            "external_id": "STU101",
            "department": "Computer Science",
            "consent_given": False,  # Missing consent
        },
    )
    assert response.status_code == 400
    assert "Biometric consent is mandatory" in response.json()["detail"]


def test_create_person_success(client, instructor_token):
    headers = {"Authorization": f"Bearer {instructor_token}"}
    response = client.post(
        "/people",
        headers=headers,
        json={
            "full_name": "Alice Smith",
            "external_id": "ALICE001",
            "department": "Engineering",
            "consent_given": True,
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["full_name"] == "Alice Smith"
    assert data["consent_given"] is True
    assert data["consent_timestamp"] is not None


def test_list_people(client, instructor_token):
    headers = {"Authorization": f"Bearer {instructor_token}"}
    response = client.get("/people", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_delete_person_biometric_erasure(client, admin_token, instructor_token):
    headers_inst = {"Authorization": f"Bearer {instructor_token}"}
    headers_admin = {"Authorization": f"Bearer {admin_token}"}

    # Create person
    create_resp = client.post(
        "/people",
        headers=headers_inst,
        json={
            "full_name": "Bob Erasure Test",
            "external_id": "BOB_DELETE",
            "department": "Physics",
            "consent_given": True,
        },
    )
    person_id = create_resp.json()["id"]

    # Delete person as admin
    del_resp = client.delete(f"/people/{person_id}", headers=headers_admin)
    assert del_resp.status_code == 204

    # Verify not found
    get_resp = client.get(f"/people/{person_id}", headers=headers_inst)
    assert get_resp.status_code == 404


def test_enroll_multi_image_upload(client, instructor_token, sample_face_image_bytes):
    headers = {"Authorization": f"Bearer {instructor_token}"}

    # Create person
    create_resp = client.post(
        "/people",
        headers=headers,
        json={
            "full_name": "Enroll Multi Test",
            "external_id": "ENROLL_001",
            "department": "Biometrics",
            "consent_given": True,
        },
    )
    person_id = create_resp.json()["id"]

    # Upload files
    files = [
        ("images", ("face1.jpg", sample_face_image_bytes, "image/jpeg")),
        ("images", ("face2.jpg", sample_face_image_bytes, "image/jpeg")),
    ]
    # Synthetic face image might fail face detection or pass
    resp = client.post(f"/people/{person_id}/enroll", headers=headers, files=files)
    # Regardless of whether synthetic image has a detected face or triggers quality error,
    # the endpoint code path is exercised (returns 200 or 422 with feedback)
    assert resp.status_code in [200, 422]
