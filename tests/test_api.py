"""API surface: auth, scenarios, runs, agents, and the hardening headers."""

from __future__ import annotations


def open_road_id(client, headers) -> int:
    scenarios = client.get("/api/scenarios", headers=headers).json()
    return next(s["id"] for s in scenarios if "open road" in s["name"])


#  meta


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_meta_describes_the_platform(client):
    response = client.get("/api/meta")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_root_serves_the_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Safar" in response.text


def test_security_headers_are_present(client):
    response = client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


#  auth


def test_scenarios_require_a_token(client):
    assert client.get("/api/scenarios").status_code == 401


def test_runs_require_a_token(client):
    assert client.post("/api/runs", json={"scenario_id": 1}).status_code == 401


def test_login_rejects_a_bad_password(client):
    response = client.post(
        "/api/auth/login", json={"email": "demo@safar.pk", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_then_me(client, auth_headers):
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "demo@safar.pk"


def test_register_rejects_a_duplicate_email(client):
    response = client.post(
        "/api/auth/register",
        json={"email": "demo@safar.pk", "password": "anotherpass1", "full_name": "Dup"},
    )
    assert response.status_code == 409


def test_register_rejects_a_weak_password(client):
    response = client.post(
        "/api/auth/register", json={"email": "weak@example.com", "password": "12345678"}
    )
    assert response.status_code == 422


#  scenarios


def test_builtin_scenarios_are_seeded(client, auth_headers):
    response = client.get("/api/scenarios", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 7
    assert any(scenario["is_builtin"] for scenario in body)


def test_create_then_delete_a_scenario(client, auth_headers):
    payload = {
        "name": "pytest scenario",
        "track_points": [[0, 0], [200, 0]],
        "duration": 10,
        "ego_speed": 8,
    }
    created = client.post("/api/scenarios", json=payload, headers=auth_headers)
    assert created.status_code == 201
    scenario_id = created.json()["id"]

    assert client.delete(f"/api/scenarios/{scenario_id}", headers=auth_headers).status_code == 204
    assert client.get(f"/api/scenarios/{scenario_id}", headers=auth_headers).status_code == 404


def test_duplicate_scenario_name_conflicts(client, auth_headers):
    payload = {"name": "dup-name", "track_points": [[0, 0], [200, 0]], "duration": 10}
    assert client.post("/api/scenarios", json=payload, headers=auth_headers).status_code == 201
    assert client.post("/api/scenarios", json=payload, headers=auth_headers).status_code == 409


def test_builtin_scenarios_cannot_be_deleted(client, auth_headers):
    scenarios = client.get("/api/scenarios", headers=auth_headers).json()
    builtin_id = next(s["id"] for s in scenarios if s["is_builtin"])
    assert client.delete(f"/api/scenarios/{builtin_id}", headers=auth_headers).status_code == 403


def test_a_one_point_track_is_rejected(client, auth_headers):
    response = client.post(
        "/api/scenarios",
        json={"name": "bad", "track_points": [[0, 0]], "duration": 10},
        headers=auth_headers,
    )
    assert response.status_code == 422


#  runs


def test_run_a_simulation(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    response = client.post(
        "/api/runs",
        json={"scenario_id": scenario_id, "agent": "pursuit", "seed": 1},
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body["frames"]) > 0
    assert "safety_score" in body["metrics"]


def test_runs_are_reproducible_over_the_api(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    payload = {"scenario_id": scenario_id, "agent": "pursuit", "seed": 5}
    first = client.post("/api/runs", json=payload, headers=auth_headers).json()
    second = client.post("/api/runs", json=payload, headers=auth_headers).json()
    assert first["frames"] == second["frames"]


def test_run_against_a_missing_scenario_is_404(client, auth_headers):
    response = client.post(
        "/api/runs",
        json={"scenario_id": 999999, "agent": "pursuit"},
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_qlearning_without_an_agent_id_is_422(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    response = client.post(
        "/api/runs",
        json={"scenario_id": scenario_id, "agent": "qlearning"},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_run_history_lists_the_run(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    client.post(
        "/api/runs",
        json={"scenario_id": scenario_id, "agent": "pursuit", "seed": 2},
        headers=auth_headers,
    )
    response = client.get("/api/runs?limit=5", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) >= 1


#  agents


def test_train_an_agent(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    response = client.post(
        "/api/agents/train",
        json={
            "name": "pytest-agent",
            "scenario_id": scenario_id,
            "episodes": 20,
            "steps_per_episode": 30,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["episodes"] == 20
    assert len(body["policy"]) > 0


def test_a_trained_agent_can_drive(client, auth_headers):
    scenario_id = open_road_id(client, auth_headers)
    trained = client.post(
        "/api/agents/train",
        json={
            "name": "pytest-driver",
            "scenario_id": scenario_id,
            "episodes": 20,
            "steps_per_episode": 30,
        },
        headers=auth_headers,
    ).json()

    response = client.post(
        "/api/runs",
        json={
            "scenario_id": scenario_id,
            "agent": "qlearning",
            "agent_id": trained["id"],
            "seed": 1,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    assert "safety_score" in response.json()["metrics"]


def test_agents_are_listed(client, auth_headers):
    response = client.get("/api/agents", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
