"""Tests for the FastAPI gateway.

Covers all routing, validation, error mapping, seed handling, and the worker-call
helpers — without invoking real workers or any GPU code. Worker HTTP calls are
intercepted by `respx`, so these tests run in milliseconds.

Doesn't cover: actual model inference (FLUX / TRELLIS), image_gen worker, gen_3d
worker. Those require a GPU and the full setup.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from orchestrator.server import GEN_3D_URL, IMAGE_GEN_URL, SEED_MAX, app

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake png content"
GLB_BYTES = b"glTF\x02\x00\x00\x00fake glb content"


@pytest.fixture
def client():
    """FastAPI TestClient with lifespan triggered (so the httpx.AsyncClient is created)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        with respx.mock:
            respx.get(f"{IMAGE_GEN_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            respx.get(f"{GEN_3D_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            r = client.get("/health")
        assert r.status_code == 200

    def test_all_workers_up(self, client):
        with respx.mock:
            respx.get(f"{IMAGE_GEN_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            respx.get(f"{GEN_3D_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            r = client.get("/health")
        assert r.json() == {"gateway": True, "image_gen": True, "gen_3d": True}

    def test_workers_unreachable(self, client):
        with respx.mock:
            respx.get(f"{IMAGE_GEN_URL}/health").mock(side_effect=httpx.ConnectError("refused"))
            respx.get(f"{GEN_3D_URL}/health").mock(side_effect=httpx.ConnectError("refused"))
            r = client.get("/health")
        assert r.json() == {"gateway": True, "image_gen": False, "gen_3d": False}

    def test_partial_availability(self, client):
        with respx.mock:
            respx.get(f"{IMAGE_GEN_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            respx.get(f"{GEN_3D_URL}/health").mock(side_effect=httpx.ConnectError("refused"))
            r = client.get("/health")
        assert r.json() == {"gateway": True, "image_gen": True, "gen_3d": False}

    def test_worker_returns_ok_false(self, client):
        with respx.mock:
            respx.get(f"{IMAGE_GEN_URL}/health").mock(return_value=httpx.Response(200, json={"ok": False}))
            respx.get(f"{GEN_3D_URL}/health").mock(return_value=httpx.Response(200, json={"ok": True}))
            r = client.get("/health")
        assert r.json()["image_gen"] is False
        assert r.json()["gen_3d"] is True


# ---------------------------------------------------------------------------
# /image
# ---------------------------------------------------------------------------


class TestImage:
    def test_empty_prompt_rejected(self, client):
        r = client.post("/image", json={"prompt": ""})
        assert r.status_code == 422

    def test_missing_prompt_rejected(self, client):
        r = client.post("/image", json={})
        assert r.status_code == 422

    def test_returns_png_bytes(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            r = client.post("/image", json={"prompt": "a barrel"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content == PNG_BYTES

    def test_seed_header_in_valid_range(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            r = client.post("/image", json={"prompt": "a barrel"})
        seed = int(r.headers["X-Seed"])
        assert 0 <= seed <= SEED_MAX

    def test_prompt_and_seed_passed_to_worker(self, client):
        with respx.mock:
            route = respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            r = client.post("/image", json={"prompt": "a barrel"})
        body = json.loads(route.calls[0].request.content)
        assert body["prompt"] == "a barrel"
        assert body["seed"] == int(r.headers["X-Seed"])

    def test_worker_unreachable_returns_503(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(side_effect=httpx.ConnectError("refused"))
            r = client.post("/image", json={"prompt": "a barrel"})
        assert r.status_code == 503
        assert "image_gen" in r.json()["detail"]

    def test_worker_timeout_returns_504(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(side_effect=httpx.TimeoutException("timed out"))
            r = client.post("/image", json={"prompt": "a barrel"})
        assert r.status_code == 504
        assert "image_gen" in r.json()["detail"]

    def test_worker_500_passes_through(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(500, text="model OOM"))
            r = client.post("/image", json={"prompt": "a barrel"})
        assert r.status_code == 500
        assert "model OOM" in r.json()["detail"]

    def test_worker_422_passes_through(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(422, text="bad prompt"))
            r = client.post("/image", json={"prompt": "a barrel"})
        assert r.status_code == 422
        assert "bad prompt" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /3d
# ---------------------------------------------------------------------------


class Test3D:
    def test_missing_image_rejected(self, client):
        r = client.post("/3d")
        assert r.status_code == 422

    def test_returns_glb_bytes(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert r.status_code == 200
        assert r.headers["content-type"] == "model/gltf-binary"
        assert r.content == GLB_BYTES

    def test_seed_header_in_valid_range(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        seed = int(r.headers["X-Seed"])
        assert 0 <= seed <= SEED_MAX

    def test_image_bytes_forwarded_unchanged(self, client):
        with respx.mock:
            route = respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert route.calls[0].request.content == PNG_BYTES

    def test_seed_passed_as_query_param(self, client):
        with respx.mock:
            route = respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert int(route.calls[0].request.url.params["seed"]) == int(r.headers["X-Seed"])

    def test_worker_unreachable_returns_503(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(side_effect=httpx.ConnectError("refused"))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert r.status_code == 503
        assert "gen_3d" in r.json()["detail"]

    def test_worker_timeout_returns_504(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(side_effect=httpx.TimeoutException("timed out"))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert r.status_code == 504

    def test_worker_500_passes_through(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(500, text="OOM"))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert r.status_code == 500

    def test_worker_422_passes_through(self, client):
        with respx.mock:
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(422, text="bad image"))
            r = client.post("/3d", files={"image": ("img.png", PNG_BYTES, "image/png")})
        assert r.status_code == 422
        assert "bad image" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /text-to-3d
# ---------------------------------------------------------------------------


class TestTextTo3D:
    def test_empty_prompt_rejected(self, client):
        r = client.post("/text-to-3d", json={"prompt": ""})
        assert r.status_code == 422

    def test_full_pipeline_returns_glb(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "model/gltf-binary"
        assert r.content == GLB_BYTES

    def test_separate_seed_headers(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        image_seed = int(r.headers["X-Image-Seed"])
        threed_seed = int(r.headers["X-3d-Seed"])
        assert 0 <= image_seed <= SEED_MAX
        assert 0 <= threed_seed <= SEED_MAX

    def test_image_bytes_piped_to_3d_worker(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            route_3d = respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            client.post("/text-to-3d", json={"prompt": "a barrel"})
        # Bytes flowing through must be the exact bytes image_gen produced.
        assert route_3d.calls[0].request.content == PNG_BYTES

    def test_3d_seed_matches_query_param(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            route_3d = respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert int(route_3d.calls[0].request.url.params["seed"]) == int(r.headers["X-3d-Seed"])

    def test_image_seed_matches_image_worker_body(self, client):
        with respx.mock:
            route_img = respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        body = json.loads(route_img.calls[0].request.content)
        assert body["seed"] == int(r.headers["X-Image-Seed"])
        assert body["prompt"] == "a barrel"

    def test_image_gen_failure_short_circuits(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(500, text="OOM"))
            route_3d = respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        # gen_3d must not be called if image_gen failed.
        assert r.status_code == 500
        assert route_3d.call_count == 0

    def test_3d_failure_after_image_success(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(500, text="OOM"))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert r.status_code == 500
        assert "gen_3d" in r.json()["detail"]

    def test_3d_422_passes_through(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(422, text="bad image"))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert r.status_code == 422
        assert "gen_3d" in r.json()["detail"]

    def test_image_gen_unreachable_returns_503(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(side_effect=httpx.ConnectError("refused"))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert r.status_code == 503

    def test_image_gen_timeout_returns_504(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(side_effect=httpx.TimeoutException("slow"))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        assert r.status_code == 504


# ---------------------------------------------------------------------------
# Cross-endpoint properties
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_seeds_are_randomized_per_call(self, client):
        """Two calls to /image should (almost certainly) produce different seeds.

        Probability of collision is 1 / SEED_MAX ≈ 5e-10. Acceptable flake rate.
        """
        seeds = set()
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            for _ in range(5):
                r = client.post("/image", json={"prompt": "a barrel"})
                seeds.add(int(r.headers["X-Seed"]))
        assert len(seeds) == 5  # all different

    def test_text_to_3d_uses_two_independent_seeds(self, client):
        with respx.mock:
            respx.post(f"{IMAGE_GEN_URL}/generate").mock(return_value=httpx.Response(200, content=PNG_BYTES))
            respx.post(f"{GEN_3D_URL}/generate").mock(return_value=httpx.Response(200, content=GLB_BYTES))
            r = client.post("/text-to-3d", json={"prompt": "a barrel"})
        # Probability of collision is negligible.
        assert r.headers["X-Image-Seed"] != r.headers["X-3d-Seed"]
