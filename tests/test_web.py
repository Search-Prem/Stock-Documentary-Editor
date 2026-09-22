"""Drive the Flask app exactly as the browser UI does (background jobs + polling)."""
import io, os, time
import pytest

from core import cache as cache_mod
from core.providers import base as prov_base
from core.providers import pexels as pexels_mod
from tests.fixtures import SAMPLE_SCRIPT, make_color_clip, make_narration
from tests.mock_pexels import API_KEY, MockPexels
from core.script_split import split_sentences

CATALOG = [(1, "yellowing-houseplant-leaves-close-up", "0xff0000", 14), (2, "woman-watering-indoor-plant", "0x00ff00", 12),
           (3, "plant-roots-in-wet-soil", "0x0000ff", 10), (4, "overwatered-houseplant-drooping-leaves", "0xffff00", 9),
           (5, "hands-touching-soil-in-plant-pot", "0xff00ff", 11), (6, "sunlight-through-green-leaves", "0x00ffff", 13),
           (7, "wilting-plant-dry-soil", "0xff8000", 8), (8, "watering-can-pouring-water-on-plant", "0x8000ff", 10)]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    cat = []
    for i, slug, col, dur in CATALOG:
        f = root / f"c{i}.mp4"; make_color_clip(f, col, dur, 1920, 1080, 30)
        cat.append({"id": i, "slug": slug, "path": str(f), "duration": dur, "width": 1920, "height": 1080})
    mock = MockPexels(cat).start()
    saved = {k: os.environ.get(k) for k in ("PEXELS_API_KEY", "PEXELS_BASE_URL", "PROJECTS_DIR", "CACHE_DIR")}
    os.environ.update({"PEXELS_API_KEY": API_KEY, "PEXELS_BASE_URL": f"http://127.0.0.1:{mock.port}",
                       "PROJECTS_DIR": str(root / "projects"), "CACHE_DIR": str(root / "cache")})
    prov_base.BACKOFF = 0; cache_mod.BACKOFF = 0; pexels_mod.MIN_INTERVAL = 0
    from web.app import create_app
    app = create_app(); app.testing = True
    c = app.test_client(); c.root = root; c.mock = mock
    yield c
    mock.stop()
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def wait(client, resp, timeout=240):
    assert resp.status_code == 200, resp.get_json()
    jid = resp.get_json()["id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["status"] != "running":
            return j
        time.sleep(0.3)
    raise AssertionError("job timed out")


def test_full_ui_flow(client):
    assert client.get("/").status_code == 200
    st = client.get("/api/status").get_json()
    assert st["ffmpeg"] and st["providers"]["pexels"] and "key" not in str(st).lower().replace("pexels", "")
    assert client.post("/api/projects", json={"name": "demo"}).status_code == 200
    assert client.post("/api/projects", json={"name": "demo"}).status_code == 409          # duplicate
    assert client.post("/api/projects", json={"name": "../x"}).status_code == 400          # traversal rejected
    sents = split_sentences(SAMPLE_SCRIPT)
    assert client.put("/api/projects/demo/script", json={"text": SAMPLE_SCRIPT}).get_json()["sentences"] == len(sents)
    client.put("/api/projects/demo/settings", json={"align_backend": "silence", "preview_width": 640, "preview_height": 360})
    dur, truth = make_narration(sents, client.root / "n.wav")
    bad = client.post("/api/projects/demo/narration", data={"file": (io.BytesIO(b"junk"), "x.wav")}, content_type="multipart/form-data")
    assert bad.status_code == 400                                                           # junk audio rejected, not stored
    ok = client.post("/api/projects/demo/narration", data={"file": (open(client.root / "n.wav", "rb"), "narration.wav")}, content_type="multipart/form-data")
    assert ok.status_code == 200 and abs(ok.get_json()["duration"] - dur) < 0.01

    j = wait(client, client.post("/api/projects/demo/align"))
    assert j["status"] == "done" and j["result"]["issues"] == [] and j["result"]["backend"] == "silence"
    j = wait(client, client.post("/api/projects/demo/build", json={}))
    assert j["status"] == "done" and j["result"]["paused"] == "" and j["result"]["needs_attention"] == 0, j

    state = client.get("/api/projects/demo").get_json()
    scenes = state["timeline"]["scenes"]
    assert all(s["clips"] and s["status"] == "ok" for s in scenes)
    assert all(c["thumb_url"].startswith("http") for s in scenes for c in s["clips"])
    assert "PEXELS" not in str(state) and API_KEY not in str(state)

    # candidates picker + replace with a stock candidate
    cands = client.get("/api/projects/demo/scenes/2/candidates").get_json()
    assert cands and all(c["kind"] == "video" for c in cands)
    r = client.post("/api/projects/demo/scenes/2/replace", json={"clip_index": 0, "candidate": cands[0]})
    assert r.status_code == 200 and client.get("/api/projects/demo").get_json()["timeline"]["scenes"][1]["clips"][0]["pinned"]

    # replace with an uploaded file (whole scene) and with a bad file
    mine = make_color_clip(client.root / "mine.mp4", "0x123456", 20, 1280, 720, 30)
    r = client.post("/api/projects/demo/scenes/3/replace_upload", data={"file": (open(mine, "rb"), "mine.mp4"), "clip_index": ""}, content_type="multipart/form-data")
    assert r.status_code == 200
    r = client.post("/api/projects/demo/scenes/3/replace_upload", data={"file": (io.BytesIO(b"nope"), "bad.mp4"), "clip_index": "0"}, content_type="multipart/form-data")
    assert r.status_code == 400

    # per-scene preview + full preview + export, all through jobs
    j = wait(client, client.post("/api/projects/demo/scenes/3/preview"))
    assert j["status"] == "done" and client.get(j["result"]["url"]).status_code == 200
    j = wait(client, client.post("/api/projects/demo/preview"))
    assert j["status"] == "done" and j["result"]["frames_actual"] == j["result"]["frames_expected"]
    j = wait(client, client.post("/api/projects/demo/export", json={"with_narration": True}))
    assert j["status"] == "done", j
    assert abs(j["result"]["audio_duration"] - dur) < 0.06
    rng = client.get("/files/demo/final/documentary_with_narration.mp4", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206                                                             # video seeking works in the browser


def test_only_one_job_at_a_time(client):
    r1 = client.post("/api/projects/demo/preview")
    r2 = client.post("/api/projects/demo/preview")
    assert r1.status_code == 200 and r2.status_code == 409
    wait(client, r1)


def test_security_guards(client):
    assert client.get("/api/projects", headers={"Host": "evil.example.com"}).status_code == 403
    assert client.post("/api/projects", json={"name": "zz"}, headers={"Origin": "http://evil.example.com"}).status_code == 403
    assert client.get("/files/demo/../../etc/passwd").status_code in (404, 400)
    assert client.get("/files/demo/%2e%2e/%2e%2e/etc/passwd").status_code in (404, 400)
    assert client.get("/api/projects/nonexistent").status_code == 404
    assert client.get("/api/projects/..%2f..").status_code == 404
