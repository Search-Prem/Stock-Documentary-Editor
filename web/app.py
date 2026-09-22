"""Local web UI (Flask). Thin layer over core.builder; binds to 127.0.0.1 only."""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from core import builder
from core.config import get_config, redact
from core.project import DEFAULT_SETTINGS, Project
from core.render import RenderError, scene_preview
from core.util import MediaError, ffmpeg_available

log = logging.getLogger("stockdoc")
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


class Jobs:
    """One background job at a time (renders and downloads are heavy; sequential is kinder to 12 GB RAM)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def start(self, kind: str, fn) -> dict:
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise RuntimeError("Another task is still running. Wait for it to finish.")
            job = {"id": uuid.uuid4().hex[:10], "kind": kind, "status": "running", "progress": 0.0,
                   "message": "Starting", "result": None, "error": "", "started": time.time()}
            self.jobs[job["id"]] = job
            if len(self.jobs) > 30:
                for k in sorted(self.jobs, key=lambda k: self.jobs[k]["started"])[:10]:
                    if self.jobs[k]["status"] != "running":
                        self.jobs.pop(k, None)

        def runner():
            def progress(f, m=""):
                job["progress"], job["message"] = round(float(f), 3), m
            try:
                job["result"] = fn(progress)
                job["status"], job["progress"], job["message"] = "done", 1.0, "Done"
            except (builder.BuildError, RenderError, MediaError) as e:
                job["status"], job["error"] = "error", redact(e)
            except Exception as e:  # never leak a traceback/keys to the UI; log it locally
                log.error("job %s failed:\n%s", kind, redact(traceback.format_exc()))
                job["status"], job["error"] = "error", redact(f"Unexpected error: {type(e).__name__}: {e}")[:300]

        threading.Thread(target=runner, daemon=True).start()
        return job


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3
    jobs = Jobs()

    @app.before_request
    def guard():
        host = (request.host or "").split(":")[0]
        if host not in ALLOWED_HOSTS:                      # blocks DNS-rebinding style access
            abort(403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("Origin")
            if origin and origin.split("//")[-1].split(":")[0] not in ALLOWED_HOSTS:
                abort(403)

    def proj(name: str) -> Project:
        try:
            return Project.open(name)
        except (ValueError, FileNotFoundError):
            abort(404)

    def err(msg: str, code: int = 400):
        return jsonify({"error": redact(msg)}), code

    def run(kind, fn):
        try:
            return jsonify(jobs.start(kind, fn))
        except RuntimeError as e:
            return err(str(e), 409)

    # ---------------------------------------------------------------- pages / status
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        cfg = get_config()
        return jsonify({"ffmpeg": ffmpeg_available(),
                        "providers": {"pexels": bool(cfg.pexels_key), "pixabay": bool(cfg.pixabay_key),
                                      "unsplash": bool(cfg.unsplash_key)},
                        "whisper": importlib.util.find_spec("faster_whisper") is not None})

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = jobs.jobs.get(jid)
        return jsonify(j) if j else err("Unknown task", 404)

    # ---------------------------------------------------------------- projects
    @app.get("/api/projects")
    def projects():
        return jsonify(Project.list_names())

    @app.post("/api/projects")
    def create_project():
        name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
        try:
            Project.create(name)
        except ValueError as e:
            return err(str(e))
        except FileExistsError as e:
            return err(str(e), 409)
        return jsonify({"name": name})

    @app.get("/api/projects/<name>")
    def project_state(name):
        p = proj(name)
        tl = p.load_timeline()
        for sc in tl["scenes"]:
            for c in sc["clips"]:
                t = c.get("thumb") or ""
                c["thumb_url"] = t if t.startswith("http") else (f"/files/{name}/{t}" if t else "")
        meta = p.meta()
        return jsonify({
            "name": name, "settings": p.settings, "defaults": DEFAULT_SETTINGS, "script": p.script_text(),
            "sentences": len(p.sentences()), "narration": meta.get("narration"), "timeline": tl,
            "files": {"preview": (p.preview_dir / "preview.mp4").exists(),
                      "video_only": (p.final_dir / "documentary_video.mp4").exists(),
                      "with_narration": (p.final_dir / "documentary_with_narration.mp4").exists(),
                      "srt": (p.final_dir / "subtitles.srt").exists() or p.srt_path.exists()}})

    @app.put("/api/projects/<name>/script")
    def save_script(name):
        p = proj(name)
        s = p.save_script((request.get_json(silent=True) or {}).get("text", ""))
        return jsonify({"sentences": len(s)})

    @app.post("/api/projects/<name>/narration")
    def upload_narration(name):
        p = proj(name)
        f = request.files.get("file")
        if not f or not f.filename:
            return err("Choose a WAV or MP3 file")
        try:
            return jsonify(p.import_narration(secure_filename(f.filename), stream=f.stream))
        except MediaError as e:
            return err(str(e))

    @app.put("/api/projects/<name>/settings")
    def save_settings(name):
        p = proj(name)
        try:
            return jsonify(p.update_settings(request.get_json(silent=True) or {}))
        except (ValueError, TypeError):
            return err("Invalid setting value")

    # ---------------------------------------------------------------- long tasks
    @app.post("/api/projects/<name>/align")
    def align(name):
        p = proj(name)
        return run("align", lambda pr: builder.align_project(p, pr))

    @app.post("/api/projects/<name>/build")
    def build(name):
        p = proj(name)
        force = bool((request.get_json(silent=True) or {}).get("force"))
        return run("build", lambda pr: builder.build_visuals(p, pr, force=force))

    @app.post("/api/projects/<name>/preview")
    def preview(name):
        p = proj(name)
        return run("preview", lambda pr: builder.preview_project(p, pr))

    @app.post("/api/projects/<name>/export")
    def export(name):
        p = proj(name)
        with_n = bool((request.get_json(silent=True) or {}).get("with_narration", True))
        return run("export", lambda pr: builder.export_project(p, with_n, pr))

    # ---------------------------------------------------------------- per-scene actions
    @app.post("/api/projects/<name>/scenes/<int:sid>/search_again")
    def search_again(name, sid):
        p = proj(name)
        return run("search_again", lambda pr: builder.search_again(p, sid, pr))

    @app.put("/api/projects/<name>/scenes/<int:sid>/queries")
    def queries(name, sid):
        p = proj(name)
        try:
            builder.set_queries(p, sid, (request.get_json(silent=True) or {}).get("queries", []))
        except builder.BuildError as e:
            return err(str(e), 404)
        return jsonify({"ok": True})

    @app.get("/api/projects/<name>/scenes/<int:sid>/candidates")
    def candidates(name, sid):
        p = proj(name)
        try:
            return jsonify(builder.list_candidates(p, sid, request.args.get("q") or None))
        except builder.BuildError as e:
            return err(str(e))

    @app.post("/api/projects/<name>/scenes/<int:sid>/replace")
    def replace(name, sid):
        p = proj(name)
        d = request.get_json(silent=True) or {}
        idx = d.get("clip_index")
        try:
            if d.get("path"):
                return jsonify(builder.replace_with_local(p, sid, idx, d["path"]))
            if d.get("candidate"):
                return jsonify(builder.replace_with_candidate(p, sid, idx, d["candidate"]))
        except builder.BuildError as e:
            return err(str(e))
        return err("Nothing to replace with")

    @app.post("/api/projects/<name>/scenes/<int:sid>/replace_upload")
    def replace_upload(name, sid):
        p = proj(name)
        f = request.files.get("file")
        if not f or not f.filename:
            return err("Choose a file")
        dest = p.scene_dir(sid) / ("upload_" + secure_filename(f.filename))
        f.save(dest)
        idx = request.form.get("clip_index")
        try:
            return jsonify(builder.replace_with_local(p, sid, int(idx) if idx not in (None, "", "null") else None, str(dest)))
        except builder.BuildError as e:
            dest.unlink(missing_ok=True)
            return err(str(e))

    @app.post("/api/projects/<name>/scenes/<int:sid>/preview")
    def scene_prev(name, sid):
        p = proj(name)

        def fn(pr):
            pr(0.1, f"Rendering scene {sid}")
            path = scene_preview(p, p.load_timeline(), sid)
            return {"url": f"/files/{name}/preview/{path.name}?t={int(time.time())}"}
        return run("scene_preview", fn)

    # ---------------------------------------------------------------- files
    @app.get("/files/<name>/<path:rel>")
    def files(name, rel):
        p = proj(name)
        return send_from_directory(p.dir, rel, conditional=True)

    return app
