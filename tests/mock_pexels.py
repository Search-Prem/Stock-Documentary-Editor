"""A local HTTP server that mimics the Pexels API shape (as documented) so the real provider/cache/download
code runs end to end. NOT a proof that live Pexels behaves identically - see README 'What was verified'."""
import json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.planner import stem

API_KEY = "test-key-123456"


class MockPexels:
    def __init__(self, catalog):
        """catalog: list of dicts {id, slug, path, duration, width, height, fps}"""
        self.catalog = catalog
        self.requests = {"search": 0, "photos": 0, "download": 0}
        self.fail_mode = None          # None | "429" | "500"
        self.remaining = 190
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, body, ctype="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path.startswith("/files/"):
                    outer.requests["download"] += 1
                    name = u.path.split("/")[-1]
                    item = next((c for c in outer.catalog if Path(c["path"]).name == name), None)
                    if not item:
                        return self._send(404, b"nope")
                    data = Path(item["path"]).read_bytes()
                    return self._send(200, data, "video/mp4")
                if self.headers.get("Authorization") != API_KEY:
                    return self._send(401, b'{"error":"bad key"}')
                if u.path == "/videos/search":
                    outer.requests["search"] += 1
                    if outer.fail_mode == "429":
                        return self._send(429, b'{"error":"rate"}', headers={"X-Ratelimit-Remaining": "0"})
                    if outer.fail_mode == "500":
                        return self._send(500, b"boom")
                    outer.remaining -= 1
                    query = q.get("query", [""])[0]
                    per = int(q.get("per_page", ["15"])[0]); page = int(q.get("page", ["1"])[0])
                    qs = {stem(w) for w in re.findall(r"[a-z]+", query.lower())}
                    scored = []
                    for c in outer.catalog:
                        sl = {stem(w) for w in c["slug"].split("-")}
                        scored.append((-len(qs & sl), c["id"], c))
                    scored.sort(key=lambda t: (t[0], t[1]))
                    part = [c for _, _, c in scored][(page - 1) * per: page * per]
                    videos = []
                    for c in part:
                        link = f"http://127.0.0.1:{outer.port}/files/{Path(c['path']).name}"
                        videos.append({"id": c["id"], "width": c["width"], "height": c["height"], "duration": c["duration"],
                                       "url": f"https://www.pexels.com/video/{c['slug']}-{c['id']}/",
                                       "image": f"https://images.pexels.com/videos/{c['id']}/thumb.jpeg",
                                       "video_files": [
                                           {"id": 1, "quality": "sd", "file_type": "video/mp4", "width": 640, "height": 360, "fps": 30.0, "link": link},
                                           {"id": 2, "quality": "hd", "file_type": "video/mp4", "width": max(1920, c["width"]) if c["width"] >= c["height"] else c["width"], "height": max(1080, c["height"]) if c["width"] >= c["height"] else c["height"], "fps": 30.0, "link": link},
                                           {"id": 3, "quality": "uhd", "file_type": "video/mp4", "width": 3840, "height": 2160, "fps": 30.0, "link": link + "?uhd=1"}]})
                    return self._send(200, json.dumps({"page": page, "per_page": per, "videos": videos}).encode(),
                                      headers={"X-Ratelimit-Remaining": str(outer.remaining), "X-Ratelimit-Limit": "200"})
                if u.path == "/v1/search":
                    outer.requests["photos"] += 1
                    return self._send(200, json.dumps({"photos": []}).encode(), headers={"X-Ratelimit-Remaining": str(outer.remaining)})
                self._send(404, b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start(); return self

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()
