"""Command-line front end (same engine as the web UI).

  python cli.py run my_doc --script script.txt --narration narration.wav      # everything, end to end
  python cli.py new my_doc | align my_doc | build my_doc | preview my_doc | export my_doc [--video-only]
"""
import argparse
import logging
import sys

from core import builder
from core.project import Project


def prog(f, m=""):
    print(f"\r[{int(f * 100):3d}%] {m:<60}", end="", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stock-video documentary editor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("new", "align", "build", "preview", "export"):
        s = sub.add_parser(name)
        s.add_argument("project")
        if name == "export":
            s.add_argument("--video-only", action="store_true")
        if name == "build":
            s.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("project")
    r.add_argument("--script", required=True)
    r.add_argument("--narration", required=True)
    r.add_argument("--video-only", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        if a.cmd in ("new", "run"):
            try:
                p = Project.create(a.project)
            except FileExistsError:
                p = Project.open(a.project)
        else:
            p = Project.open(a.project)
        if a.cmd == "run":
            p.save_script(open(a.script, encoding="utf-8").read())
            p.import_narration(a.narration.replace("\\", "/").split("/")[-1], src_path=a.narration)
            print(builder.align_project(p, prog))
            print(builder.build_visuals(p, prog))
            out = builder.export_project(p, not a.video_only, prog)
            print("\nDone:", out.get("with_narration") or out["video_only"])
        elif a.cmd == "align":
            print(builder.align_project(p, prog))
        elif a.cmd == "build":
            print(builder.build_visuals(p, prog, force=a.force))
        elif a.cmd == "preview":
            print(builder.preview_project(p, prog)["preview"])
        elif a.cmd == "export":
            print(builder.export_project(p, not a.video_only, prog))
        elif a.cmd == "new":
            print(f"Created {p.dir}")
    except (builder.BuildError, FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
