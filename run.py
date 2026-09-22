"""Start the local editor:  python run.py   (opens http://127.0.0.1:5000)"""
import logging
import threading
import webbrowser

from web.app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app = create_app()
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
