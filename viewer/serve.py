"""Serve the viewer locally, with HTTP Range so the 5 GB match video can be seeked.

    python viewer/serve.py                                  # http://localhost:8000/viewer/
    python viewer/serve.py --data-dir /elsewhere --port 8001

The repository root is served as-is (so the viewer can read evaluation/gt and evaluation/results),
and the first .mp4 in <data-dir>/tactical is exposed as /video/tactical.mp4, whatever its name.
Python's built-in http.server has no Range support, and without it a browser cannot seek a video.
"""

import argparse
import glob
import http.server
import os
import re
import shutil
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
CHUNK = 1 << 20
VIDEO_ROUTE = "/video/tactical.mp4"


def find_video(data_dir):
    videos = sorted(glob.glob(os.path.join(data_dir, "tactical", "*.mp4")))
    return videos[0] if videos else None


class Handler(http.server.SimpleHTTPRequestHandler):
    video_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def translate_path(self, path):
        if path.split("?")[0] == VIDEO_ROUTE and self.video_path:
            return self.video_path
        return super().translate_path(path)

    def send_head(self):
        path = self.translate_path(self.path)
        range_header = self.headers.get("Range")
        if not range_header or os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()

        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        size = os.path.getsize(path)
        if not match:
            return super().send_head()
        first, last = match.groups()
        if first == "":                       # suffix range: the last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            start, end = int(first), int(last) if last else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        handle = open(path, "rb")
        handle.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return handle

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return shutil.copyfileobj(source, outputfile)
        try:
            while remaining > 0:
                data = source.read(min(CHUNK, remaining))
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass                              # the browser cancelled a range after a seek
        finally:
            self._remaining = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Handler.video_path = find_video(args.data_dir)
    print("video:", Handler.video_path or "none found -- use 'Abrir vídeo' in the page")
    url = "http://localhost:{}/viewer/".format(args.port)
    print("serving", ROOT, "at", url)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
