"""Interrupted downloads must never append a full response to a partial file."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_assets import fetch
import requests


class DownloadTests(unittest.TestCase):
    def test_interrupted_response_keeps_received_bytes(self):
        payload = b"0123456789"
        record = {"bytes": 10, "sha256": hashlib.sha256(payload).hexdigest(),
                  "url": "https://huggingface.co/test"}
        class Response:
            status_code = 206
            def __init__(self, start):
                self.start = start
                self.headers = {"Content-Range": f"bytes {start}-9/10"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size):
                if self.start == 0:
                    yield payload[:4]
                    raise requests.exceptions.ChunkedEncodingError("interrupted")
                yield payload[4:]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("fetch_assets.requests.Session.get", side_effect=[Response(0), Response(4)]) as request, patch("fetch_assets.time.sleep"):
                fetch(("asset", record), Path(tmp), [], "https://huggingface.co")
            self.assertEqual(request.call_args_list[1].kwargs["headers"], {"Range": "bytes=4-9"})
            self.assertEqual((Path(tmp) / "asset").read_bytes(), payload)

    def test_resume_and_server_ignoring_range(self):
        payload = b"0123456789"
        record = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                  "url": "https://huggingface.co/test"}
        for status in (200, 206):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "asset.partial").write_bytes(payload[:4])
                class Response:
                    status_code = status
                    headers = {"Content-Range": "bytes 4-9/10"}
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def raise_for_status(self): pass
                    def iter_content(self, size):
                        yield payload[4:] if status == 206 else payload
                with patch("fetch_assets.requests.Session.get", return_value=Response()) as request:
                    fetch(("asset", record), root, [], "https://huggingface.co")
                self.assertEqual(request.call_args.kwargs["headers"], {"Range": "bytes=4-9"})
                self.assertEqual((root / "asset").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
