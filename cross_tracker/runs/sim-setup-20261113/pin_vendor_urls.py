"""Route resolved NVIDIA wheels explicitly; ordinary dependencies remain on PyPI."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
import json
import urllib.parse
import urllib.request

RUN = Path(__file__).resolve().parent
requirements = [line for line in (RUN / "sim-resolved-r3.txt").read_text().splitlines()
                if line and not line.startswith(("#", " "))]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href")


def resolve(requirement):
    name, version = requirement.split("==")
    index = f"https://pypi.nvidia.com/{name}/"
    with urllib.request.urlopen(index, timeout=30) as response:
        page = response.read().decode()
    parser = Links()
    parser.feed(page)
    matches = []
    for link in parser.urls:
        url = urllib.parse.urljoin(index, link)
        filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
        parts = filename.split("-")
        if (len(parts) == 5 and parts[0].replace("_", "-") == name
                and parts[1] == version and parts[2] in ("cp311", "py3")
                and parts[4].endswith("x86_64.whl")
                and ("manylinux_2_35" in parts[4] or parts[4] == "linux_x86_64.whl")):
            matches.append(url)
    assert len(matches) == 1, (requirement, matches)
    url = matches[0]
    assert "#sha256=" in url, url
    with urllib.request.urlopen(urllib.request.Request(url.split("#")[0], method="HEAD"), timeout=30) as response:
        size = response.headers.get("Content-Length")
    return name, {"version": version, "index": index, "url": url,
                  "bytes": int(size) if size else None}


vendor = [r for r in requirements if r.startswith(("isaacsim", "omniverse-kit=="))]
with ThreadPoolExecutor(max_workers=6) as pool:
    pins = dict(pool.map(resolve, vendor))
lines = [f"{r.split('==')[0]} @ {pins[r.split('==')[0]]['url']}"
         if r.split("==")[0] in pins else r for r in requirements]
(RUN / "sim-explicit.txt").write_text("\n".join(lines) + "\n")
(RUN / "nvidia-wheel-pins.json").write_text(json.dumps(pins, indent=2))
print("PINNED", len(pins), "VENDOR WHEELS; TOTAL REQUIREMENTS", len(lines),
      "VENDOR BYTES", sum(p["bytes"] or 0 for p in pins.values()), flush=True)
