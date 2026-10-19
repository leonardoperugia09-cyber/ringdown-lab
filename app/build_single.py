"""Build app/overtone_explorer.html: a single-file copy of the Overtone Explorer.

The website (app/index.html + app/physics.js, the same files as the
overtone-explorer repo) loads the physics from a separate file. Some places,
like an embedded page, need one self-contained file instead. This script puts
physics.js inline and keeps only what goes inside <head> and <body>.

    python app/build_single.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    page = (HERE / "index.html").read_text(encoding="utf-8")
    physics = (HERE / "physics.js").read_text(encoding="utf-8")
    head = re.search(r"<head>\n(.*?)</head>", page, re.S).group(1)
    body = re.search(r"<body>\n(.*?)</body>", page, re.S).group(1)
    tag = '<script src="physics.js"></script>'
    assert body.count(tag) == 1
    body = body.replace(tag, "<script>\n" + physics + "</script>")
    head = "\n".join(line for line in head.splitlines() if 'name="viewport"' not in line)
    return head + "\n" + body


if __name__ == "__main__":
    out = HERE / "overtone_explorer.html"
    out.write_text(build(), encoding="utf-8", newline="\n")
    print("wrote", out.name)
