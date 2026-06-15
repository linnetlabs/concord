"""Content extraction -- pull the meaningful, contradiction-worthy units out of each
file type, leaving the syntax behind.

The semantic index should hold what a reader could actually contradict -- prose,
headings, comments, string literals, config values, gating constants -- not braces,
selectors, tags, or operators. Extractors are registered by extension, so adding a
language (Rust, Go, ...) is a single function and nothing else changes.

The leak-lint does NOT use this module: it scans raw text, because a codename can
hide in an HTML attribute or a minified string that extraction would strip.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Callable, Dict, List, Tuple

Block = Tuple[str, int, int]  # (text, start_line, end_line)

_REGISTRY: Dict[str, Callable[[str], List[Block]]] = {}


def register(*exts: str):
    """Register an extractor for one or more file extensions (with the leading dot)."""
    def deco(fn: Callable[[str], List[Block]]):
        for e in exts:
            _REGISTRY[e.lower()] = fn
        return fn
    return deco


def supported_extensions() -> List[str]:
    """The extensions Concord semantically indexes, sorted -- for the CLI/UI to show."""
    return sorted(_REGISTRY)


# -- quality gate --------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z]{2,}")
_NUM = re.compile(r"\d")


def keep(text: str) -> bool:
    """Worth indexing if it reads like prose (>=3 words, mostly letters) OR pairs an
    identifier-ish word with a number -- a price/threshold/version/gate that could
    contradict another. Drops stubs, punctuation, and bare syntax (`}`, `---`, `;`)."""
    t = text.strip()
    if not t:
        return False
    words = _WORD.findall(t)
    if len(words) >= 3:
        letters = sum(c.isalpha() or c.isspace() for c in t)
        return letters / len(t) >= 0.4
    return bool(words) and bool(_NUM.search(t))


# -- prose: markdown / text / rst (paragraph blocks on blank lines) -------------
@register(".md", ".markdown", ".txt", ".rst", ".mdx")
def _paragraphs(text: str) -> List[Block]:
    blocks: List[Block] = []
    buf: List[str] = []
    start = 1
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip() == "":
            if buf:
                blocks.append(("\n".join(buf), start, i - 1))
                buf = []
            start = i + 1
        else:
            if not buf:
                start = i
            buf.append(line)
    if buf:
        blocks.append(("\n".join(buf), start, start + len(buf) - 1))
    return blocks


# -- code/config: keep the human-meaningful units -- comments, string literals, and
#    named numeric constants (MIN_N = 8, "min_n": 5, timeout = 30) -- and drop the
#    logic, calls, braces and embedded markup. Modular: a language that fits this
#    shape (most do) just adds its extension here; a quirkier one gets its own fn. -
_COMMENT = re.compile(r"(?:^|\s)(?:#|//)\s?(\S.*)$")          # trailing # or // comment
_STR = re.compile(r"""(['"])((?:\\.|(?!\1).){4,})\1""")       # quoted string, >=4 chars
_CONST = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")                # NAMED_CONSTANT token
_JSON_NUM = re.compile(r"""["'][\w.\-]+["']\s*:\s*["']?\$?\d""")  # "key": 5 (config value)


@register(".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".css", ".scss", ".r")
def _code(text: str) -> List[Block]:
    """Keep comments, string literals, and named numeric constants / config values;
    drop logic, calls, locals (start = 1) and braces."""
    out: List[Block] = []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s:
            continue
        # a named gating/config/price constant: MIN_N = 8, "min_n": 5, MAX_SEATS: u32 = 50
        if (_CONST.search(ln) and _NUM.search(ln)) or _JSON_NUM.search(ln):
            out.append((s, i, i))
            continue
        parts: List[str] = []
        m = _COMMENT.search(ln)
        if m:
            parts.append(m.group(1).strip())
        parts.extend(sm.group(2).strip() for sm in _STR.finditer(ln))
        parts = [p for p in dict.fromkeys(parts) if p]
        if parts:
            out.append((", ".join(parts), i, i))
    return out


# -- html: visible text only; skip <script>/<style>/<head> ---------------------
_HTML_SKIP = {"script", "style", "head", "noscript", "svg", "template"}
_HTML_BLOCK = {"p", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
               "section", "article", "tr", "blockquote", "figcaption", "caption", "button"}


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: List[Block] = []
        self._buf: List[str] = []
        self._line = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _HTML_SKIP:
            self._skip += 1
        elif tag in _HTML_BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _HTML_SKIP and self._skip:
            self._skip -= 1
        elif tag in _HTML_BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._skip or not data.strip():
            return
        if self._line is None:
            self._line = self.getpos()[0]
        self._buf.append(" ".join(data.split()))

    def _flush(self):
        if self._buf:
            self.blocks.append((" ".join(self._buf).strip(), self._line or 1, self.getpos()[0]))
            self._buf, self._line = [], None


@register(".html", ".htm")
def _html(text: str) -> List[Block]:
    p = _HTMLText()
    try:
        p.feed(text)
        p._flush()
    except Exception:
        return []
    return p.blocks


def extract(text: str, ext: str) -> List[Block]:
    """Meaningful, gate-passed blocks for one file's text and extension.

    Returns [] for an unregistered extension (it is not semantically indexed; the
    leak-lint still scans it raw)."""
    fn = _REGISTRY.get(ext.lower())
    if fn is None:
        return []
    return [b for b in fn(text) if keep(b[0])]
