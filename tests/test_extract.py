"""Extraction: meaningful units in, syntax/stubs out, across file types."""
from concordai import extract


def texts(blocks):
    return [b[0] for b in blocks]


def test_quality_gate_drops_stubs_keeps_prose_and_values():
    assert not extract.keep("}")
    assert not extract.keep("    }")
    assert not extract.keep("---")
    assert not extract.keep(";")
    assert extract.keep("The Starter plan costs twenty-nine dollars a seat.")
    # short line rescued because it carries an identifier + a number (a gate/price)
    assert extract.keep("MIN_RESPONDENTS = 8")
    assert extract.keep('"min_n": 5')


def test_code_lines_keep_constants_and_strings_drop_syntax():
    src = (
        "import os\n"
        "MIN_RESPONDENTS = 8  # anonymity floor\n"
        "def f():\n"
        "    return {}\n"
        "}\n"
        'BRAND = "Welcome to Acme Analytics"\n'
    )
    out = texts(extract.extract(src, ".py"))
    assert any("MIN_RESPONDENTS = 8" in t for t in out)
    assert any("Welcome to Acme Analytics" in t for t in out)
    assert "}" not in out and "return {}" not in out  # pure syntax dropped


def test_json_values_kept_braces_dropped():
    src = '{\n  "price": "$29/seat",\n  "min_n": 8\n}\n'
    out = texts(extract.extract(src, ".json"))
    assert any("price" in t and "$29" in t for t in out)
    assert any("min_n" in t and "8" in t for t in out)
    assert "{" not in out and "}" not in out


def test_js_ts_strings_and_gates():
    src = "const MIN_N = 8;\nconsole.log('totally clean syntax');\nfunction noop(){}\n"
    out = texts(extract.extract(src, ".ts"))
    assert any("MIN_N = 8" in t for t in out)
    assert not any(t == "function noop(){}" for t in out)  # no words+num, dropped


def test_html_extracts_visible_text_skips_script_style():
    html = (
        "<!doctype html><html><head><title>x</title>"
        "<style>.a{color:#fff;padding:12px}</style></head>"
        "<body><h1>Pricing for teams</h1>"
        "<p>The plan is $39 per seat per month.</p>"
        "<script>const sneaky = 'do not index this code';</script>"
        "</body></html>"
    )
    out = texts(extract.extract(html, ".html"))
    assert any("Pricing for teams" in t for t in out)
    assert any("$39 per seat" in t for t in out)
    assert not any("sneaky" in t or "do not index" in t for t in out)  # script skipped
    assert not any("color:#fff" in t or "padding" in t for t in out)   # style skipped


def test_html_line_numbers_preserved():
    html = "<html><body>\n\n<h1>Heading here now</h1>\n<p>Body text follows along.</p>\n</body></html>"
    blocks = extract.extract(html, ".html")
    # the <h1> sits on line 3
    h1 = next(b for b in blocks if "Heading" in b[0])
    assert h1[1] == 3


def test_registry_is_extensible():
    exts = extract.supported_extensions()
    for e in (".md", ".py", ".js", ".ts", ".json", ".css", ".html", ".r"):
        assert e in exts

    @extract.register(".rs")
    def _rust(text):
        return [(ln.strip(), i, i) for i, ln in enumerate(text.splitlines(), 1) if ln.strip()]

    out = texts(extract.extract('const MAX_SEATS: u32 = 50;\n', ".rs"))
    assert any("MAX_SEATS" in t and "50" in t for t in out)


def test_unregistered_extension_not_indexed():
    assert extract.extract("binary junk \x00\x01", ".bin") == []
