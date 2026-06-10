"""concord report — a self-contained, shareable consistency report (one HTML file).

Bundles the leak lint + the contradiction radar (LLM-verified if a key is set) into a
premium dark page you can open, share, or attach to a PR. By Linnet Labs.
"""
from __future__ import annotations

import html

_CSS = """
:root{--bg:#080d18;--surface:#111a2c;--surface-2:#0e1626;--border:#1f2c44;--fg:#eaf1fb;--fg-2:#b7c4d8;
--muted:#7f8ca3;--accent:#34d399;--accent-2:#6aa6ff;--warn:#fbbf24;--danger:#f87171;
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:Inter,system-ui,sans-serif;
background-image:radial-gradient(900px 440px at 82% -10%,rgba(52,211,153,.08),transparent 60%)}
.wrap{max-width:920px;margin:0 auto;padding:40px 22px 70px}
.brand{display:flex;align-items:center;gap:9px;font-weight:700}.brand .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px rgba(52,211,153,.7)}
.brand small{color:var(--muted);font-weight:500;font-size:12px}.brand small a{color:var(--accent)}
h1{font-size:26px;letter-spacing:-.02em;margin:26px 0 4px}.sub{color:var(--muted);font-family:var(--mono);font-size:12.5px;margin-bottom:26px}
h2{font-size:17px;margin:30px 0 12px;letter-spacing:-.01em}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.kpi .c{border:1px solid var(--border);border-radius:12px;padding:14px 18px;background:var(--surface-2)}
.kpi .c b{font-size:24px;display:block}.kpi .c span{color:var(--muted);font-size:12px}
.fl{font-family:var(--mono);font-size:12.5px;color:#8fd9be;word-break:break-all}
.card{border:1px solid var(--border);border-left:3px solid var(--accent-2);border-radius:11px;padding:12px 16px;margin-bottom:9px;background:var(--surface)}
.card.err{border-left-color:var(--danger)}.snip{color:var(--fg-2);font-size:13.5px;margin-top:5px}
.conflict{border:1px solid #3a2530;background:linear-gradient(180deg,#16121c,#120e16);border-radius:12px;padding:13px 16px;margin-bottom:10px}
.tag{font-family:var(--mono);font-size:11px;color:var(--danger);border:1px solid #4a2530;background:rgba(248,113,113,.07);padding:3px 9px;border-radius:100px}
.tag.ok{color:var(--accent);border-color:var(--accent-dim);background:rgba(52,211,153,.08)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:11px}@media(max-width:600px){.pair{grid-template-columns:1fr}}
.pair .c{border:1px solid #172238;border-radius:9px;padding:10px 12px;background:rgba(255,255,255,.02)}
.canon{color:var(--accent);font-family:var(--mono);font-size:12.5px;margin-top:9px}
.empty{color:var(--muted);padding:14px 0}
footer{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid #172238;padding-top:18px}
"""


def _e(s):
    return html.escape(str(s))


def build(repo, when, lint_findings, conflicts, verdicts=None):
    errs = [f for f in lint_findings if f.severity == "error"]
    confirmed = None
    if verdicts:
        confirmed = [(c, d) for c, d in zip(conflicts, verdicts) if d.get("real")]

    leak_html = "".join(
        f'<div class="card err"><span class="fl">{_e(f.file)}:{f.line}</span> '
        f'<span class="snip"><b>{_e(f.term_id)}</b> — {_e(f.reason)}</span></div>'
        for f in lint_findings
    ) or '<div class="empty">No banned terms reached public files. ✅</div>'

    def conflict_block(items, verified):
        if not items:
            return '<div class="empty">No contradictions found. ✅</div>'
        out = []
        for entry in items:
            c, d = entry if verified else (entry, None)
            tag = (f'<span class="tag">contradiction · {_e(" vs ".join(c["clash"]))}</span>'
                   if not verified else f'<span class="tag">confirmed · {_e(" vs ".join(c["clash"]))}</span>')
            canon = f'<div class="canon">canonical: {_e(d.get("canonical"))} — {_e(d.get("why",""))}</div>' if d else ""
            out.append(
                f'<div class="conflict">{tag}{canon}<div class="pair">'
                f'<div class="c"><span class="fl">{_e(c["a"]["file"])}:{c["a"]["line"]}</span>'
                f'<div class="snip">{_e(c["a"]["text"])}</div></div>'
                f'<div class="c"><span class="fl">{_e(c["b"]["file"])}:{c["b"]["line"]}</span>'
                f'<div class="snip">{_e(c["b"]["text"])}</div></div></div></div>'
            )
        return "".join(out)

    radar_html = conflict_block(confirmed, True) if confirmed is not None else conflict_block(conflicts, False)
    radar_count = (len(confirmed) if confirmed is not None else len(conflicts))
    radar_label = "confirmed contradictions" if confirmed is not None else "contradiction candidates"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concord report — {_e(repo)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>{_CSS}</style></head><body><div class="wrap">
<div class="brand"><span class="dot"></span>Concord <small>report · <a href="https://linnetlabs.org">by Linnet Labs</a></small></div>
<h1>Consistency report — {_e(repo)}</h1>
<div class="sub">{_e(when)} · computed by Concord, grounded in your repo</div>
<div class="kpi">
  <div class="c"><b style="color:{'var(--danger)' if errs else 'var(--accent)'}">{len(errs)}</b><span>codename leaks (public)</span></div>
  <div class="c"><b style="color:{'var(--warn)' if radar_count else 'var(--accent)'}">{radar_count}</b><span>{_e(radar_label)}</span></div>
</div>
<h2>🛡️ Leak guard — banned terms in public files</h2>
{leak_html}
<h2>⚠️ Contradiction radar</h2>
{radar_html}
<footer>Concord — keep a sprawling repo telling one story. AI you can check · by Linnet Labs.</footer>
</div></body></html>"""
