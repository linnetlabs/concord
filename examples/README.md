# Concord in 30 seconds

`examples/demo/` is a tiny repo with problems planted on purpose:

| File | Planted problem |
|------|-----------------|
| `docs/pricing.md` says `$49 per seat`, `docs/faq.md` says `$39 per seat` | a **value contradiction** across two docs |
| `src/limits.py` sets `MIN_RESPONDENTS = 8`, `settings.json` says `"min_respondents": 5` | a **code/config contradiction** |
| `docs/launch-notes.md` mentions `Project Nimbus` | an internal **codename leak** in a public file |
| `src/limits.py` has no doc linking to it | **undocumented code** |

Everything below runs with **no API key and no embeddings** (the deterministic core).

## 1. Catch the codename leak

```bash
concord lint examples/demo --rules examples/demo/rules.demo.yaml --scope public
```

```
docs/launch-notes.md:3:1: [error] codename-nimbus -- Internal codename -- ship the public name 'Skyline' instead.

1 finding(s): 1 error, 0 warn
```

Exit code is nonzero, so this fails CI. For inline PR annotations, add `--sarif` and
upload it to GitHub code-scanning.

## 2. See the document graph

```bash
concord graph examples/demo
```

```
# library graph -- 6 files, 3 doc-links (0 stale, 0 lagging)

## most-referenced files (incoming doc-links):
    1 <-  docs/pricing.md
    1 <-  docs/faq.md
    1 <-  docs/launch-notes.md
```

`concord graph examples/demo --mermaid` emits a freshness-coloured Mermaid graph to paste
into a README or PR. `concord ui examples/demo` renders the same graph live (the **Graph**
tab), nodes coloured by git freshness, lagging docs ringed.

## 3. Find undocumented code

```bash
concord coverage examples/demo
```

```
# doc coverage -- 6 files, 2 undocumented code file(s), 0 lagging doc(s)

## undocumented code (no doc links in; high-churn first):
       ...   src/limits.py
       ...   settings.json
```

## 4. The semantic layer: contradictions

The radar needs the local embedder (no API key; sentiment.ai downloads a ~90 MB e5 model
on first use), so it is the one step that isn't zero-setup:

```bash
concord index examples/demo
concord radar examples/demo            # add --verify to let an LLM confirm + name the canonical value
```

It flags same-topic passages whose hard values disagree -- here the `$49` (pricing.md) vs
`$39` (faq.md) price drift -- and prints a deterministic **canonical suggestion** (the side
whose file is fresher in git is the likely source of truth). `concord radar --prose` extends
this to non-numeric contradictions, LLM-judged.
