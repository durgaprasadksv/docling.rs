#!/usr/bin/env python3
"""Score docling.rs on every PDF of pdf-parse-bench, element by element, and
check the result against the committed expectations.

pdf-parse-bench (https://huggingface.co/datasets/piushorn/pdf-parse-bench,
https://github.com/phorn1/pdf-parse-bench, MIT) is a set of born-digital PDFs
compiled from LaTeX with randomised fonts, classes and layouts; every PDF
comes with the exact ground truth of what is on it, in reading order: text
passages, inline and display formulas (from Wikipedia, CC BY-SA 4.0) and
tables (LaTeX `tabular`). Three sets: `2026-q1-formulas-only`,
`2026-q1-tables-only` (the Hugging Face configs) and `2025-q4` (the
repository's earlier set).

Each ground-truth element gets a status:

- text passage: `found` (its words, in order, are in our Markdown),
  `partial` (at least 80% of its words are), `missing`;
- inline formula: `found` when its visible characters appear in order in our
  text, else `missing` (inline math is read as text today: this measures that
  nothing is lost, not that its structure is kept);
- display formula: `exact` / `near` / `wrong` / `refused` / `no region` — see
  `formula_text_eval.py` (the text-layer reconstruction, model off);
- table: `exact` (same grid, every cell equal), `good` (cell F1 >= 0.9),
  `partial` (>= 0.5), `missing`.

Usage (scripts/conformance/pdf_parse_bench.sh drives it):
    pdf_parse_bench.py <bench data dir> <run dir> [--check EXPECTED] [--update EXPECTED]
        [--fixtures DIR] [--show KIND]

`<run dir>/<set>/out/` holds the CLI's `.md` + `.json`, `<run dir>/<set>/<pdf>/`
the formula-region dumps. `--check` exits 1 when any element's status got
worse than expected (it prints the ones that got better: run `--update`).
"""

import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formula_text_eval as fte  # noqa: E402

SETS = ["2026-q1-formulas-only", "2026-q1-tables-only", "2025-q4"]

# ------------------------------------------------------------------ helpers

WORD = re.compile(r"[0-9A-Za-zÀ-ÿ]+")


def words(text):
    return [w.lower() for w in WORD.findall(text)]


GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω", r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ",
    r"\Lambda": "Λ", r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ",
    r"\Psi": "Ψ", r"\Omega": "Ω",
}


GREEK.update({r"\ell": "ℓ", r"\varpi": "π"})
# Variant letterforms the PDF draws for the same letter (math italic phi,
# lunate epsilon, …) fold to one.
FOLD = str.maketrans({"ϕ": "φ", "ϵ": "ε", "ϑ": "θ", "ϱ": "ρ", "ς": "σ", "ϖ": "π", "µ": "μ"})


def visible(latex):
    """The letters, digits and Greek a formula draws, in order."""
    out = []
    for t in fte.canonical(latex):
        if t in GREEK:
            out.append(GREEK[t])
        elif len(t) == 1 and (t.isalnum()):
            out.append(t)
    return "".join(out).translate(FOLD)


MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)


def alnum_stream(text):
    """The letters and digits of our Markdown in order: HTML entities
    decoded, our own `$…$` LaTeX read as what it draws, math alphanumerics
    folded to plain letters (NFKC)."""
    import html
    import unicodedata
    text = html.unescape(text)
    text = MATH.sub(lambda m: visible(m.group(1) or m.group(2)), text)
    text = unicodedata.normalize("NFKC", text).translate(FOLD)
    return "".join(c for c in text if c.isalnum())


def contains(stream, v, slack=3):
    """Whether `stream` holds `v`'s characters within a span barely longer
    than `v`, in any order inside it — the PDF draws a superscript before
    its subscript where the LaTeX names the subscript first."""
    if v in stream:
        return True
    need = Counter(v)
    n = len(v) + slack
    start = 0
    while True:
        i = stream.find(v[0], start)
        if i < 0:
            return False
        window = Counter(stream[max(0, i - slack): i + n])
        if not need - window:
            return True
        start = i + 1


# ------------------------------------------------------------------ tables

def group(s, i):
    """The brace group starting at or after s[i] (skipping spaces and one
    `[...]` option): (content, index after it)."""
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i < len(s) and s[i] == "[":
        i = s.index("]", i) + 1
        while i < len(s) and s[i] in " \t\n":
            i += 1
    if i >= len(s) or s[i] != "{":
        return "", i
    depth, j = 0, i
    while j < len(s):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def split_top(s, sep):
    """Split at `sep` outside braces (`&` in a row, `\\\\` between rows)."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        if s[i] == "\\" and s.startswith(sep, i) and depth == 0 and sep == "\\\\":
            out.append(cur)
            cur, i = "", i + 2
            continue
        c = s[i]
        depth += {"{": 1, "}": -1}.get(c, 0)
        if c == sep and depth == 0:
            out.append(cur)
            cur = ""
        elif c == "\\" and i + 1 < len(s):
            cur += s[i:i + 2]
            i += 2
            continue
        else:
            cur += c
        i += 1
    out.append(cur)
    return out


def strip_cmd(s, name, nargs):
    """Remove every `\\name[opt]{..}…` with its arguments."""
    while True:
        k = s.find("\\" + name)
        if k < 0 or (k + len(name) + 1 < len(s) and s[k + len(name) + 1].isalpha()):
            return s
        j = k + len(name) + 1
        for _ in range(nargs):
            _, j = group(s, j)
        s = s[:k] + s[j:]


def tabular_cells(latex):
    """A LaTeX `tabular` as rows of normalised cell strings, spanned cells
    repeated in every position they cover (the way docling's grid holds a
    merged cell)."""
    body = latex
    m = re.search(r"\\begin\{tabular\*?\}", body)
    if m:
        _, j = group(body, m.end())
        body = body[j:]
    body = re.sub(r"\\end\{tabular\*?\}", "", body)
    for name, n in (("rowcolor", 1), ("cellcolor", 1), ("color", 1), ("cline", 1),
                    ("cmidrule", 1), ("arrayrulecolor", 1), ("addlinespace", 0)):
        body = strip_cmd(body, name, n)
    body = re.sub(r"\\(hline|toprule|midrule|bottomrule|hhline)\b", "", body)
    rows, pending = [], {}  # pending: column -> (text, rows left) of \multirow
    for row in split_top(body, "\\\\"):
        if not row.strip():
            continue
        cells, col = [], 0
        for cell in split_top(row, "&"):
            cell = cell.strip()
            if col in pending and not cell:
                # the empty placeholder under a \multirow: the spanned text
                text, left = pending[col]
                cells.append(text)
                pending[col] = (text, left - 1)
                if left - 1 <= 0:
                    del pending[col]
                col += 1
                continue
            span, rspan = 1, 1
            if cell.startswith("\\multicolumn"):
                n, j = group(cell, len("\\multicolumn"))
                _, j = group(cell, j)
                cell, _ = group(cell, j)
                span = int(n) if n.strip().isdigit() else 1
                cell = cell.strip()
            if cell.startswith("\\multirow"):
                n, j = group(cell, len("\\multirow"))
                _, j = group(cell, j)
                cell, _ = group(cell, j)
                rspan = int(n) if n.strip().lstrip("-").isdigit() else 1
            text = norm_cell(cell)
            for k in range(span):
                if text == "" and (col + k) in pending:
                    continue
                cells.append(text)
                if rspan > 1:
                    pending[col + k] = (text, rspan - 1)
            col += span
        while col in pending:
            text, left = pending[col]
            cells.append(text)
            pending[col] = (text, left - 1)
            if left - 1 <= 0:
                del pending[col]
            col += 1
        rows.append(cells)
    return rows


SYMBOLS = {
    r"\pm": "±", r"\mp": "∓", r"\uparrow": "↑", r"\downarrow": "↓", r"\times": "×",
    r"\sim": "∼", r"\approx": "≈", r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\checkmark": "✓", r"\cdot": "·", r"\%": "%", r"\&": "&", r"\_": "_", r"\#": "#",
    r"\dagger": "†", r"\ddagger": "‡", r"\star": "⋆", r"\ast": "*", r"\rightarrow": "→",
    r"\to": "→", r"\infty": "∞", r"\circ": "∘", r"\degree": "°", r"\textdegree": "°",
    r"\textasciitilde": "~", r"\textless": "<", r"\textgreater": ">", r"\$": "$",
}
SYMBOLS.update(GREEK)


def norm_cell(text):
    """A cell's text as it reads: LaTeX symbols as their characters, markup
    and math delimiters gone, NFKC, minus as hyphen, spaces collapsed,
    lower case — applied to both sides, ground truth and ours."""
    import unicodedata
    # a line break inside a cell (`\\makecell{a \\\\ b}`) is a space
    text = text.replace("\\\\", " ")
    text = re.sub(r"\\[a-zA-Z]+|\\[%&_#$]", lambda m: SYMBOLS.get(m.group(0), ""), text)
    text = text.replace("$", "").replace("^", "").replace("_", "").replace("{", "").replace("}", "").replace("~", " ")
    text = unicodedata.normalize("NFKC", text).replace("−", "-").translate(FOLD)
    return " ".join(text.split()).lower()


def json_tables(doc):
    out = []
    for t in doc.get("tables", []):
        grid = (t.get("data") or {}).get("grid") or []
        out.append([[norm_cell(c.get("text", "")) for c in row] for row in grid])
    return out


def cell_f1(a, b):
    ca = Counter(c for r in a for c in r if c)
    cb = Counter(c for r in b for c in r if c)
    inter = sum((ca & cb).values())
    if not ca or not cb:
        return 0.0
    p, r = inter / sum(cb.values()), inter / sum(ca.values())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ------------------------------------------------------------------ scoring

RANK = {
    "text": {"found": 2, "partial": 1, "missing": 0},
    "inline": {"found": 1, "missing": 0},
    "display": {"exact": 4, "near": 3, "refused": 2, "no region": 2, "wrong": 1},
    "table": {"exact": 3, "good": 2, "partial": 1, "missing": 0},
}


def score_set(bench, run, name):
    gt_dir = os.path.join(bench, name, "ground_truth")
    out_dir = os.path.join(run, name, "out")
    result = {}
    # display formulas through the formula evaluator (region dumps)
    disp_rows = fte.evaluate(os.path.join(bench, name), os.path.join(run, name))
    disp = {}
    for r in disp_rows:
        if r["status"] == "unmatched":
            continue
        disp.setdefault(r["pdf"], []).append((r["gt"], r["status"], r))
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.json"))):
        pdf = os.path.splitext(os.path.basename(gt_path))[0]
        gt = json.load(open(gt_path))
        md_path = os.path.join(out_dir, pdf + ".md")
        js_path = os.path.join(out_dir, pdf + ".json")
        md = open(md_path).read() if os.path.exists(md_path) else ""
        doc = json.load(open(js_path)) if os.path.exists(js_path) else {}
        md_words = " ".join(words(md))
        md_bag = Counter(words(md))
        stream = alnum_stream(md)
        tables = json_tables(doc)
        used_tables = set()
        by_gt = {g: s for g, s, _ in disp.get(pdf, [])}
        elems = []
        for el in gt:
            kind, data = el["type"], el["data"]
            if kind == "text":
                ws = words(data)
                if not ws or " ".join(ws) in md_words:
                    st = "found"
                else:
                    have = sum((Counter(ws) & md_bag).values()) / len(ws)
                    st = "partial" if have >= 0.8 else "missing"
                elems.append(["text", st])
            elif kind == "inline-formula":
                v = visible(data)
                elems.append(["inline", "found" if not v or contains(stream, v) else "missing"])
            elif kind == "display-formula":
                elems.append(["display", by_gt.get(data, "no region")])
            elif kind == "table":
                want = tabular_cells(data)
                best, bi = 0.0, None
                for k, have in enumerate(tables):
                    if k in used_tables:
                        continue
                    f = cell_f1(want, have)
                    if f > best:
                        best, bi = f, k
                if bi is not None:
                    used_tables.add(bi)
                if bi is not None and tables[bi] == want:
                    st = "exact"
                else:
                    st = "good" if best >= 0.9 else "partial" if best >= 0.5 else "missing"
                elems.append(["table", st])
        result[f"{name}/{pdf}"] = elems
    return result, disp_rows


def summary(result):
    tally = {}
    for elems in result.values():
        for kind, st in elems:
            tally.setdefault(kind, Counter())[st] += 1
    return tally


def report(result):
    for kind, c in sorted(summary(result).items()):
        n = sum(c.values())
        parts = ", ".join(f"{st} {k}" for k, st in sorted(((v, s) for s, v in c.items()), reverse=True))
        print(f"{kind:8} {n:5}  {parts}")


def compare(result, expected):
    worse, better = [], []
    for pdf, elems in result.items():
        exp = expected.get(pdf)
        if exp is None or len(exp) != len(elems):
            worse.append((pdf, "element list changed"))
            continue
        for i, ((kind, st), (_, was)) in enumerate(zip(elems, exp)):
            r, w = RANK[kind][st], RANK[kind][was]
            if r < w:
                worse.append((pdf, f"#{i} {kind}: {was} -> {st}"))
            elif r > w:
                better.append((pdf, f"#{i} {kind}: {was} -> {st}"))
    return worse, better


if __name__ == "__main__":
    bench, run = sys.argv[1], sys.argv[2]
    flag = {sys.argv[i]: (sys.argv[i + 1] if i + 1 < len(sys.argv) else None)
            for i in range(3, len(sys.argv)) if sys.argv[i].startswith("--")}
    result, disp_rows = {}, []
    for name in SETS:
        if os.path.isdir(os.path.join(bench, name)):
            r, d = score_set(bench, run, name)
            result.update(r)
            for row in d:
                row["pdf"] = f"{name}/{row['pdf']}"
            disp_rows += d
    report(result)
    if "--show" in flag:
        kind = flag["--show"]
        for pdf, elems in result.items():
            for i, (k, st) in enumerate(elems):
                if k == kind and RANK[k][st] < max(RANK[k].values()):
                    print(pdf, i, st)
    if "--fixtures" in flag:
        fte.write_fixtures(disp_rows, flag["--fixtures"])
    if "--update" in flag:
        with open(flag["--update"], "w") as fh:
            json.dump({"source": "pdf-parse-bench @ de72a161", "elements": result}, fh, indent=0)
        print("wrote", flag["--update"])
    if "--check" in flag:
        expected = json.load(open(flag["--check"]))["elements"]
        worse, better = compare(result, expected)
        for pdf, what in better[:30]:
            print("better:", pdf, what)
        if better:
            print(f"{len(better)} element(s) improved: re-run with --update")
        for pdf, what in worse[:50]:
            print("WORSE: ", pdf, what)
        if worse:
            print(f"{len(worse)} element(s) regressed")
            sys.exit(1)
        print("no regressions")
