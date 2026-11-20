#!/usr/bin/env python3
"""Measure `formula_text` (formulas rebuilt from the PDF text layer) against
LaTeX ground truth, and write the regression fixtures its Rust test replays.

Ground truth: pdf-parse-bench (https://github.com/phorn1/pdf-parse-bench, MIT)
— synthetic PDFs compiled from LaTeX with randomised fonts, classes and
layouts, each with the exact LaTeX of every formula in reading order. Its
formulas come from `piushorn/wikipedia-latex-formulas-319k` (Wikipedia,
CC BY-SA 4.0).

Usage:
    # 1. convert with the dump on (the model off: only the text layer is measured)
    DOCLING_CODE_FORMULA_DIR=/nonexistent DOCLING_RS_FORMULA_TEXT_DUMP=<out>/<name> \
        docling-rs --enrich-formula --to md --output <out>/md <bench>/pdfs/<name>.pdf
    # 2. evaluate (and optionally write fixtures)
    python3 formula_text_eval.py <bench-dir> <out-dir> [--fixtures DIR]

A dumped region is matched to the ground-truth *display* formula whose
visible characters it shares most (in order); inline formulas are not
regions of their own and are not measured. Both sides are compared after
`canonical()`: LaTeX spells one formula many ways, and the text layer cannot
see `\\left`, spacing commands or font switches.
"""

import glob
import json
import unicodedata
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------- tokenizing

TOKEN = re.compile(r"\\[a-zA-Z]+\*?|\\.|[^\s]")


def tokens(latex):
    # a lone backslash is what is left of a trailing `\ ` space once the
    # formula is trimmed
    return [t for t in TOKEN.findall(latex) if t != "\\"]


DROP = {
    r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr",
    r"\Bigl", r"\Bigr", r"\biggl", r"\biggr", r"\Biggl", r"\Biggr", r"\middle",
    r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad", "~", r"\displaystyle",
    r"\textstyle", r"\scriptstyle", r"\limits", r"\nolimits", r"\nonumber",
    r"\notag", "&", r"\allowbreak", r"\hfill", r"\enspace", r"\thinspace",
    r"\medspace", r"\thickspace", r"\negthinspace",
}
# One-argument wrappers whose argument stays, the wrapper going.
UNWRAP = {
    r"\mathbf", r"\mathrm", r"\mathit", r"\mathsf", r"\mathtt", r"\boldsymbol",
    r"\bm", r"\mathcal", r"\mathbb", r"\mathfrak", r"\mathscr", r"\text",
    r"\textrm", r"\textit", r"\textbf", r"\mbox", r"\operatorname", r"\rm",
    r"\it", r"\bf", r"\cal", r"\hbox", r"\mathnormal", r"\textnormal", r"\emph",
    r"\pmb", r"\smash",
}
# Commands that draw nothing: they go with their argument.
INVISIBLE = {r"\vphantom", r"\phantom", r"\hphantom"}
# Typographic variants of one symbol — a size (`\bigcap`/`\cap`, a long
# arrow), a wide accent, a letterform (`\varkappa`) — which the page cannot
# tell apart from the spelling the author typed and which are spoken alike.
VARIANTS = {
    r"\widehat": r"\hat", r"\widetilde": r"\tilde", r"\bigcap": r"\cap",
    r"\bigcup": r"\cup", r"\bigoplus": r"\oplus", r"\bigotimes": r"\otimes",
    r"\bigvee": r"\vee", r"\bigwedge": r"\wedge", r"\bigsqcup": r"\sqcup",
    r"\Longleftrightarrow": r"\Leftrightarrow", r"\Longrightarrow": r"\Rightarrow",
    r"\Longleftarrow": r"\Leftarrow", r"\longrightarrow": r"\rightarrow",
    r"\longleftarrow": r"\leftarrow", r"\longleftrightarrow": r"\leftrightarrow",
    r"\longmapsto": r"\mapsto", r"\varkappa": r"\kappa", r"\varpi": r"\pi",
    r"\parallel": r"\|", r"\lVert": r"\|", r"\rVert": r"\|", r"\Vert": r"\|",
    r"\dotsm": "...", r"\dotsi": "...", r"\vdots": "...",
    r"\vDash": r"\models", r"\geqslant": r"\geq", r"\leqslant": r"\leq",
    r"\bot": r"\perp", r"\bigtriangleup": r"\triangle", r"\varnothing": r"\emptyset",
    r"\backepsilon": r"\ni", r"\varpropto": r"\propto",
}
SYN = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\to": r"\rightarrow",
    r"\gets": r"\leftarrow", r"\lbrace": r"\{", r"\rbrace": r"\}", r"\lvert": "|",
    r"\rvert": "|", r"\vert": "|", r"\mid": "|", r"\lbrack": "[", r"\rbrack": "]",
    r"\langle": "⟨", r"\rangle": "⟩", r"\bar": r"\overline", r"\dots": "...",
    r"\ldots": "...", r"\cdots": "...", r"\dotsc": "...", r"\dotsb": "...",
    r"\colon": ":", r"\varepsilon": r"\epsilon", r"\vartheta": r"\theta",
    r"\varphi": r"\phi", r"\varrho": r"\rho", r"\varsigma": r"\sigma",
    r"\lnot": r"\neg", r"\land": r"\wedge", r"\lor": r"\vee", r"\owns": r"\ni",
    r"\iff": r"\Leftrightarrow", r"\implies": r"\Rightarrow", r"\dfrac": r"\frac",
    r"\tfrac": r"\frac", r"\cfrac": r"\frac", r"\ast": "*", r"\lt": "<", r"\gt": ">",
    r"\textendash": "-", r"\minus": "-", r"\dag": r"\dagger", r"\hbar": r"\hbar",
r"\varnothing": r"\emptyset", r"\setminus": "\\backslash",
}
FUNCS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min",
    "det", "deg", "gcd", "sup", "inf", "arcsin", "arccos", "arctan", "sinh", "cosh",
    "tanh", "dim", "ker", "arg", "Pr", "hom", "lg", "liminf", "limsup", "mod",
}
ACCENTS = {r"\hat", r"\tilde", r"\bar", r"\overline", r"\vec", r"\dot", r"\ddot",
           r"\check", r"\acute", r"\grave", r"\breve"}
ARGS = {r"\frac": 2, r"\binom": 2, r"\sqrt": 1, r"\overline": 1, r"\underline": 1,
        r"\hat": 1, r"\tilde": 1, r"\vec": 1, r"\dot": 1, r"\ddot": 1, r"\check": 1,
        r"\acute": 1, r"\grave": 1, r"\breve": 1, r"\widehat": 1, r"\widetilde": 1,
        r"\overrightarrow": 1, r"\pmod": 1, r"\overset": 2, r"\underset": 2,
        r"\stackrel": 2, r"\overbrace": 1, r"\underbrace": 1}


def parse(toks, i=0, stop=None):
    """Tokens → nested lists: a group is a list; `^`/`_` and commands with
    arguments become tuples."""
    out = []
    while i < len(toks):
        t = toks[i]
        if t == stop:
            return out, i + 1
        if t == "{":
            g, i = parse(toks, i + 1, "}")
            out.append(g)
            continue
        if t == "}":  # unbalanced: ignore
            i += 1
            continue
        out.append(t)
        i += 1
    return out, i


def arg(seq, i):
    """The argument at seq[i] (a group or a single token), and the next index."""
    while i < len(seq) and isinstance(seq[i], str) and seq[i] in DROP:
        i += 1
    if i >= len(seq):
        return [], i
    a = seq[i]
    return (a if isinstance(a, list) else [a]), i + 1


def canon(seq):
    """A canonical token list: wrappers gone, scripts braced (sub before sup),
    function names spelled out, plain groups flattened."""
    # `{a \over b}` / `{n \choose k}`: TeX's infix fractions.
    for infix, cmd in ((r"\over", r"\frac"), (r"\choose", r"\binom")):
        if infix in [t for t in seq if isinstance(t, str)]:
            k = next(j for j, t in enumerate(seq) if t == infix)
            return [cmd, "{"] + canon(seq[:k]) + ["}", "{"] + canon(seq[k + 1:]) + ["}"]
    out = []
    i = 0
    while i < len(seq):
        t = seq[i]
        if isinstance(t, list):
            out.extend(canon(t))
            i += 1
            continue
        t = SYN.get(t, t)
        t = VARIANTS.get(t, t)
        t = SYN.get(t, t)
        if t in DROP:
            i += 1
            continue
        if t in UNWRAP:
            a, i = arg(seq, i + 1)
            out.extend(canon(a))
            continue
        if t in (r"\begin", r"\end"):
            # the environment name goes; rows (`\\`) and cells (`&`) stay
            # comparable through the row breaks the text layer also sees —
            # except the brackets a matrix environment draws
            env, i = arg(seq, i + 1)
            name = "".join(x for x in env if isinstance(x, str))
            fence = {"bmatrix": "[]", "pmatrix": "()", "vmatrix": "||", "Bmatrix": ["\\{", "\\}"]}.get(name)
            if fence:
                out.append(fence[0] if t == r"\begin" else fence[1])
            continue
        if t == r"\not" and i + 1 < len(seq) and isinstance(seq[i + 1], str):
            neg = {"=": r"\neq", r"\in": r"\notin", r"\equiv": r"\not\equiv",
                   "<": r"\nless", ">": r"\ngtr", r"\subset": r"\not\subset"}
            nxt = SYN.get(seq[i + 1], seq[i + 1])
            if nxt in neg:
                out.append(neg[nxt])
                i += 2
                continue
        if t in INVISIBLE:
            _, i = arg(seq, i + 1)
            continue
        if t == r"\smash" and i + 1 < len(seq) and seq[i + 1] == "[" and "]" in seq[i + 1:]:
            # `\smash[t]{2}`: the option goes, the argument stays
            j = seq.index("]", i + 1)
            a, i = arg(seq, j + 1)
            out.extend(canon(a))
            continue
        if t == r"\bmod":
            out.extend(list("mod"))
            i += 1
            continue
        if t.startswith("\\") and t[1:] in FUNCS:
            out.extend(list(t[1:]))
            i += 1
            continue
        if t == r"\pmod":
            a, i = arg(seq, i + 1)
            out.extend(["(", "m", "o", "d"] + canon(a) + [")"])
            continue
        if t in ("^", "_", "'"):
            # collect a run of scripts and order sub before sup; primes are
            # superscripts (`d'_{i}` = `d_{i}^{\prime}`)
            scripts = {}
            while i < len(seq) and (
                (isinstance(seq[i], str) and seq[i] in ("^", "_", "'"))
                # an empty group staggering indices (`\Gamma^{k}{}_{ij}`)
                or (seq[i] == [] and i + 1 < len(seq) and seq[i + 1] in ("^", "_"))
            ):
                if seq[i] == []:
                    i += 1
                    continue
                if seq[i] == "'":
                    scripts.setdefault("^", []).append(r"\prime")
                    i += 1
                    continue
                kind = seq[i]
                a, i = arg(seq, i + 1)
                scripts.setdefault(kind, []).extend(canon(a))
            for kind in ("_", "^"):
                if kind in scripts:
                    out.extend([kind, "{"] + scripts[kind] + ["}"])
            continue
        if False:
            i += 1
            continue
        if t == r"\sqrt" and i + 1 < len(seq) and seq[i + 1] == "[" and "]" in seq[i + 1:]:
            j = seq.index("]", i + 1)
            idx = canon(seq[i + 2:j])
            a, i = arg(seq, j + 1)
            out.extend([r"\sqrt", "["] + idx + ["]", "{"] + canon(a) + ["}"])
            continue
        if t in ACCENTS and i + 1 < len(seq) and isinstance(seq[i + 1], list):
            # `\tilde{Q_{s}}` and `\tilde{Q}_{s}` print alike: the accent
            # takes the first atom, the scripts follow it
            inner = canon(seq[i + 1])
            if len(inner) > 1 and inner[1] in ("_", "^"):
                out.extend([t, "{", inner[0], "}"] + inner[1:])
                i += 2
                continue
        if t in ARGS:
            i += 1
            out.append(t)
            for _ in range(ARGS[t]):
                a, i = arg(seq, i)
                out.extend(["{"] + canon(a) + ["}"])
            continue
        if t == "...":  # `\ldots` and three periods are the same dots
            out.extend([".", ".", "."])
        else:
            out.append(t)
        i += 1
    return out


def canonical(latex):
    latex = latex.strip().strip("$").strip()
    toks = tokens(latex)
    # split multi-char plain tokens are already single chars; digits split.
    tree, _ = parse(toks)
    out = canon(tree)
    # `{e'}^{2}` and `e^{\prime 2}`: two superscripts in a row are one
    k = 0
    while k + 1 < len(out):
        if out[k] == "}" and out[k + 1] == "^":
            # find the group this `}` closes and the `^` that opened it
            depth, j = 0, k
            while j >= 0:
                depth += {"}": 1, "{": -1}.get(out[j], 0)
                if depth == 0:
                    break
                j -= 1
            if j >= 1 and out[j - 1] == "^" and k + 2 < len(out) and out[k + 2] == "{":
                del out[k:k + 3]  # drop "} ^ {" — the two groups become one
                continue
        k += 1
    # a trailing row break (`…\\ \end{matrix}`) ends no row
    while out and out[-1] == "\\\\":
        out.pop()
    return out


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def similarity(a, b):
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


# ------------------------------------------------------- matching regions to GT

GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon",
    "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu", "π": r"\pi", "σ": r"\sigma",
    "φ": r"\phi", "ω": r"\omega", "∑": r"\sum", "∫": r"\int", "∏": r"\prod",
}


def visible(canon_toks):
    """The characters a formula draws, as a multiset (for matching)."""
    return Counter(t for t in canon_toks if t not in ("{", "}", "^", "_", r"\frac", r"\sqrt",
                                                     r"\overline", r"\underline"))


def region_chars(rec):
    """The letters and digits inside the region's box (a dump also carries
    the glyphs around it, for the guards that read the surroundings)."""
    out = Counter()
    l, t, r, b = rec["bbox"]
    for g in rec["glyphs"]:
        xc, yc = (g[1] + g[3]) / 2, (g[2] + g[4]) / 2
        if not (l - 1 <= xc <= r + 1 and t - 1 <= yc <= b + 1):
            continue
        # Math alphanumerics (`𝑀`, the text layer's reading of TeX's
        # `u1D440` glyph names) count as the letters they style.
        ch = unicodedata.normalize('NFKC', g[0])
        if len(ch) == 1 and ch.isascii() and ch.isalnum():
            out[ch] += 1
    return out


def match_score(a, b):
    a = Counter({k: v for k, v in a.items() if len(k) == 1 and k.isascii() and k.isalnum()})
    b = Counter({k: v for k, v in b.items() if len(k) == 1 and k.isascii() and k.isalnum()})
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def replay(dirs):
    """Re-run the current `formula_text` over dumped regions (the replay
    example): {dump path: (latex, error)}."""
    import subprocess
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    cmd = ["cargo", "run", "-q", "--release", "-p", "docling-pdf", "--example",
           "formula_text_replay", "--"] + dirs
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("formula_text_replay failed:\n" + proc.stderr[-3000:])
    out = proc.stdout
    got = {}
    for line in out.splitlines():
        r = json.loads(line)
        got[r["id"]] = (r["latex"], r["error"])
    return got


def evaluate(bench, out_dir, replayed=None, label=""):
    rows = []
    for gt_path in sorted(glob.glob(os.path.join(bench, "ground_truth", "*.json"))):
        name = os.path.splitext(os.path.basename(gt_path))[0]
        gt = [g["data"] for g in json.load(open(gt_path)) if g["type"] == "display-formula"]
        gt_c = [canonical(g) for g in gt]
        recs = []
        for p in sorted(glob.glob(os.path.join(out_dir, name, "*.json"))):
            rec = json.load(open(p))
            if replayed is not None and p in replayed:
                rec["latex"], rec["error"] = replayed[p]
            recs.append(rec)
        name = f"{label}{name}"
        used = set()
        for rec in recs:
            rc = region_chars(rec)
            best, bi = 0.0, None
            for k, c in enumerate(gt_c):
                if k in used:
                    continue
                s = match_score(rc, visible(c))
                if s > best:
                    best, bi = s, k
            if bi is None or best < 0.5:
                rows.append(dict(pdf=name, status="unmatched", rec=rec, gt=None, score=best))
                continue
            used.add(bi)
            row = dict(pdf=name, rec=rec, gt=gt[bi], gt_canon=gt_c[bi], score=best)
            if rec.get("latex"):
                ours = canonical(rec["latex"])
                row["ours_canon"] = ours
                row["sim"] = similarity(ours, gt_c[bi])
                # Anything but an exact match is wrong: a near miss read aloud
                # is a wrong formula. `sim` stays for diagnosis only.
                row["status"] = "exact" if ours == gt_c[bi] else "wrong"
                # One line of a multi-line display read as its own formula
                # (the region was split per row): not wrong, not the whole.
                if row["status"] == "wrong" and "\\\\" in gt_c[bi] and "\\\\" not in ours:
                    rows_gt = " ".join(gt_c[bi]).split(" \\\\ ")
                    if " ".join(ours) in [r.strip() for r in rows_gt]:
                        row["status"] = "one row"
            else:
                row["status"] = "refused"
                row["why"] = re.sub(r"[\d.]+", "#", rec.get("error") or "")[:70]
            rows.append(row)
        for k in range(len(gt)):
            if k not in used:
                rows.append(dict(pdf=name, status="no region", gt=gt[k]))
    return rows


def report(rows):
    st = Counter(r["status"] for r in rows)
    regions = sum(st[k] for k in ("exact", "one row", "wrong", "refused"))
    rebuilt = sum(st[k] for k in ("exact", "one row", "wrong"))
    print("display formulas with a matched region:", regions)
    print("  rebuilt from the text layer:", rebuilt, f"({100*rebuilt/max(regions,1):.1f}%)")
    print("    exact after canonicalisation:", st["exact"], f"({100*st['exact']/max(rebuilt,1):.1f}% of rebuilt)")
    print("    one row of a multi-line display:", st["one row"])
    print("    wrong:", st["wrong"], f"({100*st['wrong']/max(rebuilt,1):.1f}% of rebuilt)")
    sims = [r["sim"] for r in rows if "sim" in r]
    if sims:
        print(f"    mean token similarity: {sum(sims)/len(sims):.3f}")
    print("  refused (left to the model):", st["refused"])
    for why, n in Counter(r["why"] for r in rows if r["status"] == "refused").most_common(8):
        print(f"      {n:4}  {why}")
    print("display formulas the layout model gave no formula region:", st["no region"])
    print("formula regions matching no display formula (inline / mis-labelled):", st["unmatched"])


def write_fixtures(rows, dst):
    os.makedirs(dst, exist_ok=True)
    cases = []
    for r in rows:
        if r["status"] in ("unmatched", "no region"):
            continue
        rec = r["rec"]
        cases.append({
            "id": f"{r['pdf']}-p{rec.get('page', 0)}-{len(cases):04d}",
            "gt": r["gt"].strip().strip("$").strip(),
            "status": r["status"],
            "latex": rec.get("latex"),
            "bbox": rec["bbox"],
            "glyphs": rec["glyphs"],
            "rules": rec["rules"],
        })
    path = os.path.join(dst, "cases.json")
    with open(path, "w") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=0)
    # The recorded output is the replay of the fixture itself — the stored
    # coordinates are rounded to 0.01 pt, which can tip a borderline glyph —
    # so the Rust test reproduces it exactly; statuses are re-scored on it.
    replayed = replay([os.path.abspath(path)])
    for c in cases:
        latex, err = replayed[c["id"]]
        c["latex"] = latex
        if latex is None:
            c["status"] = "refused"
        else:
            ours, gt = canonical(latex), canonical(c["gt"])
            sim = similarity(ours, gt)
            c["status"] = "exact" if ours == gt else "wrong"
    with open(path, "w") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=0)
    print(f"wrote {len(cases)} fixtures to {path}: {dict(Counter(c['status'] for c in cases))}")


if __name__ == "__main__":
    # Positional arguments come in pairs: <bench-set dir> <dump dir>; several
    # sets are measured and written together.
    args, flags = [], []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--fixtures":
            flags.append((a, next(it)))
        elif a.startswith("--"):
            flags.append((a, None))
        else:
            args.append(a)
    flag = dict(flags)
    pairs = list(zip(args[0::2], args[1::2]))
    replayed = None
    if "--replay" in flag:
        dirs = [d for _, out in pairs for d in sorted(glob.glob(os.path.join(out, "*")))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "p*.json"))]
        replayed = replay(dirs)
    rows = []
    for bench, out in pairs:
        label = os.path.basename(os.path.normpath(bench)) + "/" if len(pairs) > 1 else ""
        rows += evaluate(bench, out, replayed, label)
    report(rows)
    if "--show" in flag:
        for r in rows:
            if r["status"] == "wrong":
                print("---", r["pdf"], r["status"], f"{r.get('sim', 0):.2f}")
                print("  gt:  ", r["gt"])
                print("  ours:", r["rec"]["latex"])
    if "--fixtures" in flag:
        write_fixtures(rows, flag["--fixtures"])
