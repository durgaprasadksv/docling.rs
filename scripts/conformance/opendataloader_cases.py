#!/usr/bin/env python3
"""OpenDataLoader's synthetic issue tests as docling.rs conformance cases.

    opendataloader_cases.py extract  <upstream-clone> <fixtures-dir> <draft.json>
    opendataloader_cases.py evaluate <draft.json> <ours-dir> <fixtures-dir> <cases.json> [report.md]

`extract` reads every `Issue*Test.java` of the upstream clone and, for each
`@Test` method, the fixture it converts (matched to `<fixtures-dir>/issue-*/
<name>.pdf`, the PDFs its run left under `target/test-samples/`) and the
assertions that can be expressed as checks on our output:

    assertOrder(SYMPTOM, out, "a", "b", …)        -> order [a, b, …]
    assertBefore(out, "a", "b")                    -> order [a, b]
    assertTrue(out.contains("x"))                   -> contains [x]
    assertFalse(out.contains("x")), !out.contains  -> not_contains [x]
    list.toString().contains("x") / item.path("content")…contains("x")
                                                    -> in_list / not_in_list [x]
    assertOneList(out, "a", …)                      -> one_list [a, …]
    assertSectionLists(out, a, b, c, d)             -> one_list [a, b], one_list [c, d], separate_lists [a, c]
    isInList(out, "x")                              -> in_list / not_in_list [x]
    lineThatContains(out, "x").startsWith("- ")     -> line_starts [[x, "- "]]
    assertEquals(N, …"list items"….size())          -> list_items N
    assertRowsInOrder(markdown, ENTRIES)  (#389)    -> same_line rows + order

Anything else is recorded under `unparsed` and the case is `unchecked`.

`evaluate` runs the checks against docling.rs's outputs (`<ours-dir>/<issue-dir>/
<name>.md` and `<name>.reader.json`, as `docling-rs --to md,reader` writes them),
measures each fixture's word recall against OpenDataLoader's own Markdown
(`<name>.md` beside the fixture, committed as `<name>.odl.md`), and writes the
committed `cases.json` with a `status` per case (`pass` | `xfail` | `unchecked`)
and a `recall` per fixture. `crates/docling/tests/opendataloader.rs` replays it.
"""
import json
import re
import sys
from pathlib import Path

STR = r'"((?:[^"\\]|\\.)*)"'
CHECKS = ("order", "contains", "not_contains", "line_starts", "same_line", "one_list", "separate_lists", "in_list", "not_in_list")


# ---------------------------------------------------------------- extract ---

def unescape(s):
    return bytes(s, "utf-8").decode("unicode_escape")


def strings(arg_text):
    # A literal that is only whitespace (`" "` in a join) is never an expectation.
    return [unescape(m) for m in re.findall(STR, arg_text) if unescape(m).strip()]


def balanced(src, i, open_c, close_c):
    """End index (exclusive) of the group that opened just before `i`."""
    depth = 1
    j = i
    while depth and j < len(src):
        c = src[j]
        if c == '"':
            j += 1
            while src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
        j += 1
    return j


def methods(src):
    out = []
    for m in re.finditer(r"@Test\s*(?:@[\w.]+(?:\([^)]*\))?\s*)*(?:\w+\s+)*void\s+(\w+)\s*\([^)]*\)\s*(?:throws[^{]*)?\{", src):
        end = balanced(src, m.end(), "{", "}")
        out.append((m.group(1), src[m.end():end - 1]))
    return out


def javadoc_before(src, name):
    m = re.search(r"/\*\*((?:(?!\*/).)*)\*/\s*@Test[^{]*\bvoid\s+" + re.escape(name) + r"\s*\(", src, re.S)
    return " ".join(l.strip(" *") for l in m.group(1).splitlines()).strip() if m else ""


def calls(body, fn):
    out = []
    for m in re.finditer(r"\b" + re.escape(fn) + r"\s*\(", body):
        end = balanced(body, m.end(), "(", ")")
        out.append(body[m.end():end - 1])
    return out


def negated(cond):
    """Whether the `.contains(…)` in `cond` is negated: a `!` right before its
    receiver (`!out.contains(x)`), not a `!=` elsewhere in the condition
    (`item != null && item.path("content")…contains(x)`)."""
    head = cond.split(".contains(")[0]
    seg = re.split(r"&&|\|\|", head)[-1].strip()
    return seg.startswith("!")


def receiver_is_whole(expr):
    """`markdown.contains(…)`, `document.toString().contains(…)` and a paragraph
    or title element's `content` test the whole output (a substring present
    on some element); only a receiver named for a list or an item
    (`list`, `item`, `lastItem`, `onlyList(…)`) tests list membership."""
    head = expr.split(".contains(")[0]
    seg = re.split(r"&&|\|\|", head)[-1]
    return not re.search(r"\b\w*([iI]tem|[lL]ist)\w*\b", seg)


def receiver_is_output(cond):
    """`markdown`, `document`, `text`, `json`, `html`, `out`…: the whole output,
    on which a negated `contains` is a real absence check. Negated on an
    `element` or `node`, it says "not in *that* element", which has no
    counterpart here."""
    head = re.split(r"&&|\|\|", cond.split(".contains(")[0])[-1].strip().lstrip("!").strip()
    return re.match(r"(markdown|document|text|json|html|out|output|result|md|content)\b", head) is not None


def contained_literals(cond):
    """The string literals that are arguments of `.contains(…)` in `cond` — not
    the field names of the `.path("kids")` chain in front of them."""
    return [unescape(m) for m in re.findall(r"\.contains\(\s*" + STR, cond) if unescape(m).strip()]


def extract(upstream, fixtures):
    on_disk = {}
    for pdf in fixtures.glob("issue-*/*.pdf"):
        m = re.match(r"issue-(\d+)", pdf.parent.name)
        on_disk.setdefault(m.group(1), {})[pdf.stem] = str(pdf.relative_to(fixtures))
    tests = sorted(upstream.glob("java/opendataloader-pdf-core/src/test/java/org/opendataloader/pdf/issues/Issue*Test.java"))
    cases = []
    for path in tests:
        src = path.read_text()
        issue = re.match(r"Issue(\d+)", path.stem).group(1)
        known = on_disk.get(issue, {})
        for name, body in methods(src):
            conv = re.search(r"\b(convertMarkdown|convertJson|convertText|convert|markdownOf|jsonOf|markdownLines\w*|jsonLines\w*|textOf|htmlOf)\s*\(\s*" + STR, body)
            sample = conv.group(2) if conv else None
            if sample not in known:
                hits = [x for x in (unescape(m) for m in re.findall(STR, body)) if x in known]
                sample = hits[0] if hits else sample
            case = {"id": f"{issue}/{name}", "issue": issue, "test": path.stem, "method": name,
                    "pdf": known.get(sample), "doc": javadoc_before(src, name),
                    "checks": {k: [] for k in CHECKS}, "list_items": None, "unparsed": []}
            ch = case["checks"]
            for a in calls(body, "assertOrder"):
                args = a.split(",", 2)
                s = strings(args[2]) if len(args) == 3 else strings(a)
                if s:
                    ch["order"].append(s)
            for a in calls(body, "assertBefore"):
                s = strings(a)
                if len(s) >= 2:
                    ch["order"].append(s[:2])
            for a in calls(body, "assertOneList"):
                s = strings(a)
                if s:
                    ch["one_list"].append(s)
            # assertSectionLists(document, first, last, nextFirst, nextLast): the
            # two sections' items are two lists, each whole, and not the same.
            for a in calls(body, "assertSectionLists"):
                s = strings(a)
                if len(s) == 4:
                    ch["one_list"].append(s[:2])
                    ch["one_list"].append(s[2:])
                    ch["separate_lists"].append([s[0], s[2]])
                elif s:
                    ch["one_list"].append(s)
            def on_a_row(cond):
                """A table row's or cell's text has no counterpart check here."""
                head = re.split(r"&&|\|\|", cond.split(".contains(")[0])[-1]
                return re.search(r"\b\w*([rR]ow|[cC]ell|[tT]able)\w*\b", head) is not None

            for a in calls(body, "assertTrue"):
                cond = a.split(",")[0]
                if ".contains(" in cond and on_a_row(cond):
                    case["unparsed"].append("assertTrue(" + a.strip()[:120] + ")")
                elif "isInList" in a:
                    ch["in_list"] += strings(calls(a, "isInList")[0])
                elif ".contains(" in cond and not negated(cond):
                    lits = contained_literals(cond)
                    (ch["contains"] if receiver_is_whole(cond) else ch["in_list"]).extend(lits)
                elif ".contains(" in cond and (not receiver_is_whole(cond) or receiver_is_output(cond)):
                    lits = contained_literals(cond)
                    (ch["not_contains"] if receiver_is_whole(cond) else ch["not_in_list"]).extend(lits)
                elif "lineThatContains" not in cond:
                    case["unparsed"].append("assertTrue(" + a.strip()[:120] + ")")
            for a in calls(body, "assertFalse"):
                cond = a.split(",")[0]
                if ".contains(" in cond and on_a_row(cond):
                    case["unparsed"].append("assertFalse(" + a.strip()[:120] + ")")
                elif "isInList" in a:
                    ch["not_in_list"] += strings(calls(a, "isInList")[0])
                elif ".contains(" in cond and (not receiver_is_whole(cond) or receiver_is_output(cond)):
                    lits = contained_literals(cond)
                    (ch["not_contains"] if receiver_is_whole(cond) else ch["not_in_list"]).extend(lits)
                else:
                    case["unparsed"].append("assertFalse(" + a.strip()[:120] + ")")
            for m in re.finditer(r'lineThatContains\(\s*\w+\s*,\s*' + STR + r'\s*\)\s*\.startsWith\(\s*' + STR + r'\s*\)', body):
                ch["line_starts"].append([unescape(m.group(1)), unescape(m.group(2))])
            for a in calls(body, "assertEquals"):
                m = re.match(r"\s*(\d+)\s*,(.*)", a, re.S)
                if m and "list items" in m.group(2) and ".size()" in m.group(2):
                    case["list_items"] = int(m.group(1))
                else:
                    case["unparsed"].append("assertEquals(" + a.strip()[:120] + ")")
            if calls(body, "assertRowsInOrder"):
                # Issue 389's ENTRIES, transcribed from the test.
                entries = [("01", "Introduction", "03"), ("02", "Background of the plan", "09"), ("03", "Methods", "14"),
                           ("04", "Results of the survey", "22"), ("05", "Discussion", "27"), ("06", "Appendix tables", "41")]
                if "entries = {ENTRIES[0]}" in body:
                    entries = entries[:1]
                ch["same_line"] = [list(e) for e in entries]
                ch["order"].append([e[1] for e in entries])
            for fn in ("assertNotNull", "assertNull"):
                for a in calls(body, fn):
                    case["unparsed"].append(fn + "(" + a.strip()[:120] + ")")
            cases.append(case)
    return cases


# --------------------------------------------------------------- evaluate ---

def norm(s):
    return re.sub(r"\s+", " ", s)


def words(s):
    return {w for w in re.split(r"[^0-9a-zA-Z]+", s.lower()) if len(w) > 1}


def has_checks(case):
    return any(case["checks"].get(k) for k in CHECKS) or case.get("list_items") is not None


def check(case, md, rd):
    """The failures of a case against our Markdown `md` and reader JSON `rd`."""
    ch = case["checks"]
    fails = []
    n = norm(md or "")
    lines = (md or "").splitlines()
    for seq in ch.get("order", []):
        pos = -1
        for s in seq:
            i = n.find(norm(s), pos + 1)
            if i < 0:
                fails.append(f"order: {s!r} missing after position {pos}")
                break
            pos = i
    for s in ch.get("contains", []):
        if norm(s) not in n:
            fails.append(f"contains: {s!r} missing")
    for s in ch.get("not_contains", []):
        if norm(s) in n:
            fails.append(f"not_contains: {s!r} present")
    for needle, prefix in ch.get("line_starts", []):
        hit = [l for l in lines if needle in l]
        if not hit:
            fails.append(f"line_starts: no line with {needle!r}")
        elif not hit[0].startswith(prefix):
            fails.append(f"line_starts: line {hit[0][:60]!r} does not start with {prefix!r}")
    for group in ch.get("same_line", []):
        # On one line, in this order — a line that merely contains the
        # pieces (a whole contents table flattened into one paragraph) is not
        # a row.
        def in_order(line):
            pos = -1
            for s in group:
                i = norm(line).find(norm(s), pos + 1)
                if i < 0:
                    return False
                pos = i
            return True
        if not any(in_order(l) for l in lines):
            fails.append(f"same_line: {group} not on one line in order")
    chunks = [c for page in (rd or {}).get("pages", []) for c in page]
    items = [c for c in chunks if c.get("kind") == "list_item"]
    # An upstream check names an item with its marker (`1. Find the length`);
    # the reader chunk keeps the marker apart from `t`.
    for c in items:
        c["_t"] = norm(((c.get("marker") or "") + " " + c.get("t", "")).strip())
    for group in ch.get("one_list", []):
        refs = set()
        for s in group:
            hit = [c for c in items if norm(s) in c["_t"]]
            if not hit:
                fails.append(f"one_list: {s!r} is not in a list item")
                continue
            refs.add(hit[0].get("list"))
        if len(refs) > 1:
            fails.append(f"one_list: items spread over {len(refs)} lists: {group}")
    for pair in ch.get("separate_lists", []):
        refs = []
        for s in pair:
            hit = [c for c in items if norm(s) in c["_t"]]
            if not hit:
                fails.append(f"separate_lists: {s!r} is not in a list item")
            else:
                refs.append(hit[0].get("list"))
        if len(refs) == 2 and refs[0] == refs[1]:
            fails.append(f"separate_lists: {pair} are in one list")
    for s in ch.get("in_list", []):
        if not any(norm(s) in c["_t"] for c in items):
            fails.append(f"in_list: {s!r} is not a list item")
    for s in ch.get("not_in_list", []):
        if any(norm(s) in c["_t"] for c in items):
            fails.append(f"not_in_list: {s!r} is a list item")
    if case.get("list_items") is not None and len(items) != case["list_items"]:
        fails.append(f"list_items: {len(items)} != {case['list_items']}")
    return fails


# -------------------------------------------------------------- structure ---
#
# Structural parity with OpenDataLoader — the property a client needs when a
# page may be parsed on the device (the Dart port of OpenDataLoader) or on the
# server (this pipeline): the same sequence of spoken units with the same
# roles, not the same bytes. Both outputs are flattened to a common form —
# (role, normalized text, box) in reading order, body content only (page
# furniture is dropped by OpenDataLoader and kept with a role here) — and
# compared on four measures per fixture.

MARKER = re.compile(r"^(?:[\u2022\u2023\u25e6\u2043\u204c\u204d\u2219\u25aa\u25ab\u25cf\u25cb\-\*\+\u00b7\u2023o]|\(?\w{1,3}[.)])\s+")


def norm_text(s):
    s = norm(s).strip().lower()
    return MARKER.sub("", s)


def odl_elements(doc):
    """OpenDataLoader's JSON as (role, text, [l, b, r, t], depth)."""
    out = []

    def walk(node, in_cell, depth):
        if isinstance(node, list):
            for n in node:
                walk(n, in_cell, depth)
            return
        if not isinstance(node, dict):
            return
        t = node.get("type")
        box = node.get("bounding box")
        text = node.get("content") or ""
        if t == "heading":
            out.append(("heading", norm_text(text), box, 0))
        elif t == "paragraph":
            out.append(("table_cell" if in_cell else "paragraph", norm_text(text), box, 0))
        elif t == "caption":
            out.append(("caption", norm_text(text), box, 0))
        elif t == "list item":
            out.append(("list_item", norm_text(text), box, depth))
            walk(node.get("kids", []), in_cell, depth)
            return
        elif t == "list":
            walk(node.get("list items", []), in_cell, depth + 1)
            return
        elif t in ("header", "footer"):
            return  # furniture: not compared
        elif t == "table":
            for row in node.get("rows", []):
                walk(row.get("cells", []), True, depth)
            return
        elif t == "table cell":
            walk(node.get("kids", []), True, depth)
            return
        walk(node.get("kids", []), in_cell, depth)

    walk(doc.get("kids", []), False, 0)
    return [e for e in out if e[1]]


def reader_elements(rd):
    """The reader export as (role, text, [l, b, r, t], depth)."""
    out = []
    for page in (rd or {}).get("pages", []):
        for c in page:
            kind = c.get("kind")
            role = {"heading": "heading", "title": "heading", "paragraph": "paragraph", "list_item": "list_item",
                    "table_cell": "table_cell", "caption": "caption", "toc_entry": "paragraph",
                    "picture_text": "paragraph", "code": "paragraph", "footnote": "paragraph"}.get(kind)
            if role is None:
                continue
            text = norm_text(c.get("t", ""))
            if not text:
                continue
            out.append((role, text, [c["l"], c["b"], c["r"], c["tp"]], c.get("depth", 0)))
    return out


def tokens(s):
    return set(re.findall(r"[0-9a-z]+", s))


def iou(a, b):
    if not a or not b:
        return 0.0
    il, ib, ir, it = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ir <= il or it <= ib:
        return 0.0
    inter = (ir - il) * (it - ib)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def lcs(seq):
    """Length of the longest increasing subsequence of `seq` (matched server
    indices in OpenDataLoader order): how much of the matched order agrees."""
    import bisect
    tails = []
    for x in seq:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def structure(theirs, ours):
    """elements: share of their elements with a server element of the same text
    (token Jaccard ≥ 0.6, best match); roles: share of those with the same
    role; order: share of matches in the same relative order; boxes: mean
    overlap of matched pairs."""
    if not theirs:
        return None
    matches = []
    for i, (role, text, box, depth) in enumerate(theirs):
        tt = tokens(text)
        best, best_j = 0.0, None
        for j, (r2, t2, b2, d2) in enumerate(ours):
            o = tokens(t2)
            if not tt or not o:
                continue
            jac = len(tt & o) / len(tt | o)
            if jac > best:
                best, best_j = jac, j
        if best >= 0.6:
            matches.append((i, best_j))
    n = len(theirs)
    if not matches:
        return {"elements": 0.0, "roles": 0.0, "order": 0.0, "boxes": 0.0}
    same_role = sum(1 for i, j in matches if theirs[i][0] == ours[j][0])
    order = lcs([j for _, j in matches])
    boxes = sum(iou(theirs[i][2], ours[j][2]) for i, j in matches) / len(matches)
    return {"elements": round(len(matches) / n, 3), "roles": round(same_role / len(matches), 3),
            "order": round(order / len(matches), 3), "boxes": round(boxes, 3)}


def evaluate(draft, ours, fixtures, out_path, report_path):
    cases = [c for c in draft["cases"] if c.get("pdf") and (fixtures / c["pdf"]).exists()]
    recall = {}
    structures = {}
    rows = []
    for case in cases:
        rel = Path(case["pdf"]).with_suffix("")
        md_path = ours / rel.with_suffix(".md")
        rd_path = ours / (str(rel) + ".reader.json")
        md = md_path.read_text() if md_path.exists() else None
        rd = json.load(open(rd_path)) if rd_path.exists() else None
        odl = fixtures / rel.with_suffix(".md")
        if not odl.exists():
            odl = fixtures / (str(rel) + ".odl.md")
        if md is not None and odl.exists() and case["pdf"] not in recall:
            g = words(odl.read_text())
            recall[case["pdf"]] = round(len(g & words(md)) / len(g), 3) if g else 1.0
        odl_json = fixtures / (str(rel) + ".odl.json")
        if not odl_json.exists():
            odl_json = fixtures / rel.with_suffix(".json")
        if rd is not None and odl_json.exists() and case["pdf"] not in structures:
            structures[case["pdf"]] = structure(odl_elements(json.load(open(odl_json))), reader_elements(rd))
        if md is None:
            status, fails = "no-output", []
        elif not has_checks(case):
            status, fails = "unchecked", []
        else:
            fails = check(case, md, rd)
            status = "pass" if not fails else "xfail"
        case["status"] = status
        case["recall"] = recall.get(case["pdf"])
        case["structure"] = structures.get(case["pdf"])
        rows.append((status, case, fails))
    counts = {}
    for status, _, _ in rows:
        counts[status] = counts.get(status, 0) + 1
    out = {"upstream": draft.get("upstream", {}), "counts": counts, "cases": cases}
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if report_path:
        lines = ["# OpenDataLoader cases against docling.rs", "", f"Totals: {counts}", ""]
        if recall:
            lines += [f"Mean word recall over {len(recall)} fixtures: {sum(recall.values())/len(recall):.3f}", "",
                      "## Lowest recall", ""] + [f"- {k}: {v:.2f}" for k, v in sorted(recall.items(), key=lambda kv: kv[1])[:15]] + [""]
        st = {k: v for k, v in structures.items() if v}
        if st:
            mean = {m: sum(v[m] for v in st.values()) / len(st) for m in ("elements", "roles", "order", "boxes")}
            lines += [f"## Structural parity with OpenDataLoader over {len(st)} fixtures", "",
                      "mean: " + ", ".join(f"{m} {v:.3f}" for m, v in mean.items()), "",
                      "Lowest element match:", ""]
            lines += [f"- {k}: {v}" for k, v in sorted(st.items(), key=lambda kv: (kv[1]["elements"], kv[1]["roles"]))[:15]] + [""]
            lines += ["Lowest role agreement:", ""]
            lines += [f"- {k}: {v}" for k, v in sorted(st.items(), key=lambda kv: kv[1]["roles"])[:15]] + [""]
        lines += ["## Cases", "", "| status | id | fixture | detail |", "|---|---|---|---|"]
        for status, case, fails in rows:
            detail = "; ".join(fails)[:160] if fails else ("; ".join(case["unparsed"])[:120] if status == "unchecked" else "")
            lines.append(f"| {status} | {case['id']} | {case['pdf']} | {detail.replace('|', '/')} |")
        report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(counts))


def main():
    cmd = sys.argv[1]
    if cmd == "extract":
        upstream, fixtures, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
        commit = ""
        head = upstream / ".git" / "HEAD"
        if head.exists():
            ref = head.read_text().strip()
            commit = ref.split()[-1] if not ref.startswith("ref:") else (upstream / ".git" / ref.split()[1]).read_text().strip()
        cases = extract(upstream, fixtures)
        out.write_text(json.dumps({"upstream": {"repo": "https://github.com/opendataloader-project/opendataloader-pdf", "commit": commit[:7]}, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({"methods": len(cases), "with_pdf": sum(1 for c in cases if c["pdf"]), "with_checks": sum(1 for c in cases if has_checks(c))}))
    elif cmd == "evaluate":
        draft = json.load(open(sys.argv[2]))
        evaluate(draft, Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
