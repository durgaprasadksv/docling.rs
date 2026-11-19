"""Word boxes of the `reader` export against pdfium's character boxes.

Both are mapped to the frame a pdfium viewer draws (top-left origin, 1 px =
1 pt, the page's /Rotate applied, via FPDF_PageToDevice) — what a client that
displays pages with pdfium (pdfrx) highlights in. Each word of ours is paired
with the pdfium word of the same text nearest to it; reported per case: the
share matched, split differently (a superscript run like `Mg2+` is two words
here, one in pdfium) or absent from pdfium's text (OCR'd text: a scan, a
picture), the edge differences in points (median / 95th percentile) against
pdfium's loose (font ascent/descent) box, and how often our box's centre and
pdfium's tight box contain each other.

    pip install pypdfium2
    docling-rs --to reader --pages 760-761 --output out book.pdf
    python3 word_boxes_vs_pdfium.py name::book.pdf::out/book.reader.json ...
"""
import ctypes, json, sys, unicodedata, statistics as st
import pypdfium2 as pdfium, pypdfium2.raw as c

FOLD = str.maketrans({"‘":"'","’":"'","‚":"'","‛":"'","“":'"',"”":'"',"„":'"',
    "–":"-","—":"-","‐":"-","‑":"-","−":"-","­":"","\x02":"","￾":"","…":"..."})
def norm(s): return unicodedata.normalize("NFKC", s).translate(FOLD).strip()

SCALE = 64  # PageToDevice returns ints; scale up then divide for sub-point precision
def to_dev(page, W, H, x, y):
    dx, dy = ctypes.c_int(), ctypes.c_int()
    c.FPDF_PageToDevice(page.raw, 0, 0, int(W*SCALE), int(H*SCALE), 0, x, y, dx, dy)
    return dx.value/SCALE, dy.value/SCALE
def dev_box(page, W, H, l, b, r, t):
    pts = [to_dev(page, W, H, x, y) for x in (l, r) for y in (b, t)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))  # left, top, right, bottom (top-left origin)

def pdfium_words(page, W, H):
    tp = page.get_textpage(); n = c.FPDFText_CountChars(tp.raw)
    words, cur = [], None
    for i in range(n):
        ch = chr(c.FPDFText_GetUnicode(tp.raw, i) or 32)
        if c.FPDFText_IsGenerated(tp.raw, i) == 1 or ch.isspace() or ch in "\r\n":
            if cur: words.append(cur); cur = None
            continue
        rf = c.FS_RECTF(); c.FPDFText_GetLooseCharBox(tp.raw, i, rf)
        loose = dev_box(page, W, H, rf.left, rf.bottom, rf.right, rf.top)
        l, r, b, t = (ctypes.c_double() for _ in range(4))
        c.FPDFText_GetCharBox(tp.raw, i, l, r, b, t)
        tight = dev_box(page, W, H, l.value, b.value, r.value, t.value)
        if cur is None: cur = {"t": "", "loose": loose, "tight": tight}
        cur["t"] += ch
        u = lambda a, b2: (min(a[0], b2[0]), min(a[1], b2[1]), max(a[2], b2[2]), max(a[3], b2[3]))
        cur["loose"] = u(cur["loose"], loose); cur["tight"] = u(cur["tight"], tight)
    if cur: words.append(cur)
    return words

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix*iy; ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0
def ctr(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def compare(pdf, reader):
    doc = pdfium.PdfDocument(pdf); R = json.load(open(reader))
    rows = {"ours": 0, "matched": 0, "split": 0, "missing": 0, "pages": [], "d": []}
    for k, chunks in enumerate(R["pages"]):
        pno = R["firstPageNumber"] + k; page = doc[pno]
        W, H = page.get_size(); oW, oH = R["pageSizes"][k]
        rows["pages"].append((pno+1, round(W,2), round(H,2), oW, oH, page.get_rotation()))
        pw = pdfium_words(page, W, H); by = {}
        for w in pw: by.setdefault(norm(w["t"]), []).append(w)
        for ch in chunks:
            u16 = ch.get("t", "").encode("utf-16-le")
            for cs, ce, l, t, r, b in ch.get("w", []):
                text = norm(u16[2*cs:2*ce].decode("utf-16-le", "replace"))
                if not text: continue
                rows["ours"] += 1
                ob = (l, oH - t, r, oH - b)  # ours → top-left origin
                cands = by.get(text)
                if not cands:
                    near = [w for w in pw if text in norm(w["t"]) or (norm(w["t"]) and norm(w["t"]) in text)]
                    rows["split" if near else "missing"] += 1
                    if not near and len(rows.setdefault("miss_ex", [])) < 6: rows["miss_ex"].append(text)
                    continue
                oc = ctr(ob)
                w = min(cands, key=lambda w: (ctr(w["loose"])[0]-oc[0])**2 + (ctr(w["loose"])[1]-oc[1])**2)
                rows["matched"] += 1
                L, T = w["loose"], w["tight"]
                tc = ctr(T)
                rows["d"].append({
                    "dl": ob[0]-L[0], "dr": ob[2]-L[2], "dt_loose": ob[1]-L[1], "db_loose": ob[3]-L[3],
                    "dt_tight": ob[1]-T[1], "db_tight": ob[3]-T[3],
                    "center": ((oc[0]-tc[0])**2 + (oc[1]-tc[1])**2) ** .5,
                    "hit": T[0] <= oc[0] <= T[2] and min(T[1], ob[1]) <= oc[1] <= max(T[3], ob[3]) and ob[0] <= tc[0] <= ob[2] and ob[1] <= tc[1] <= ob[3],
                    "iou": iou(ob, L)})
    return rows

def q(v, p): v = sorted(v); return v[min(len(v)-1, int(p*len(v)))] if v else float("nan")
if __name__ == "__main__":
    print(f"{'case':14} {'words':>5} {'match':>6} {'split':>5} {'miss':>5} | {'|dl| p50/p95':>12} {'|dr| p50/p95':>12} {'top vs loose':>12} {'bot vs loose':>12} | {'ctr p95/max':>11} {'hit%':>5} {'IoU p50':>7}")
    for arg in sys.argv[1:]:
        name, pdf, reader = arg.split("::")
        r = compare(pdf, reader); d = r["d"]
        a = lambda k: [abs(x[k]) for x in d]
        f = lambda k: f"{q(a(k),.5):5.2f}/{q(a(k),.95):5.2f}"
        n = r["ours"] or 1
        print(f"{name:14} {r['ours']:5} {100*r['matched']/n:5.1f}% {100*r['split']/n:4.1f}% {100*r['missing']/n:4.1f}% | {f('dl'):>12} {f('dr'):>12} {f('dt_loose'):>12} {f('db_loose'):>12} | {q(a('center'),.95):5.2f}/{max(a('center') or [0]):5.1f} {100*sum(x['hit'] for x in d)/max(1,len(d)):5.1f} {q([x['iou'] for x in d],.5):7.3f}")
        bad = [p for p in r["pages"] if abs(p[1]-p[3]) > .01 or abs(p[2]-p[4]) > .01]
        if bad: print("   page size mismatch (page, pdfium W,H, ours W,H, rot):", bad)
        if r.get("miss_ex"): print("   unmatched e.g.:", r["miss_ex"])
