"""U2i (docs/U2I_CORPUS_ALIASES.md): corpus aliases, step 3 of the hyperlink-gap plan. Laptop only; nothing trains.

    python outputs/mp_unified/zu2i.py --selftest
    python outputs/mp_unified/zu2i.py pass --dataset D             -> outputs/u2i/<D>/pass.json, gate1.json
    python outputs/mp_unified/zu2i.py grade --dataset D --carve C [--graphs g1,g2,...]
    python outputs/mp_unified/zu2i.py choose                       -> outputs/u2i/choice.json
    python outputs/mp_unified/zu2i.py report                       -> outputs/u2i/report.md

A passage's first sentence often states its subject's other names. Three cues read them (K4a the restatement, K4b the
cue phrases, K4c the bracket acronym); each alias is a surface of kind 4 owned by its passage's title group, and links
under W2's gates, U2b's rule unchanged. The scan is zu2b.scan's code with the evidence steps 1 and 2 read (the first
offset, the occurrence count, the sentence of the first occurrence); the selftest checks it equal to zu2b.scan, and its
W2 links equal to zu2b.link_pass's. Gate 1: the rebuilt W2_S equals U2b's filed W2_S array for array.
"""
import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import time
from array import array
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402

DECLARED = "docs/U2I_CORPUS_ALIASES.md"
OUT = ROOT / "outputs" / "u2i"
U2B = ROOT / "outputs" / "u2b"
PASSAGE = zu2b.PASSAGE
RULE = zu2b.NEW_RULE
MIN_LEN = zu2b.MIN_LEN
AMBIG = zu2b.AMBIG
LEAD_MAX = 600
MAX_CHARS, MAX_TOKS = 80, 8
CHUNK = 200_000
CUES = ("a", "b", "c")
CANDS = {"A_c": ("c",), "A_b": ("b",), "A_a": ("a",), "A_abc": ("a", "b", "c")}   # declared order
GRAPHS = ("W2_S",) + tuple(CANDS) + ("PKG",)
TIE = 0.002
MIN_DEN = 0.005
CONNECT = frozenset("of the and de du des la le les von van der den del della di da y e & al el bin ibn for "
                    "upon".split())
SEPS = (" (", " is ", " was ", " are ", " were ", " has ", " had ", ", ", " – ", " — ", " - ", "; ", ": ")
CUE_PHRASES = ("also known as", "better known as", "commonly known as", "otherwise known as", "formerly known as",
               "known professionally as", "professionally known as", "known as", "also called", "commonly called",
               "often called", "or simply", "nicknamed", "stage name", "pen name", "born", "née", "nee", "formerly",
               "officially", "short for", "stylized as", "stylised as", "abbreviated as", "abbreviated")
CUE_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(c) for c in sorted(CUE_PHRASES, key=len, reverse=True))
                    + r")\b[:,]?\s*")
MONTHS = frozenset("January February March April May June July August September October November December".split())
END_RE = re.compile(r"[.!?](?=\s|$)")
CLOSERS = frozenset(")]\"'”’")
QUOTES = "\"'“”‘’"
OPENQ = {'"': '"', "“": "”", "‘": "’"}
TOKEN_RE = re.compile(r"\S+")
WORDS_RE = re.compile(r"\w+")
PAREN_RE = re.compile(r"\(([^()]{1,120})\)")
CONT = (", or ", " or ")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p, obj):
    zu2b.write_json(p, obj)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def codes():
    return {"zu2i_sha256": u1b.sha_src(__file__), "zu2b_sha256": u1b.sha_src(zu2b.__file__),
            "u1d_sha256": u1b.sha_src(u1d.__file__), "u1b_sha256": u1b.sha_src(u1b.__file__)}


# ── the aliases (section 2) ─────────────────────────────────────────────────


def _ends(text):
    """Sentence ends: '.', '!' or '?' before white space or the end, after a lowercase letter, digit, ')' ']' or a
    closing quote."""
    out = []
    for m in END_RE.finditer(text):
        p = m.start()
        if p > 0 and (text[p - 1].islower() or text[p - 1].isdigit() or text[p - 1] in CLOSERS):
            out.append(p)
    return out


def first_sentence(text):
    for m in END_RE.finditer(text):
        p = m.start()
        if p > LEAD_MAX:
            break
        if p > 0 and (text[p - 1].islower() or text[p - 1].isdigit() or text[p - 1] in CLOSERS):
            return text[:p]
    return text[:LEAD_MAX]


def cap_tok(t):
    return t[:1].isupper() or t[:1].isdigit()


def content_words(s):
    return {w for w in WORDS_RE.findall(s.lower()) if len(w) >= 3 and w not in u1b.STOP}


def restatement(fs):
    """K4a's opening span, or None."""
    cuts = [p for p in (fs.find(s) for s in SEPS) if p >= 0]
    if not cuts or min(cuts) <= 0:
        return None
    span = fs[:min(cuts)].strip().strip(QUOTES).strip()
    toks = span.split()
    if not 1 <= len(toks) <= MAX_TOKS or not cap_tok(toks[0]):
        return None
    if (toks[0].lower() in u1b.STOP and toks[0].lower() != "the") or any(t in MONTHS for t in toks):
        return None                                    # "On April 18, ...", "In 1990, ..."
    if not all(cap_tok(t) or t.lower() in CONNECT for t in toks):
        return None
    if not any(sum(c.isalpha() for c in t) >= 3 and t.lower() not in u1b.STOP for t in toks):
        return None
    return span


def take_run(s, e):
    """The alias at s[e:]: a quoted string, or the capitalized run. (alias, end) or (None, e)."""
    if e < len(s) and s[e] in OPENQ:
        f = s.find(OPENQ[s[e]], e + 1)
        if f < 0:
            return None, e
        return s[e + 1:f].strip(), f + 1
    toks, end = [], e
    for m in TOKEN_RE.finditer(s, e):
        t = m.group(0)
        stop = t[-1] in ",;:)."
        t = t.rstrip(",;:).") if stop else t
        if not t:
            break
        if not toks:
            if t == "the" and not stop and m.start() == e:     # "also called the North Atlantic Alliance"
                e = m.end() + 1
                continue
            if not t[:1].isupper() or t in MONTHS:             # a name, not a date ("born 12 July 1955")
                break
        elif not (cap_tok(t) or t.lower() in CONNECT):
            break
        toks.append(t)
        end = m.start() + len(t)
        if stop:
            break
    while toks and toks[-1].lower() in CONNECT:
        toks.pop()
    if not toks:
        return None, e
    alias = " ".join(toks)
    return alias, end


def cue_aliases(fs):
    """K4b: up to three aliases after each cue phrase."""
    out = []
    for m in CUE_RE.finditer(fs):
        e = m.end()
        for _ in range(3):
            a, e2 = take_run(fs, e)
            if a is None:
                break
            out.append(a)
            rest = fs[e2:]
            nxt = next((c for c in CONT if rest.startswith(c)), None)
            if nxt is None:
                break
            e = e2 + len(nxt)
    return out


def initials(s):
    return "".join(w[0].upper() for w in s.split() if w.lower() not in CONNECT and w[:1].isalpha())


def acronyms(fs, title_base, span):
    """K4c: bracket tokens whose capitals are the initials of the title or of the opening span."""
    refs = {x for x in (initials(title_base), initials(span) if span else "") if len(x) >= 2}
    out = []
    if not refs:
        return out
    for m in PAREN_RE.finditer(fs):
        for t in re.split(r"[\s,;:]+", m.group(1)):
            if not 2 <= len(t) <= 12 or not t[:1].isupper():
                continue
            up = "".join(c for c in t if c.isupper())
            if len(up) >= 2 and up in refs:
                out.append(t)
    return out


def own_forms(nt):
    return {nt, u1d.PAREN.sub("", nt).strip(), nt.split(",")[0].strip()}


def clean(a, own):
    a = a.strip().strip(QUOTES).strip()
    toks = a.split()
    if len(a) < MIN_LEN or len(a) > MAX_CHARS or not toks or len(toks) > MAX_TOKS or not cap_tok(toks[0]):
        return None
    return None if a in own or (a.startswith("The ") and a[4:] in own) else a


def mine(text, nt):
    """[(alias, cue, shares)] from a passage's first sentence; nt its normalized title. shares matters only for K4a."""
    fs = first_sentence(text)
    own = own_forms(nt)
    base = u1d.PAREN.sub("", nt).strip()
    out = []
    span = restatement(fs)
    if span is not None:
        a = clean(span, own)
        if a is not None:
            out.append((a, "a", bool(content_words(a) & content_words(base))))
    for a in cue_aliases(fs):
        a = clean(a, own)
        if a is not None:
            out.append((a, "b", True))
    for a in acronyms(fs, base, span):
        a = clean(a, own)
        if a is not None:
            out.append((a, "c", True))
    return out


# ── the table: titles (kind 0) and aliases (kind 4) ─────────────────────────


class Table:
    """The surfaces zu2b.scan reads (surf, lowid, n_low, table) with each surface's title group (kind 0) or its owner
    groups by cue (kind 4), and the title groups' nodes."""

    def __init__(self, titles, alias_owners):
        nt = [zu2b.norm_title(t, RULE).strip() for t in titles]
        g = {}
        self.group = np.asarray([g.setdefault(t, len(g)) for t in nt], np.int64)
        order = np.argsort(self.group, kind="stable")
        self.g_nodes = order
        self.g_ptr = np.searchsorted(self.group[order], np.arange(len(g) + 1))
        tsurf = {t for t in g if len(t) >= MIN_LEN}
        self.alias_owners = {a: o for a, o in alias_owners.items() if a not in tsurf}   # exact-first
        self.surf = sorted(tsurf | set(self.alias_owners))
        self.is_title = np.asarray([s in tsurf for s in self.surf], bool)
        self.tgroup = np.asarray([g[s] if s in tsurf else -1 for s in self.surf], np.int64)
        del g, nt
        lows = {}
        self.lowid = np.asarray([lows.setdefault(s.lower(), len(lows)) for s in self.surf], np.int64)
        self.n_low = len(lows)
        del lows
        self.table = zu2b.Table(self.surf, RULE)
        self.sha = hashlib.sha256("\n".join(self.surf).encode("utf-8")).hexdigest()

    def group_nodes(self, gid):
        return self.g_nodes[self.g_ptr[gid]:self.g_ptr[gid + 1]]


def scan_ev(S, text, rule=RULE):
    """zu2b.scan with each surface's first look-occurrence and its occurrence count: [(k, ok, first, n)], counts,
    capmid. The loop is zu2b.scan's line for line (n stops where zu2b's loop breaks)."""
    tl = text.lower()
    same = len(tl) == len(text)
    out, stat = [], {}
    capmid = set()
    done_low = set()
    for k in S.table.match(-1, text):
        sf = S.surf[k]
        low = S.table.low[k]
        lid = int(S.lowid[k])
        ok = False
        count = lid not in done_low
        done_low.add(lid)
        first, n = -1, 0
        for p in zu2b.occ_look(tl, low):
            if first < 0:
                first = p
            n += 1
            if not same:
                ok = True
                break
            sl = text[p:p + len(low)]
            if sl[1:] == sf[1:] and sl[0].lower() == sf[0].lower():
                ok = True
            mid = not u1d.sent_start(text, p)
            if mid and sl == sf and sf[:1].isupper():
                capmid.add(k)
            if count and mid:
                c = stat.setdefault(lid, [0, 0])
                if sl[0].isupper():
                    c[0] += 1
                elif sl == low:
                    c[1] += 1
        out.append((k, ok, first, n))
    return out, stat, capmid


# ── the pass (section 4) ────────────────────────────────────────────────────


def stream(served, d):
    """(i, title, text) over nodes.jsonl, its line count and sha256 checked against DATASET.json at the end."""
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    want, n = man["nodes"]["sha256"], int(man["nodes"]["n"])
    h = hashlib.sha256()
    i = -1
    with open(served / d / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            h.update(raw)
            rec = json.loads(raw)
            yield i, u1b.title_of(rec), rec["text"]
    if i + 1 != n or h.hexdigest() != want:
        raise SystemExit(f"{d}: nodes.jsonl is not DATASET.json's ({i + 1} lines, sha256 {h.hexdigest()})")


def mine_all(served, d):
    """Stream 1: titles, descriptive flags, text lengths and the mined (node, alias, cue, shares)."""
    titles, desc, tlen, found = [], bytearray(), array("i"), []
    for i, t, x in stream(served, d):
        titles.append(t)
        ds = zu2b.is_descriptive(t, x)
        desc.append(ds)
        tlen.append(len(x))
        if ds:
            for a, cue, sh in mine(x, zu2b.norm_title(t, RULE).strip()):
                found.append((i, a, cue, sh))
    return titles, np.frombuffer(desc, np.uint8).astype(bool), np.frombuffer(tlen, np.int32).copy(), found


def alias_table(found, group, desc):
    """{alias: {cue: sorted groups}} after K4a's sharing-or-single-passage condition."""
    n_desc = np.bincount(group[desc], minlength=int(group.max()) + 1)
    own = {}
    for i, a, cue, sh in found:
        g = int(group[i])
        if cue == "a" and not sh and n_desc[g] != 1:
            continue
        own.setdefault(a, {}).setdefault(cue, set()).add(g)
    return {a: {c: sorted(gs) for c, gs in o.items()} for a, o in own.items()}


def scan_chunk(S, texts, lo, desc):
    """Stream 2 for positions lo..: the case-passing hits with their evidence, and the chunk's capitalization counts."""
    src, ks, first, nocc, sent, cm = array("i"), array("i"), array("i"), array("H"), array("B"), array("B")
    stat = {}
    for off, x in enumerate(texts):
        i = lo + off
        if not desc[i]:
            continue
        hits, st, capm = scan_ev(S, x)
        for lid, (c, l_) in st.items():
            a = stat.setdefault(lid, [0, 0])
            a[0] += c
            a[1] += l_
        ends = None
        for k, ok, p, n in hits:
            if not ok:
                continue
            if ends is None:
                ends = _ends(x)
            src.append(i)
            ks.append(k)
            first.append(p)
            nocc.append(min(n, 65535))
            sent.append(min(bisect.bisect_right(ends, p - 1), 255))
            cm.append(k in capm)
    lids = np.asarray(sorted(stat), np.int64)
    return {"src": np.frombuffer(src, np.int32), "k": np.frombuffer(ks, np.int32),
            "first": np.frombuffer(first, np.int32), "nocc": np.frombuffer(nocc, np.uint16),
            "sent": np.frombuffer(sent, np.uint8), "capmid": np.frombuffer(cm, np.uint8).astype(bool),
            "lid": lids, "cap": np.asarray([stat[x][0] for x in lids.tolist()], np.int64),
            "low": np.asarray([stat[x][1] for x in lids.tolist()], np.int64)}


def scan_all(served, d, S, desc, work):
    """Stream 2 in chunks of CHUNK passages; a chunk whose file exists (same table) is not scanned again."""
    work.mkdir(parents=True, exist_ok=True)
    buf, lo, t0 = [], 0, time.time()

    def flush(lo_, texts):
        p = work / f"chunk_{lo_:09d}.npz"
        if p.exists():
            with np.load(p) as z:
                if str(z["table_sha"]) == S.sha:
                    return
            raise SystemExit(f"{p} was scanned with another table")
        arrs = scan_chunk(S, texts, lo_, desc)
        zu2b.save_npz(p, table_sha=np.asarray(S.sha), **arrs)
        log(f"{d}: chunk {lo_:,} done ({lo_ + len(texts):,} passages, {time.time() - t0:.0f} s)")

    for i, _t, x in stream(served, d):
        if not buf:
            lo = i
        buf.append(x)
        if len(buf) == CHUNK:
            flush(lo, buf)
            buf = []
    if buf:
        flush(lo, buf)


def load_chunks(work, S, n):
    files = sorted(work.glob("chunk_*.npz"))
    parts, cap, low = [], np.zeros(S.n_low, np.int64), np.zeros(S.n_low, np.int64)
    covered = 0
    for p in files:
        with np.load(p) as z:
            if str(z["table_sha"]) != S.sha:
                raise SystemExit(f"{p} was scanned with another table")
            np.add.at(cap, z["lid"], z["cap"])
            np.add.at(low, z["lid"], z["low"])
            parts.append({k: z[k] for k in ("src", "k", "first", "nocc", "sent", "capmid")})
        covered += 1
    if covered != -(-n // CHUNK):
        raise SystemExit(f"{work}: {covered} chunks, {-(-n // CHUNK)} wanted")
    hits = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    return hits, cap, low


def expand(S, src, grp):
    """(src, group) pairs to (src, node) pairs over each group's nodes, without the source itself."""
    cnt = (S.g_ptr[grp + 1] - S.g_ptr[grp]).astype(np.int64)
    s = np.repeat(src.astype(np.int64), cnt)
    starts = np.repeat(S.g_ptr[grp], cnt)
    off = np.arange(cnt.sum(), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    t = S.g_nodes[starts + off].astype(np.int64)
    keep = s != t
    return s[keep], t[keep]


def w2_links(S, hits, common):
    """W2's mention links (kind 0, no capmid): (src, dst), unique."""
    k = hits["k"]
    m = S.is_title[k] & ~common[S.lowid[k]]
    s, t = expand(S, hits["src"][m], S.tgroup[k[m]])
    key = np.unique(s * S.group.size + t)
    return key // S.group.size, key % S.group.size


def alias_links(S, hits, common, cues, dense_get):
    """One candidate's alias links: owners by the given cues, the AMBIG guard and the context choice (U2b's rule)."""
    k = hits["k"]
    m = ~S.is_title[k] & ~common[S.lowid[k]]
    idx = np.flatnonzero(m)
    surf = S.surf
    pairs, need = [], []
    for j in idx.tolist():
        i, kk = int(hits["src"][j]), int(k[j])
        o = S.alias_owners[surf[kk]]
        gs = sorted({g for c in cues for g in o.get(c, ())})
        gs = [g for g in gs if not (S.g_ptr[g + 1] - S.g_ptr[g] == 1 and S.g_nodes[S.g_ptr[g]] == i)]
        if not gs or len(gs) > AMBIG:
            continue
        if len(gs) == 1:
            pairs.append((i, gs[0]))
        else:
            need.append((i, tuple(gs)))
    amb = len(need)
    if need:
        srcs = np.asarray(sorted({i for i, _ in need}), np.int64)
        sv = dict(zip(srcs.tolist(), dense_get(srcs)))
        gcache = {}
        for i, gs in need:
            best, bg = -9.0, -1
            for g in gs:
                v = gcache.get(g)
                if v is None:
                    v = gcache[g] = dense_get(S.group_nodes(g))
                s_ = float((v @ sv[i]).max())
                if s_ > best:
                    best, bg = s_, g
            pairs.append((i, bg))
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), amb
    pa = np.asarray(pairs, np.int64)
    s, t = expand(S, pa[:, 0], pa[:, 1])
    key = np.unique(s * S.group.size + t)
    return key // S.group.size, key % S.group.size, amb


def ref_graph(d):
    """U2b's filed W2_S for d, fetched to outputs/u2i/<d>/ref/ and checked against build_new.json's sha256."""
    b = json.loads((U2B / d / "graphs" / "build_new.json").read_text(encoding="utf-8"))
    e = b["graphs"]["W2_S"]
    p = OUT / d / "ref" / e["file"]
    if not p.exists():
        return None, e
    if sha_file(p) != e["npz_sha256"]:
        raise SystemExit(f"{p}: not the filed W2_S ({e['npz_sha256'][:12]})")
    return p, e


def cmd_pass(a):
    d = a.dataset
    if d not in PASSAGE:
        raise SystemExit(f"{d}: U2i reads the passage datasets only ({PASSAGE})")
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    titles, desc, tlen, found = mine_all(served, d)
    n = len(titles)
    t1 = time.time()
    log(f"{d}: stream 1, {n:,} nodes, {len(found):,} cue hits ({t1 - t0:.0f} s)")
    g = {}
    group = np.asarray([g.setdefault(zu2b.norm_title(t, RULE).strip(), len(g)) for t in titles], np.int64)
    del g
    owners = alias_table(found, group, desc)
    del found
    S = Table(titles, owners)
    del titles
    t2 = time.time()
    log(f"{d}: table of {len(S.surf):,} surfaces ({int(S.is_title.sum()):,} titles, {len(S.alias_owners):,} aliases) "
        f"({t2 - t1:.0f} s)")
    work = OUT / d / "work"
    scan_all(served, d, S, desc, work)
    t3 = time.time()
    hits, cap, low = load_chunks(work, S, n)
    common = low > cap
    del S.table
    ws, wt = w2_links(S, hits, common)
    if RULE.descriptive:
        sib_s, sib_d = zu2b.sibling_pairs(group, desc)
    else:
        sib_s = sib_d = np.zeros(0, np.int64)
    ip, col = zu2b.undirected_csr(n, np.concatenate([ws, sib_s]), np.concatenate([wt, sib_d]))
    ref, rec = ref_graph(d)
    if ref is None:
        gate = {"status": "MISSING", "want": rec["file"]}
    else:
        with np.load(ref) as z:
            eq = bool(np.array_equal(z["indptr"], ip) and np.array_equal(z["col"], col))
        gate = {"status": "PASS" if eq else "FAIL", "file": rec["file"], "npz_sha256": rec["npz_sha256"],
                "directed_w2": int(ws.size), "filed_directed": rec["directed_edges"],
                "undirected": int(col.size), "filed_undirected": rec["undirected_neighbours"]}
    write_json(OUT / d / "gate1.json", {"dataset": d, "declared_in": DECLARED, **gate, **codes()})
    log(f"{d}: gate 1 {gate['status']} ({int(ws.size):,} W2 links)")
    del ip, col
    D = u1d.Dense(served, d, n)
    links, stats = {}, {}
    for cand, cues in CANDS.items():
        s_, t_, amb = alias_links(S, hits, common, cues, D.get)
        links[f"{cand}_src"], links[f"{cand}_dst"] = s_.astype(np.int32), t_.astype(np.int32)
        stats[cand] = {"alias_links": int(s_.size), "ambiguous_choices": amb}
        log(f"{d}: {cand} {int(s_.size):,} alias links ({amb:,} by the context choice)")
    lsha = zu2b.save_npz(OUT / d / "links.npz", w2_src=ws.astype(np.int32), w2_dst=wt.astype(np.int32),
                         sib_src=sib_s.astype(np.int32), sib_dst=sib_d.astype(np.int32), **links)
    al = sorted(S.alias_owners)
    cue_of = np.asarray([sum(1 << CUES.index(c) for c in S.alias_owners[x]) for x in al], np.uint8)
    o_ptr, o_col = zu2b.ragged([sorted({gg for gs in S.alias_owners[x].values() for gg in gs}) for x in al])
    asha = zu2b.save_npz(OUT / d / "aliases.npz", alias=np.asarray(al), cues=cue_of, owner_ptr=o_ptr,
                         owner_col=o_col)
    esha = zu2b.save_npz(OUT / d / "evidence.npz", **hits, surf_is_title=S.is_title, surf_lowid=S.lowid,
                         surf_tgroup=S.tgroup, low_cap=cap, low_low=low, group=group, desc=desc, text_len=tlen)
    by_cue = {c: sum(1 for x in al if c in S.alias_owners[x]) for c in CUES}
    rec = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": n,
           "descriptive_nodes": int(desc.sum()), "surfaces": len(S.surf), "titles": int(S.is_title.sum()),
           "aliases": len(al), "aliases_by_cue": by_cue, "table_sha256": S.sha, "hits": int(hits["k"].size),
           "common_forms": int(common.sum()), "w2_links": int(ws.size), "sibling_links": int(sib_s.size),
           "candidates": stats, "gate1": gate["status"], "links_npz_sha256": lsha, "aliases_npz_sha256": asha,
           "evidence_npz_sha256": esha, "secs": {"stream1": round(t1 - t0), "table": round(t2 - t1),
                                                  "scan": round(t3 - t2), "links": round(time.time() - t3)},
           **codes()}
    write_json(OUT / d / "pass.json", rec)
    log(f"{d}: pass done ({time.time() - t0:.0f} s)")
    return 0


# ── grading (section 5) ─────────────────────────────────────────────────────


def graphs_of(d, names):
    """{name: (indptr, col)}: W2_S from the gate-checked reference, a candidate as W2_S plus its alias links, PKG."""
    pas = json.loads((OUT / d / "pass.json").read_text(encoding="utf-8"))
    g1 = json.loads((OUT / d / "gate1.json").read_text(encoding="utf-8"))
    if g1["status"] != "PASS":
        raise SystemExit(f"{d}: gate 1 is {g1['status']}; REFUSED")
    if sha_file(OUT / d / "links.npz") != pas["links_npz_sha256"]:
        raise SystemExit(f"{d}: links.npz is not the pass's")
    ref, _e = ref_graph(d)
    with np.load(ref) as z:
        ip0, col0 = z["indptr"].astype(np.int64), z["col"]
    n = ip0.size - 1
    out = {}
    lk = np.load(OUT / d / "links.npz")
    for name in names:
        if name == "W2_S":
            out[name] = (ip0, col0)
        elif name == "PKG":
            qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
            p = U2B / d / "pkg_structural.npz"
            if hashlib.sha256(p.read_bytes()).hexdigest() != qrec["pkg_structural_npz_sha256"]:
                raise SystemExit(f"{d}: pkg_structural.npz is not its record's")
            with np.load(p) as z:
                out[name] = (z["indptr"].astype(np.int64), z["col"])
        else:
            a = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip0))
            ip, col = zu2b.undirected_csr(n, np.concatenate([a, lk[f"{name}_src"].astype(np.int64)]),
                                          np.concatenate([col0.astype(np.int64), lk[f"{name}_dst"].astype(np.int64)]))
            del a
            out[name] = (ip, col)
    return out


def cmd_grade(a):
    import m3b_pools
    from types import SimpleNamespace
    d, carve = a.dataset, a.carve
    names = a.graphs.split(",") if a.graphs else list(GRAPHS)
    t0 = time.time()
    served, _f = u1b.package_root(False)
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    qp = U2B / d / f"questions_{carve}.npz"
    if sha_file(qp) != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: the questions file is not its record's")
    q = np.load(qp)
    base = zu2b.unragged(q["base_ptr"], q["base_col"])
    seeds = zu2b.unragged(q["seeds_ptr"], q["seeds_col"])
    golds = zu2b.unragged(q["gold_ptr"], q["gold_col"])
    con = qrec["construction"]
    fams = [str(f) for f in q["fam_order"]]
    others = {f: SimpleNamespace(indptr=q[f"{f}_indptr"], col=q[f"{f}_col"]) for f in fams if f != "structural"}
    B = [np.union1d(b, s) for b, s in zip(base, seeds)]
    G = graphs_of(d, names)
    n = next(iter(G.values()))[0].size - 1
    D = u1d.Dense(served, d, n)
    t1 = time.time()
    res = {}
    for gname, (ip, col) in G.items():
        ent = {"I": None}
        if con["regime"] != "RETRIEVAL":
            st = SimpleNamespace(indptr=ip, col=col)
            fam_objs = [st if f == "structural" else others[f] for f in fams]
            al = rc = size = 0.0
            for j in range(len(golds)):
                sd = np.asarray(seeds[j], np.int64)
                exp = m3b_pools.expand_hops(sd, fam_objs, con["setting"])
                pool = np.union1d(np.union1d(np.asarray(base[j], np.int64), exp), sd)
                hit = np.isin(golds[j], pool)
                al += hit.all()
                rc += hit.mean()
                size += pool.size
            nq = len(golds)
            ent["I"] = {"ALL": round(al / nq, 4), "recall": round(rc / nq, 4), "pool": round(size / nq, 1)}
        ent["undirected_neighbours"] = int(col.size)
        res[gname] = ent
    t2 = time.time()
    curves = zu2b.reach_curves(G, B, golds, q["qemb"], D.get)
    for gname in G:
        res[gname].update(curves[gname])
        res[gname]["choice_score"] = round(zu2b.choice_score(curves[gname]["curve"]), 4)
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "questions": len(golds),
           "questions_npz_sha256": qrec["carves"][carve]["npz_sha256"], "dense_manifest_sha256": D.man_sha,
           "graphs": res, "secs_load": round(t1 - t0, 1), "secs_I": round(t2 - t1, 1),
           "secs_reach": round(time.time() - t2, 1), **codes()}
    tag = "" if not a.graphs else "_" + hashlib.sha1(a.graphs.encode()).hexdigest()[:8]
    write_json(OUT / d / f"grade_{carve}{tag}.json", out)
    log(f"{d}/{carve}: graded {', '.join(G)} ({time.time() - t0:.0f} s)")
    return 0


# ── the choice (section 6) and the report ───────────────────────────────────


def share(m, w, p):
    den = p - w
    return round((m - w) / den, 3) if den >= MIN_DEN else None


def measures(e):
    return {"I_ALL": e["I"]["ALL"] if e["I"] else None, "allX_ALL": e["all_neighbours"]["ALL"],
            "choice": e["choice_score"], "allX_rows": e["all_neighbours"]["rows"]}


def choose(grades):
    """grades: {dataset: grade record or None}; gate-1 refusals arrive as None."""
    missing = [d for d in PASSAGE if grades.get(d) is None or any(g not in grades[d]["graphs"] for g in GRAPHS)]
    if missing:
        return {"outcome": "REFUSED", "missing": missing}
    score = {g: round(float(np.mean([grades[d]["graphs"][g]["choice_score"] for d in PASSAGE])), 4)
             for g in GRAPHS if g != "PKG"}
    top = max(score[c] for c in CANDS)
    best = next(c for c in CANDS if score[c] >= top - TIE)
    falls = []
    for d in PASSAGE:
        w, b = measures(grades[d]["graphs"]["W2_S"]), measures(grades[d]["graphs"][best])
        if b["choice"] < w["choice"] - TIE:
            falls.append(f"{d} choice {b['choice']} < {w['choice']}")
        if w["I_ALL"] is not None and b["I_ALL"] < w["I_ALL"] - TIE:
            falls.append(f"{d} I {b['I_ALL']} < {w['I_ALL']}")
    ok = score[best] >= score["W2_S"] + TIE and not falls
    gap = {}
    for d in ("hotpotqa", "2wiki"):
        w, p = measures(grades[d]["graphs"]["W2_S"]), measures(grades[d]["graphs"]["PKG"])
        gap[d] = {c: {m: share(measures(grades[d]["graphs"][c])[m], w[m], p[m]) for m in ("I_ALL", "allX_ALL", "choice")
                      if w[m] is not None} for c in CANDS}
    return {"outcome": "ADOPT_FOR_POOLS" if ok else "NO_GAIN", "best": best, "score": score, "falls": falls,
            "gap_share": gap}


def cmd_choose(_a):
    grades = {}
    for d in PASSAGE:
        p = OUT / d / "grade_s1sel.json"
        g1 = OUT / d / "gate1.json"
        ok = g1.exists() and json.loads(g1.read_text(encoding="utf-8"))["status"] == "PASS"
        grades[d] = json.loads(p.read_text(encoding="utf-8")) if (p.exists() and ok) else None
    res = {"declared_in": DECLARED, "carve": "s1sel", **choose(grades), **codes()}
    write_json(OUT / "choice.json", res)
    print(json.dumps({k: res[k] for k in ("outcome", "best", "score") if k in res}, indent=1))
    return 0


def cmd_report(_a):
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    lines = ["# U2i report (docs/U2I_CORPUS_ALIASES.md)", "", f"Outcome: **{ch['outcome']}**"
             + (f", best {ch['best']}" if "best" in ch else ""), ""]
    for d in PASSAGE:
        p = OUT / d / "pass.json"
        if not p.exists():
            continue
        pa = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"## {d}", "", f"Gate 1: {pa['gate1']}. Aliases {pa['aliases']:,} (by cue {pa['aliases_by_cue']}); "
                  f"W2 links {pa['w2_links']:,}; alias links "
                  + ", ".join(f"{c} {v['alias_links']:,}" for c, v in pa["candidates"].items()), ""]
        for carve in ("s1sel", "s1eval"):
            gs = sorted(OUT.glob(f"{d}/grade_{carve}*.json"))
            for gp in gs:
                gr = json.loads(gp.read_text(encoding="utf-8"))
                lines += [f"{carve} ({gr['questions']} questions):", "",
                          "| graph | I's ALL | all of X: ALL | all of X: rows | choice score | k=1000 ALL |",
                          "| --- | --- | --- | --- | --- | --- |"]
                for g, e in gr["graphs"].items():
                    m = measures(e)
                    lines.append(f"| {g} | {m['I_ALL']} | {m['allX_ALL']} | {m['allX_rows']} | {m['choice']} | "
                                 f"{e['curve']['1000']['ALL']} |")
                lines.append("")
    if "gap_share" in ch:
        lines += ["## The share of the hyperlink gap closed (s1sel)", "", "| dataset | graph | I's ALL | all of X | "
                  "choice score |", "| --- | --- | --- | --- | --- |"]
        for d, per in ch["gap_share"].items():
            for c, s in per.items():
                lines.append(f"| {d} | {c} | {s.get('I_ALL')} | {s.get('allX_ALL')} | {s.get('choice')} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ────────────────────────────────────────────────────────────────


def _selftest(_a=None):
    # the cues
    assert first_sentence("Leeds United F.C. is a club. It plays.") == "Leeds United F.C. is a club"
    assert restatement("Leeds United Football Club is a club in Leeds") == "Leeds United Football Club"
    assert restatement("In 1990, the club won") is None
    assert restatement("On April 18, 1988, the awards were held") is None
    assert cue_aliases("released as The Yellow Winton Flyer in the UK; also known as The Flyer in Leeds") == [
        "The Flyer"]
    assert mine("Leeds United Football Club (LUFC) is a club. More.", "Leeds United F.C.") == [
        ("Leeds United Football Club", "a", True), ("LUFC", "c", True)]
    assert mine("North Atlantic Treaty Organization (NATO), also called the North Atlantic Alliance, is an alliance.",
                "North Atlantic Treaty Organization") == [("North Atlantic Alliance", "b", True), ("NATO", "c", True)]
    m = mine("Samuel Langhorne Clemens (1835–1910), known by his pen name Mark Twain, was a writer.", "Mark Twain")
    assert m == [("Samuel Langhorne Clemens", "a", False)], m
    assert cue_aliases('He was born John Smith, also known as "Johnny" or Big John, in 1950') == [
        "John Smith", "Johnny", "Big John"]
    assert cue_aliases("a Polish-born Jewish American engineer (born 12 July 1955; born October 13)") == []
    assert mine("The Great Comet of 1577 is a comet.", "Great Comet of 1577") == []
    assert acronyms("Organisation for Economic Co-operation and Development (OECD) is", "Organisation for Economic "
                    "Co-operation and Development", None) == ["OECD"]
    # the scan and W2's links against zu2b
    titles = ["Apple", "Orchard", "Apple Records", "Leeds United F.C.", "Leeds", "Apple", "Mark Twain",
              "North Atlantic Treaty Organization", "Leeds United Football Club (LUFC) is a club."]
    texts = ["Apple is a company. Apple Records too.", "In the orchard an apple fell, and Apple sued Apple Records.",
             "Apple Records is a label; Leeds United played Leeds.",
             "Leeds United Football Club (LUFC) is a club in Leeds. LUFC won.",
             "Leeds is a city. Apple and Orchard. The LUFC fans and Samuel Langhorne Clemens.",
             "Apple, the fruit. North Atlantic Alliance members.",
             "Samuel Langhorne Clemens (1835–1910), known by his pen name Mark Twain, was a writer.",
             "North Atlantic Treaty Organization (NATO), also called the North Atlantic Alliance, is an alliance.",
             "Leeds United Football Club (LUFC) is a club."]
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((len(titles), 1536)).astype(np.float32)

    def get(r):
        return vec[np.asarray(r, np.int64)].copy()

    g = {}
    group = np.asarray([g.setdefault(zu2b.norm_title(t, RULE).strip(), len(g)) for t in titles], np.int64)
    desc = np.asarray([zu2b.is_descriptive(t, x) for t, x in zip(titles, texts)], bool)
    assert not desc[8] and desc[:8].all()
    found = [(i, a_, c, sh) for i, (t, x) in enumerate(zip(titles, texts)) if desc[i]
             for a_, c, sh in mine(x, zu2b.norm_title(t, RULE).strip())]
    S = Table(titles, alias_table(found, group, desc))
    assert {"Leeds United Football Club", "LUFC", "NATO", "North Atlantic Alliance",
            "Samuel Langhorne Clemens"} <= set(S.alias_owners), sorted(S.alias_owners)
    for x in texts:
        a1, s1, c1 = zu2b.scan(S, x, RULE)
        a2, s2, c2 = scan_ev(S, x)
        assert a1 == [(k, ok) for k, ok, _f, _n in a2] and s1 == s2 and c1 == c2
    ch = scan_chunk(S, texts, 0, desc)
    cap = np.zeros(S.n_low, np.int64)
    low = np.zeros(S.n_low, np.int64)
    np.add.at(cap, ch["lid"], ch["cap"])
    np.add.at(low, ch["lid"], ch["low"])
    common = low > cap
    ws, wt = w2_links(S, ch, common)
    full = zu2b.Surfaces(titles, RULE)
    fc, fl = zu2b.stat_pass(full, texts, titles, RULE)
    e, _st = zu2b.link_pass(full, texts, titles, RULE, fl > fc, get, {"W2": ((0,), False)})
    ref = sorted(zip(e["W2"][0].tolist(), e["W2"][1].tolist()))
    assert sorted(zip(ws.tolist(), wt.tolist())) == ref, (ref, list(zip(ws.tolist(), wt.tolist())))
    s_, t_, _amb = alias_links(S, ch, common, ("a", "b", "c"), get)
    al = set(zip(s_.tolist(), t_.tolist()))
    assert (4, 3) in al and (4, 6) in al and (5, 7) in al, al     # LUFC -> Leeds United F.C.; Clemens; the Alliance
    assert (3, 3) not in al
    s_, t_, _amb = alias_links(S, ch, common, ("c",), get)
    assert set(zip(s_.tolist(), t_.tolist())) == {(4, 3)}, set(zip(s_.tolist(), t_.tolist()))
    # the choice
    def gr(cs, ia):
        return {"graphs": {gname: {"choice_score": cs.get(gname, 0.5), "I": {"ALL": ia.get(gname, 0.5)},
                                   "all_neighbours": {"ALL": 0.5, "rows": 1.0}} for gname in GRAPHS}}
    base = {d: gr({"PKG": 0.6}, {}) for d in PASSAGE}
    assert choose(base)["outcome"] == "NO_GAIN"
    up = {d: gr({"A_abc": 0.51, "A_b": 0.509, "PKG": 0.6}, {}) for d in PASSAGE}
    r = choose(up)
    assert r["outcome"] == "ADOPT_FOR_POOLS" and r["best"] == "A_b", r
    assert r["gap_share"]["hotpotqa"]["A_b"]["choice"] == 0.09
    fall = dict(up)
    fall["squad"] = gr({"A_abc": 0.51, "A_b": 0.509, "PKG": 0.6}, {"A_b": 0.49})
    assert choose(fall)["outcome"] == "NO_GAIN"
    assert choose({**up, "2wiki": None})["outcome"] == "REFUSED"
    print("zu2i selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--carve", default="s1sel")
    ap.add_argument("--graphs", default="")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return _selftest(a)
    return {"pass": cmd_pass, "grade": cmd_grade, "choose": cmd_choose, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
