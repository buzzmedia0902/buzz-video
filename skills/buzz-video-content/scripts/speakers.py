#!/usr/bin/env python3
"""Tag cues with WHO is speaking - decided by the voice (ECAPA speaker embeddings), never by the picture.

  python speakers.py edit.json [--src ep] [--range A B] [--min-margin 0.15] [--min-cos 0.5] [--min-dur 1.0]
                     [--check] [--dry-run] [--force]

edit.json   "speakers": {"山田": {"color": "#FF7A59", "refs": [["ep", 234.6, 249.9], ["ep", 283.6, 292.2]]}, ...}
  refs = [source id, start s, end s]: 2-4 clean lines per person, 5-15 s each (longer = firmer: 4 s scores ~0.6,
  6-15 s 0.8+): one voice only, no laughter, no overlap. Pick them from close-ups (still frames + transcript),
  NOT from the name plate: the plate stays on one person while others shout. Two voices that sound alike (twins)
  must be separate speakers; the refs check will say when they are too close. Define everybody who talks in the range (the runner-up is what stops wrong tags);
  people without refs, the narrator and BGM simply stay untagged.

Every cue is embedded (long cues in 2 s windows) and compared with each speaker's centroid. It is tagged only if
cos >= --min-cos AND it beats the runner-up by --min-margin AND its windows agree; otherwise the speaker stays
null - never a guess. Untagged reasons: 短 shorter than --min-dur, 笑 laughter / applause / music marker,
低 no known voice, 差 two voices too close, 混 voices inside the cue disagree (split the cue and re-run).
A reference that fails the leave-one-out check (it sounds like somebody else, or unlike its own other refs) is
dropped and reported. Tags set by hand (a "speaker" without "voice", or "locked": true) are kept, but the table
flags every hand tag the voice contradicts (<<<): check those against the voice, not the face or name plate.

Writes cues.json (previous: cues.json.bak): "speaker" plus "voice": [cos, margin] on cues it tagged. --check only
verifies the refs; --dry-run prints the table without writing; on a long video use --range A B or --no-table. Runs under the speaker env (torch, ~2 GB, built once
by env.sh --speakers) - it re-launches itself there. Embeddings are cached in work/spk/<src>.npz.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 16000
BATCH = 16
MODEL_DIR = os.path.expanduser(os.environ.get("CV_SPK_MODEL", "~/.local/share/content-video/spk_model"))
CR = "\r" if sys.stderr.isatty() else "\n"   # progress overwrites itself on a terminal, one line each in a log
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_env() -> None:
    """Re-run under the speaker env (torch + speechbrain) when this python has no torch."""
    if importlib.util.find_spec("torch") and importlib.util.find_spec("speechbrain"):
        return
    if os.environ.get("CV_SPK_CHILD"):
        raise SystemExit("torch / speechbrain are missing in the speaker env: bash env.sh --speakers")
    from cv import osdep
    out = subprocess.run([osdep.bash(), osdep.script_arg(os.path.join(HERE, "env.sh")), "--speakers"],
                         stdout=subprocess.PIPE, text=True)
    if out.returncode:
        raise SystemExit("the speaker env could not be built (see the messages above)")
    py = out.stdout.strip().splitlines()[-1]
    print(f"[speakers] running under {py}", file=sys.stderr, flush=True)
    env = {**os.environ, "CV_SPK_CHILD": "1", "PYTHONUTF8": "1"}
    if osdep.WIN:  # execve does not replace the process on Windows: wait for the child instead
        sys.exit(subprocess.run([py, os.path.abspath(__file__), *sys.argv[1:]], env=env).returncode)
    os.execve(py, [py, os.path.abspath(__file__), *sys.argv[1:]], env)


class Embedder:
    """ECAPA (speechbrain spkrec-ecapa-voxceleb) unit embeddings of source ranges, cached in work/spk/<src>.npz."""

    def __init__(self, project):
        self.p = project
        self._model = None
        self._dev = os.environ.get("CV_SPK_DEVICE", "")
        self._wav: dict[str, np.ndarray] = {}
        self._cache: dict[str, dict] = {}
        self._new: set[str] = set()

    def _wave(self, sid: str) -> np.ndarray:
        if sid not in self._wav:
            from scipy.io import wavfile
            from cv import media
            wav = media.extract_audio(self.p.src_path(sid), self.p.work("audio", f"{sid}_16k.wav"), sr=SR)
            self._wav[sid] = wavfile.read(wav, mmap=True)[1]
        return self._wav[sid]

    def _store(self, sid: str) -> dict:
        if sid not in self._cache:
            self._cache[sid] = {}
            path = self.p.work("spk", f"{sid}.npz")
            wav = self.p.work("audio", f"{sid}_16k.wav")
            if os.path.exists(path) and os.path.exists(wav) and os.path.getmtime(path) >= os.path.getmtime(wav):
                d = np.load(path)
                self._cache[sid] = {(float(a), float(b)): e for (a, b), e in zip(d["k"], d["e"])}
        return self._cache[sid]

    def _load(self):
        """The ECAPA model on the Apple GPU when there is one (15x faster than the CPU, same embeddings)."""
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self._dev = self._dev or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=MODEL_DIR,
                                                     run_opts={"device": self._dev})

    def _encode(self, sid: str, batch: list[tuple[float, float]]) -> list[np.ndarray]:
        """Embeddings of ranges that all have the same length in samples (no padding: results do not depend on
        which ranges happen to share a batch)."""
        import torch
        if self._model is None:
            self._load()
        w = self._wave(sid)
        n = int(batch[0][1] * SR) - int(batch[0][0] * SR)
        X = np.zeros((len(batch), n), np.float32)
        for i, (a, b) in enumerate(batch):  # a range running past the end of the audio is zero-padded
            x = w[max(0, int(a * SR)):][:n]
            X[i, :len(x)] = x
        X = torch.from_numpy(X)
        try:
            with torch.inference_mode():
                e = self._model.encode_batch(X.to(self._dev)).squeeze(1).cpu().numpy()
        except RuntimeError as err:
            if self._dev == "cpu":
                raise
            print(f"\n[speakers] {self._dev} failed ({str(err)[:80]}): continuing on the CPU", file=sys.stderr)
            self._model, self._dev = None, "cpu"
            return self._encode(sid, batch)
        return [v / (np.linalg.norm(v) + 1e-9) for v in e]

    def embed(self, sid: str, ranges: list[tuple[float, float]], label: str = "") -> np.ndarray:
        """(len(ranges), dim) unit embeddings of source ranges; new ones are computed (and cached) in batches of
        equal length."""
        store = self._store(sid)
        keys = [(round(a, 3), round(b, 3)) for a, b in ranges]
        by_len: dict[int, list] = defaultdict(list)
        for k in dict.fromkeys(k for k in keys if k not in store):
            by_len[int(k[1] * SR) - int(k[0] * SR)].append(k)
        jobs = [g[i:i + BATCH] for g in by_len.values() for i in range(0, len(g), BATCH)]
        total, done = sum(len(j) for j in jobs), 0
        for n, job in enumerate(jobs, 1):
            for k, e in zip(job, self._encode(sid, job)):
                store[k] = e
            self._new.add(sid)
            done += len(job)
            if n % 40 == 0:  # a long run survives an interruption: keep what is computed so far
                self.save()
                if label:
                    print(f"[speakers] {label}: {done}/{total}", end=CR, file=sys.stderr, flush=True)
        if label and total:
            print(f"[speakers] {label}: {total}/{total}", file=sys.stderr)
        return np.stack([store[k] for k in keys])

    def save(self) -> None:
        for sid in self._new:
            store = self._cache[sid]
            np.savez(self.p.work("spk", f"{sid}.npz"), k=np.array(list(store), dtype=np.float64),
                     e=np.stack(list(store.values())))


def load_refs(project, emb: Embedder) -> list:
    """Reference ranges of edit.json 'speakers' -> embedded Ref objects (validated)."""
    from cv import media
    from cv.speakers_lib import Ref, chunks
    refs = []
    for name, spk in project.speakers.items():
        for r in spk.get("refs", []):
            if not (isinstance(r, list) and len(r) == 3):
                raise SystemExit(f"speakers.{name}.refs: each ref is [source id, start s, end s] (got {r})")
            sid, t0, t1 = r[0], float(r[1]), float(r[2])
            project.src_path(sid)
            if t1 - t0 < 1.5 or t1 > media.probe(project.src_path(sid))["duration"] + 0.01:
                raise SystemExit(f"speakers.{name}.refs: {sid} {t0}-{t1}: need >= 1.5 s inside the source")
            refs.append(Ref(name, sid, t0, t1, emb.embed(sid, chunks(t0, t1))))
    return refs


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def print_refs(reports: list, kept: list) -> None:
    from cv.speakers_lib import centroids, pad
    used = {id(r) for r in kept}
    print("参照チェック（自分 = 同じ人の他の参照との類似 / 他人 = 最も近い別人）")
    for rp in reports:
        r = rp.ref
        own = "  - " if rp.own is None else f"{rp.own:.2f}"
        other = f"{rp.other:.2f} {rp.other_name}" if rp.other_name else "-"
        flag = "OK" if rp.ok else ("NG 除外" if id(r) not in used else "NG（代わりがないのでそのまま使用）")
        print(f"  {pad(r.speaker, 8)} {r.src} {mmss(r.t0)}-{mmss(r.t1)} ({r.t1 - r.t0:4.1f}s)  自分 {own}  他人 {other}  {flag}")
    c = centroids(kept)
    pairs = [(a, b, float(c[a] @ c[b])) for i, a in enumerate(c) for b in list(c)[i + 1:]]
    if pairs:
        print("声の近さ: " + "  ".join(f"{a}-{b} {v:.2f}" for a, b, v in pairs))
    for a, b, v in pairs:
        if v >= 0.6:
            print(f"  ! {a} と {b} は声が近い（{v:.2f}）: 2人の間は未判定が増える。参照を選び直す")


def is_manual(c: dict) -> bool:
    """A tag somebody set by hand: a speaker without our 'voice' score (or an explicit lock)."""
    return bool(c.get("locked") or (c.get("speaker") and "voice" not in c))


def analyse(cues: list[dict], todo: list[int], emb: Embedder, cents: dict, a) -> dict[int, object]:
    """Verdict per cue index in todo (embeds every analysis window of the cues that need a voice check)."""
    from cv.speakers_lib import Verdict, is_laugh, judge, spans
    out, plan = {}, defaultdict(list)
    for i in todo:
        c = cues[i]
        if float(c["t1"]) - float(c["t0"]) < a.min_dur:
            out[i] = Verdict(None, "short")
        elif is_laugh(c.get("text", "")):
            out[i] = Verdict(None, "laugh")
        else:
            plan[c["src"]].append((i, spans(float(c["t0"]), float(c["t1"]))))
    for sid, items in plan.items():
        flat = [w for _, ws in items for w in ws]
        e = emb.embed(sid, flat, label=f"{sid} 声の埋め込み")
        k = 0
        for i, ws in items:
            out[i] = judge(e[k:k + len(ws)], cents, a.min_cos, a.min_margin)
            k += len(ws)
    return out


def print_table(cues: list[dict], verdicts: dict, names: list) -> None:
    from cv.speakers_lib import WHY_JA, clip_text, legend, pad
    let = legend(names)
    print("\n  " + "  ".join(f"{L}={n}" for n, L in let.items()) + "   （窓: 長いキューを2秒窓で見た各窓の声）")
    head = f"{'#':>4} {'時刻':<14}{'秒':>5}  {'判定':<8}{'cos':>5}{'差':>5}  " + " ".join(pad(n[:4], 5) for n in names) + "  窓" + " " * 10 + "本文"
    print(head)
    for i in sorted(verdicts):
        c, v = cues[i], verdicts[i]
        t0, t1 = float(c["t0"]), float(c["t1"])
        manual = is_manual(c)
        mark = ""
        if manual:
            label = f"手:{c.get('speaker') or '-'}"
            if v.label and c.get("speaker") and v.label != c["speaker"]:
                mark = f"  <<< 声は{v.label}（{v.cos:.2f}）手動タグと違う？"
        else:
            label = v.label or f"({WHY_JA[v.why]})"
        sims = " ".join(pad(f"{v.sims[n]:.2f}" if n in v.sims else "  - ", 5) for n in names)
        cos = f"{v.cos:.2f}" if v.sims else "  - "
        mg = f"{v.margin:.2f}" if v.sims else "  - "
        print(f"{i:>4} {mmss(t0) + '-' + f'{t1 % 60:04.1f}':<14}{t1 - t0:5.1f}  {pad(label, 8)}{cos:>5}{mg:>5}  {sims}  "
              f"{pad(v.wins if len(v.wins) <= 12 else v.wins[:11] + '+', 13)}{clip_text(c.get('text', ''), 34)}{mark}")


def print_summary(cues: list[dict], verdicts: dict, names: list) -> None:
    from cv.speakers_lib import WHY_HELP, WHY_JA
    secs, count, why = defaultdict(float), defaultdict(int), defaultdict(int)
    wrong = []
    for i, v in verdicts.items():
        c = cues[i]
        d = float(c["t1"]) - float(c["t0"])
        who = c.get("speaker") if is_manual(c) else v.label
        if is_manual(c) and v.label and c.get("speaker") and v.label != c["speaker"]:
            wrong.append(i)
        if who:
            secs[who] += d
            count[who] += 1
        else:
            why[WHY_JA["manual" if is_manual(c) else v.why]] += 1
    total = sum(float(cues[i]["t1"]) - float(cues[i]["t0"]) for i in verdicts)
    print(f"\n合計 {len(verdicts)} 本 / {total:.1f} 秒")
    for n in names:
        print(f"  {n:<8}{count[n]:>4} 本 {secs[n]:7.1f} 秒")
    left = sum(why.values())
    print(f"  未判定  {left:>4} 本（" + " / ".join(f"{k} {v}" for k, v in why.items()) + "）")
    print("  " + "  ".join(f"{k}={v}" for k, v in WHY_HELP.items()))
    if wrong:
        print(f"  ! 手動タグと声が食い違う: #{', #'.join(map(str, wrong))}（顔・名札ではなく声で再確認）")


def main() -> None:
    from cv.speakers_lib import MIN_COS, MIN_DUR, MIN_MARGIN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--src", nargs="*", help="cue sources to tag (default: all)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("A", "B"), help="only cues overlapping A-B source seconds")
    ap.add_argument("--min-cos", type=float, default=MIN_COS, help="best similarity needed (default %(default)s)")
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN, help="lead over the runner-up (default %(default)s)")
    ap.add_argument("--min-dur", type=float, default=MIN_DUR, help="shorter cues stay untagged (default %(default)s s)")
    ap.add_argument("--check", action="store_true", help="only verify the references")
    ap.add_argument("--dry-run", action="store_true", help="print the table, do not write cues.json")
    ap.add_argument("--force", action="store_true", help="also overwrite tags set by hand (not 'locked' cues)")
    ap.add_argument("--no-table", action="store_true", help="summary only")
    a = ap.parse_args()
    ensure_env()
    from cv import spec
    from cv.speakers_lib import centroids, check_refs, usable_refs
    p = spec.load(a.project)
    if not any(s.get("refs") for s in p.speakers.values()):
        raise SystemExit('edit.json "speakers" has no refs: {"山田": {"refs": [["ep", 234.6, 249.9], ...]}} (see --help)')
    emb = Embedder(p)
    refs = load_refs(p, emb)
    reports = check_refs(refs, a.min_cos, a.min_margin)
    kept = usable_refs(reports)
    print_refs(reports, kept)
    cents = centroids(kept)
    names = list(cents)
    if len(names) < 2:
        print("  ! 話者が1人だけ: 比べる相手がいないので、しきい値だけの判定になる（弱い）。同じ場面の他の人も refs に足す")
    if a.check:
        emb.save()
        return
    cues_path = p.cues_path or os.path.join(p.root, "cues.json")
    cues = json.load(open(cues_path, encoding="utf-8"))
    srcs = set(a.src or {c["src"] for c in cues})
    lo, hi = a.range or (0.0, 1e12)
    todo = [i for i, c in enumerate(cues) if c["src"] in srcs and not c.get("cut") and c.get("text")
            and float(c["t1"]) > lo and float(c["t0"]) < hi]
    if not todo:
        raise SystemExit("no cue to check: look at --src / --range, and that cues.json has text for this source")
    verdicts = analyse(cues, todo, emb, cents, a)
    emb.save()
    if not a.no_table:
        print_table(cues, verdicts, names)
    print_summary(cues, verdicts, names)
    if a.dry_run:
        return
    for i, v in verdicts.items():
        c = cues[i]
        if c.get("locked") or (is_manual(c) and not a.force):
            continue
        c["speaker"] = v.label
        if v.label:
            c["voice"] = [round(v.cos, 3), round(v.margin, 3)]
        else:
            c.pop("voice", None)
    if os.path.exists(cues_path):
        os.replace(cues_path, cues_path + ".bak")
    json.dump(cues, open(cues_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {cues_path}（前の版: {cues_path}.bak）", file=sys.stderr)


if __name__ == "__main__":
    main()
