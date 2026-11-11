"""
Phase 2 — measure how accurate Visual Match Detector really is.

Runs both face models over the CACD-VS cross-age benchmark
(2,000 same-person pairs + 2,000 different-person pairs, 10 folds)
and reports accuracy, error rates, and recommended thresholds.

Usage:
    python evaluate.py R:\\CACD_VS
    python evaluate.py R:\\CACD_VS --limit 200     # quick trial run

CACD-VS layout (from the dataset authors):
    images are named  <pairId>_0.jpg  and  <pairId>_1.jpg
    every 400 pairs form one fold:
        first 200 pairs  -> same person
        next  200 pairs  -> different people
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

from face_engine import FaceEngine, THRESHOLDS, load_image

FOLD_SIZE = 400
POSITIVES_PER_FOLD = 200


# ---------------------------------------------------------------- dataset
def find_pairs(root):
    """Return a sorted list of (pair_id, path_0, path_1)."""
    pairs = []
    for path_0 in glob.glob(os.path.join(root, "**", "*_0.jpg"), recursive=True):
        name = os.path.basename(path_0)[:-len("_0.jpg")]
        if not name.isdigit():
            continue
        path_1 = path_0[:-len("_0.jpg")] + "_1.jpg"
        if os.path.exists(path_1):
            pairs.append((int(name), path_0, path_1))
    return sorted(pairs)


def is_same_person(pair_id):
    return (pair_id % FOLD_SIZE) < POSITIVES_PER_FOLD


# ---------------------------------------------------------------- scoring
def score_pairs(engine, pairs, cache_path):
    """Compute both models' scores for every pair. Results are cached in a CSV,
    so an interrupted run continues where it stopped."""
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["pair_id"])] = row

    new_file = not os.path.exists(cache_path)
    with open(cache_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "fold", "same", "arcface", "sface", "status"])
        if new_file:
            writer.writeheader()

        todo = [p for p in pairs if p[0] not in done]
        start = time.time()
        for i, (pair_id, path_0, path_1) in enumerate(todo, 1):
            row = {"pair_id": pair_id, "fold": pair_id // FOLD_SIZE,
                   "same": int(is_same_person(pair_id)), "arcface": "", "sface": "", "status": "ok"}
            try:
                a = engine.analyze(load_image(path_0))
                b = engine.analyze(load_image(path_1))
                if a is None or b is None:
                    row["status"] = "no_face"
                else:
                    v = engine.compare(a, b)
                    row["sface"] = f"{v.sface_score:.5f}"
                    if v.arcface_score is not None:
                        row["arcface"] = f"{v.arcface_score:.5f}"
            except Exception as exc:
                row["status"] = f"error: {exc}"
            writer.writerow(row)
            done[pair_id] = row

            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                left = (len(todo) - i) / rate
                print(f"\r[*] {i}/{len(todo)} pairs  ·  {rate:.1f} pairs/s  ·  ~{left/60:.1f} min left",
                      end="", flush=True)
        if todo:
            print()
    return [done[p[0]] for p in pairs]


# ---------------------------------------------------------------- metrics
def best_threshold(scores, labels):
    """Threshold that gives the highest accuracy on these pairs."""
    s = np.unique(scores)
    # test the midpoints between neighbouring scores, so the threshold
    # sits in the gap instead of exactly on a training sample
    candidates = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        acc = np.mean((scores >= t) == labels)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def kfold_accuracy(scores, labels, folds):
    """Standard benchmark protocol: pick the threshold on 9 folds, test on the 10th."""
    if len(np.unique(folds)) < 2:
        # a small sample can fall inside one fold: split it into 10 balanced folds instead
        folds = np.zeros(len(scores), dtype=int)
        for cls in (True, False):
            idx = np.where(labels == cls)[0]
            folds[idx] = np.arange(len(idx)) % 10
    accs, thresholds = [], []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        t, _ = best_threshold(scores[train], labels[train])
        accs.append(np.mean((scores[test] >= t) == labels[test]))
        thresholds.append(t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thresholds))


def auc(scores, labels):
    """Area under the ROC curve: chance that a random same-person pair
    scores higher than a random different-person pair."""
    pos, neg = scores[labels], scores[~labels]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_for_false_match_rate(scores, labels, rate):
    """Lowest threshold where at most `rate` of different-person pairs are called a match."""
    impostor = np.sort(scores[~labels])
    k = int(np.ceil((1 - rate) * len(impostor))) - 1
    return float(impostor[k]) + 1e-6      # just above the k-th impostor score


def verdict_table(arc, sf, labels, use_second_opinion):
    """Count verdicts using the app's real decision rule and current thresholds."""
    def label(s, model):
        strong, possible = THRESHOLDS[model]
        return np.where(s >= strong, "match", np.where(s >= possible, "review", "no_match"))

    final = label(arc, "arcface")
    if use_second_opinion:
        sl = label(sf, "sface")
        opposite = ((final == "match") & (sl == "no_match")) | ((final == "no_match") & (sl == "match"))
        final = np.where(opposite, "review", final)

    table = {}
    for name, mask in (("same person", labels), ("different people", ~labels)):
        n = mask.sum()
        table[name] = {v: (int((final[mask] == v).sum()), (final[mask] == v).mean() * 100 if n else 0)
                       for v in ("match", "review", "no_match")}
    return table


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description="Evaluate Visual Match Detector on CACD-VS")
    ap.add_argument("dataset", help="folder that contains the CACD-VS images")
    ap.add_argument("--limit", type=int, help="only use the first N pairs (quick test)")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    args = ap.parse_args()

    pairs = find_pairs(args.dataset)
    if not pairs:
        sys.exit(f"[-] No CACD-VS pairs (*_0.jpg / *_1.jpg) found in {args.dataset}")
    if args.limit:
        # keep the sample balanced: half same-person, half different-people
        # spread the sample across all folds, half same-person and half different-people
        def spread(items, n):
            step = max(1, len(items) // max(1, n))
            return items[::step][:n]
        same = spread([p for p in pairs if is_same_person(p[0])], args.limit // 2)
        diff = spread([p for p in pairs if not is_same_person(p[0])], args.limit // 2)
        pairs = sorted(same + diff)
    print(f"[+] {len(pairs)} pairs found")

    os.makedirs(args.out, exist_ok=True)
    engine = FaceEngine()
    if engine.arcface is None:
        sys.exit(f"[-] ArcFace is required for evaluation: {engine.arcface_error}")

    rows = score_pairs(engine, pairs, os.path.join(args.out, "scores.csv"))
    ok = [r for r in rows if r["status"] == "ok"]
    failed = len(rows) - len(ok)

    labels = np.array([r["same"] == "1" or r["same"] == 1 for r in ok])
    folds = np.array([int(r["fold"]) for r in ok])
    arc = np.array([float(r["arcface"]) for r in ok])
    sf = np.array([float(r["sface"]) for r in ok])

    lines = []
    out = lines.append
    out("# Visual Match Detector — CACD-VS evaluation\n")
    out(f"Pairs evaluated: **{len(ok)}** "
        f"({labels.sum()} same person, {(~labels).sum()} different people)")
    if failed:
        out(f"Pairs skipped (no face found / unreadable): **{failed}**")
    out("")

    out("## 1. Accuracy (10-fold protocol)\n")
    out("| Model | Accuracy | AUC | Best threshold |")
    out("|---|---|---|---|")
    for name, s in (("ArcFace", arc), ("SFace", sf)):
        mean, std, t = kfold_accuracy(s, labels, folds)
        out(f"| {name} | {mean*100:.2f}% ± {std*100:.2f} | {auc(s, labels)*100:.2f}% | {t:.3f} |")
    out("\nReference on CACD-VS: human average 94.6%, human voting 99.0%.\n")

    out("## 2. Current app verdicts\n")
    for title, second in (("ArcFace only", False), ("ArcFace + SFace second opinion", True)):
        out(f"**{title}**\n")
        out("| Truth | 🟢 match | 🟡 review | 🔴 different |")
        out("|---|---|---|---|")
        for truth, counts in verdict_table(arc, sf, labels, second).items():
            cells = " | ".join(f"{c} ({p:.1f}%)" for c, p in counts.values())
            out(f"| {truth} | {cells} |")
        out("")
    out("The most dangerous error for a suspect-search tool is **different people → 🟢 match** "
        "(an innocent person flagged as the suspect).\n")

    out("## 3. Recommended ArcFace thresholds\n")
    t_possible, _ = best_threshold(arc, labels)
    t_strong = max(threshold_for_false_match_rate(arc, labels, 0.01), t_possible)
    out(f"- 🟢 **match** ≥ **{t_strong:.3f}** — at most 1% of different-person pairs reach this score")
    out(f"- 🟡 **review** ≥ **{t_possible:.3f}** — best overall accuracy point")
    out(f"- current values: match ≥ {THRESHOLDS['arcface'][0]}, review ≥ {THRESHOLDS['arcface'][1]}")

    report = "\n".join(lines)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)

    try:
        plot(arc, sf, labels, os.path.join(args.out, "scores.png"))
        print(f"\n[+] Chart saved to {os.path.join(args.out, 'scores.png')}")
    except ImportError:
        print("\n[i] Install matplotlib to also get a chart:  pip install matplotlib")


def plot(arc, sf, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, s, model) in zip(axes, (("ArcFace", arc, "arcface"), ("SFace", sf, "sface"))):
        bins = np.linspace(min(s.min(), -0.2), 1, 60)
        ax.hist(s[labels], bins=bins, alpha=0.7, label="same person", color="#2a9d8f")
        ax.hist(s[~labels], bins=bins, alpha=0.7, label="different people", color="#e76f51")
        for t, style in zip(THRESHOLDS[model], ("-", "--")):
            ax.axvline(t, color="black", linestyle=style, linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("similarity score")
        ax.legend(frameon=False)
    axes[0].set_ylabel("number of pairs")
    fig.suptitle("CACD-VS: score distributions (lines = current thresholds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
