"""Make PNGs from results.json and training_log.csv into results/round2/plots/."""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results/round2/plots", exist_ok=True)
R = json.load(open("results/round2/results.json"))
ORDER = ["base", "tfidf", "round1", "finetuned", "claude"]
arms = {k: R["arms"][k] for k in ORDER if k in R["arms"]}
NAMES = {"tfidf": "TF-IDF baseline (no neural net)", "round1": "Round 1 fine-tuned", "base": "Laya base (zero-shot)", "finetuned": "Round 2 fine-tuned", "claude": "Claude classifier"}
COL = {"tfidf": "#c9b458", "round1": "#7fb3d5", "base": "#9aa3ad", "finetuned": "#1f77b4", "claude": "#d9822b"}
foot = f"Synthetic test set, n={R['n']} held-out messages"

# confusion matrices
fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.4), squeeze=False)
for ax, (k, a) in zip(axes[0], arms.items()):
    cm = np.array(a["confusion_matrix"]["matrix"])
    labels = a["confusion_matrix"]["labels"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{NAMES[k]}\nacc {a['accuracy']:.1%}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.text(0.5, 0.015, foot, ha="center", fontsize=9)
fig.savefig("results/round2/plots/confusion.png", dpi=160)

# comparison bars
metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
           ("injection_catch_rate", "Injection catch"), ("unsafe_error_rate", "Unsafe error (lower=better)")]
fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.8 / len(arms)
for i, (k, a) in enumerate(arms.items()):
    vals = [a[m] or 0 for m, _ in metrics]
    b = ax.bar(np.arange(len(metrics)) + i * w, vals, w, label=NAMES[k], color=COL[k])
    ax.bar_label(b, fmt="%.2f", fontsize=8)
ax.set_xticks(np.arange(len(metrics)) + w * (len(arms) - 1) / 2, [m[1] for m in metrics])
ax.set_ylim(0, 1.3)
ax.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
ax.set_title("Return-intent classification\n" + foot, fontsize=10)
fig.tight_layout()
fig.savefig("results/round2/plots/comparison.png", dpi=160)

# accuracy vs coverage
fig, ax = plt.subplots(figsize=(6, 4))
for k, a in arms.items():
    if "accuracy_vs_coverage" in a:
        pts = [p for p in a["accuracy_vs_coverage"] if p["accuracy"] is not None]
        ax.plot([p["coverage"] for p in pts], [p["accuracy"] for p in pts], "o-", label=NAMES[k], color=COL[k])
ax.set_xlabel("coverage (share answered above confidence threshold)")
ax.set_ylabel("accuracy on answered")
ax.legend()
ax.set_title("Abstention trade-off\n" + foot, fontsize=10)
fig.tight_layout()
fig.savefig("results/round2/plots/accuracy_vs_coverage.png", dpi=160)

# training curve
if os.path.exists("results/round2/training_log.csv"):
    rows = list(csv.DictReader(open("results/round2/training_log.csv")))
    if rows:
        loss = np.array([float(r["loss"]) for r in rows])
        ep = np.array([int(r["epoch"]) for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(loss, alpha=0.3, color="#1f77b4", label="per-10-batch objective")
        k = 5
        if len(loss) > k:
            ax.plot(np.arange(k - 1, len(loss)), np.convolve(loss, np.ones(k) / k, "valid"), color="#1f77b4", label="moving avg")
        for e in sorted(set(ep))[1:]:
            ax.axvline(np.argmax(ep == e), color="gray", ls=":", lw=0.8)
        ax.set_xlabel("logged step (every 10 micro-batches); dotted = epoch boundary")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylabel("total objective (cross-entropy + policy-gradient term)", fontsize=8)
        ax.set_title("Laya fine-tuning on 640 synthetic return requests (round 2)\nThe objective can dip below 0: its policy-gradient term is zero-mean noise", fontsize=10)
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/round2/plots/training_loss.png", dpi=160)
print("plots written")


