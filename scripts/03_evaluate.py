"""Evaluate arms on data/test.jsonl and write results/round2/results.json.

Arms: base Laya zero-shot, fine-tuned Laya, Claude as classifier (optional).
Usage: python eval.py [--arms base,finetuned,claude] [--claude-model claude-haiku-4-5-20251001]
"""
import argparse
import json
import os
import time

import numpy as np

LABELS = ["refund", "exchange", "clarify", "escalate"]
UNSAFE_PRED = {"refund", "exchange"}


def load_test():
    return [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8") if l.strip()]


def run_laya(path, rows, questions):
    import laya

    agent = laya.load(path)
    agent.predict_batch([rows[0]["text"]], questions)  # warmup
    preds, confs, probs, lat = [], [], [], []
    for r in rows:
        t0 = time.perf_counter()
        ans = agent.predict_batch([r["text"]], questions)[0]["answers"]["intent"]
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(ans["choice"])
        p = ans["probabilities"]
        probs.append([float(p.get(k, 0.0)) for k in LABELS])
        confs.append(float(ans.get("answer_confidence", max(p.values()))))
    return preds, confs, probs, lat


def run_tfidf(rows):
    """Baseline with no neural network: TF-IDF (uni+bigrams) + softmax regression in plain NumPy."""
    import re
    from collections import Counter

    train = []
    for line in open("data/train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        p = json.loads(r["gold"])["intent"]["probabilities"]
        train.append((json.loads(r["state"]), LABELS.index(max(p, key=p.get))))

    def tokens(s):
        w = re.findall(r"[a-z0-9$']+", s.lower())
        return w + [a + "_" + b for a, b in zip(w, w[1:])]

    df = Counter(t for text, _ in train for t in set(tokens(text)))
    keys = sorted(df)
    vocab = {t: i for i, t in enumerate(keys)}
    idf = np.array([np.log((1 + len(train)) / (1 + df[t])) + 1 for t in keys])

    def vec(text):
        v = np.zeros(len(vocab))
        for t, c in Counter(tokens(text)).items():
            if t in vocab:
                v[vocab[t]] = (1 + np.log(c)) * idf[vocab[t]]
        n = np.linalg.norm(v)
        return v / n if n else v

    X = np.stack([vec(t) for t, _ in train])
    Y = np.eye(len(LABELS))[[y for _, y in train]]
    W = np.zeros((X.shape[1], len(LABELS)))
    for _ in range(300):
        z = X @ W
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W -= 5.0 * (X.T @ (p - Y) / len(X) + 1e-4 * W)
    preds, confs, probs, lat = [], [], [], []
    for r in rows:
        t0 = time.perf_counter()
        z = vec(r["text"]) @ W
        p = np.exp(z - z.max())
        p /= p.sum()
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(LABELS[int(p.argmax())])
        confs.append(float(p.max()))
        probs.append([float(x) for x in p])
    return preds, confs, probs, lat


def run_claude(model, rows, questions):
    import anthropic

    client = anthropic.Anthropic()
    crit = questions["intent"]["criteria"]
    system = (
        "Classify a customer's return-request message. Reply with exactly one label and nothing else.\n"
        + "\n".join(f"- {k}: {v}" for k, v in crit.items())
    )
    preds, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        msg = client.messages.create(
            model=model, max_tokens=8, temperature=0, system=system,
            messages=[{"role": "user", "content": r["text"]}],
        )
        lat.append((time.perf_counter() - t0) * 1000)
        out = msg.content[0].text.strip().lower()
        preds.append(next((l for l in LABELS if l in out), "clarify"))
    return preds, None, None, lat


def metrics(rows, preds, confs, probs, lat):
    gold = [r["label"] for r in rows]
    n = len(rows)
    acc = sum(g == p for g, p in zip(gold, preds)) / n
    per, f1s = {}, []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, preds))
        fp = sum(g != c and p == c for g, p in zip(gold, preds))
        fn = sum(g == c and p != c for g, p in zip(gold, preds))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[c] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    cm = [[sum(g == a and p == b for g, p in zip(gold, preds)) for b in LABELS] for a in LABELS]
    inj = [p for r, p in zip(rows, preds) if r["category"] == "injection"]
    esc = [p for g, p in zip(gold, preds) if g == "escalate"]
    unsafe_idx = [i for i, g in enumerate(gold) if g == "escalate"]
    out = {
        "n": n,
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
        "injection_n": len(inj),
        "injection_catch_rate": (sum(p == "escalate" for p in inj) / len(inj)) if inj else None,
        "unsafe_error_rate": sum(preds[i] in UNSAFE_PRED for i in unsafe_idx) / max(1, len(unsafe_idx)),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        out["by_category"][cat] = {"n": len(idx), "accuracy": sum(gold[i] == preds[i] for i in idx) / len(idx)}
    if confs is not None:
        c = np.array(confs)
        ok = np.array([g == p for g, p in zip(gold, preds)], dtype=float)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (c > lo) & (c <= hi)
            if m.any():
                ece += m.mean() * abs(ok[m].mean() - c[m].mean())
        out["ece"] = float(ece)
        curve = []
        for th in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            m = c >= th
            curve.append({"threshold": th, "coverage": float(m.mean()),
                          "accuracy": float(ok[m].mean()) if m.any() else None})
        out["accuracy_vs_coverage"] = curve
        # confidence gate: below the threshold the message goes to a person, so it cannot be auto-actioned
        gold_a = np.array(gold)
        esc = gold_a == "escalate"
        pred_a = np.array(preds)
        out["gate"] = {}
        for th in [0.0, 0.5, 0.7, 0.9]:
            auto = c >= th
            unsafe = esc & auto & np.isin(pred_a, list(UNSAFE_PRED))
            out["gate"][str(th)] = {"auto_rate": float(auto.mean()),
                                    "unsafe_error_rate": float(unsafe.sum() / max(1, esc.sum())),
                                    "accuracy_on_auto": float(ok[auto].mean()) if auto.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,finetuned,claude")
    ap.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--finetuned-dir", default="./models/round2")
    ap.add_argument("--base-dir", default="convaiinnovations/laya")
    args = ap.parse_args()

    rows = load_test()
    questions = json.load(open("data/questions.json", encoding="utf-8"))
    results = {"test_set": "data/test.jsonl (synthetic, held-out templates)", "n": len(rows), "arms": {}}
    if os.path.exists("results/round2/results.json"):
        results["arms"] = json.load(open("results/round2/results.json")).get("arms", {})
    raw = {}
    for arm in args.arms.split(","):
        print("running", arm, flush=True)
        if arm == "base":
            res = run_laya(args.base_dir, rows, questions)
        elif arm == "tfidf":
            res = run_tfidf(rows)
        elif arm == "round1":
            res = run_laya("./models/round1", rows, questions)
        elif arm == "finetuned":
            res = run_laya(args.finetuned_dir, rows, questions)
        elif arm == "claude":
            res = run_claude(args.claude_model, rows, questions)
        else:
            continue
        results["arms"][arm] = metrics(rows, *res)
        raw[arm] = {"preds": res[0], "confs": res[1], "probs": res[2]}
        a = results["arms"][arm]
        print(f"  acc={a['accuracy']:.3f} macroF1={a['macro_f1']:.3f} inj={a['injection_catch_rate']} "
              f"unsafe={a['unsafe_error_rate']:.3f} p50={a['latency_ms']['p50']:.0f}ms", flush=True)
        os.makedirs("results/round2", exist_ok=True)
        json.dump(results, open("results/round2/results.json", "w"), indent=2)
        json.dump({**(json.load(open("results/round2/predictions.json")) if os.path.exists("results/round2/predictions.json") else {}), **raw},
                  open("results/round2/predictions.json", "w"))


if __name__ == "__main__":
    main()
