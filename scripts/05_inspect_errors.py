"""Print every test mistake of the round-2 model, highest confidence first."""
import json

rows = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
P = json.load(open("results/round2/predictions.json"))["finetuned"]
bad = [(c, r, p) for r, p, c in zip(rows, P["preds"], P["confs"]) if p != r["label"]]
for c, r, p in sorted(bad, key=lambda x: -x[0]):
    print(f"conf {c:.2f} [{r['category']}] gold={r['label']} pred={p} | {r['text']}")
print(len(bad), "errors")
