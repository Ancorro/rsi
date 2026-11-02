"""Probe: does the authors' released weight edit reproduce its math gain on our hardware?

Rebuilds Qwen3.5-0.8B with each released edit (vendor/apply_edit.py checks every tensor hash), then samples
MATH-500 and GSM8K with the paper's thinking-on protocol (think35: temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, 32,768 new tokens) and grades the last \\boxed{} with math-verify.

    python probe.py --models unedited,p50c --n 100 --samples 1 --out results/<name>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}


def base_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO_ID, revision=REV)


def build(name, work):
    """Model folder for `name`: the base snapshot, or a copy with the edited weights file."""
    src = base_dir()
    if name == "unedited":
        return src
    out = os.path.join(work, f"qwen35-{name}")
    st = "model.safetensors-00001-of-00001.safetensors"
    if not os.path.exists(os.path.join(out, st)):
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(src):
            if f != st:
                shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
        subprocess.run([sys.executable, os.path.join(HERE, "vendor/apply_edit.py"), os.path.join(src, st),
                        os.path.join(HERE, "edits", name), os.path.join(out, st)], check=True)
    return out


def problems(n):
    from datasets import load_dataset
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    g = load_dataset("openai/gsm8k", "main", split="test")
    # fixed evenly spaced subsets so every model sees the same problems
    pick = lambda ds, k: [ds[int(i * len(ds) / k)] for i in range(k)]
    return {"math500": [{"problem": r["problem"], "gold": r["answer"]} for r in pick(m, n)],
            "gsm8k": [{"problem": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                      for r in pick(g, n)]}


def last_boxed(text):
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def correct(pred, gold):
    from math_verify import parse, verify
    if pred is None:
        return False
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="unedited,p50c")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "probe"))
    ap.add_argument("--work", default=os.path.expanduser("~/.cache/rsi-wsp"))
    ap.add_argument("--mem", type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = problems(a.n)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    summary = {"hyperparameters": {**SAMPLING, "n_problems_per_set": a.n, "samples": a.samples, "thinking": True,
                                   "base": f"{REPO_ID}@{REV}", "subset": "evenly spaced over each test set"},
               "results": {}}
    for name in a.models.split(","):
        path = build(name, a.work)
        tok = AutoTokenizer.from_pretrained(path)
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=SAMPLING["max_tokens"] + 2048,
                  seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
        sp = SamplingParams(n=a.samples, seed=0, **SAMPLING)
        res = {}
        for sname, rows in sets.items():
            prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=True)
                       for r in rows]
            t0 = time.time()
            outs = llm.generate(prompts, sp)
            recs, ok, cut, ntok = [], 0, 0, 0
            for r, o in zip(rows, outs):
                for c in o.outputs:
                    pred = last_boxed(c.text.split("</think>")[-1]) if "</think>" in c.text else None
                    good = correct(pred, r["gold"])
                    ok += good
                    cut += c.finish_reason == "length"
                    ntok += len(c.token_ids)
                    recs.append({"gold": r["gold"], "pred": pred, "correct": good, "finish": c.finish_reason,
                                 "tokens": len(c.token_ids), "tail": c.text[-400:]})
            k = len(recs)
            res[sname] = {"acc": ok / k, "cut_share": cut / k, "mean_tokens": ntok / k, "answers": k,
                          "seconds": round(time.time() - t0)}
            json.dump(recs, open(os.path.join(a.out, f"{name}_{sname}.json"), "w"), indent=1)
            print(name, sname, res[sname], flush=True)
        summary["results"][name] = res
        path_s = os.path.join(a.out, "summary.json")
        if os.path.exists(path_s):   # one process per model (run.sh): merge into the shared summary
            old = json.load(open(path_s))
            old["results"].update(summary["results"])
            summary = old
        json.dump(summary, open(path_s, "w"), indent=1)


if __name__ == "__main__":
    main()
