#!/usr/bin/env python3
"""Evaluate a checkpoint on LCB-v6 (held-out tests), HumanEval+, APPS-300 or ErrCalBench.

Protocol (paper Sec. 4.1): n = 16 samples per problem, temperature 0.6, top-p 0.95, Qwen3 thinking
mode, max 8192 new tokens; pass@1 = avg@16; pass@4 = fraction of problems solved by at least one of
four samples (the first four of the 16). The student then predicts the outcome of each of its own
samples with the training diagnosis prompt (thinking disabled, temperature 0.1, top-p 0.95 as in
training, <= 64 tokens), giving BAcc and BSR of the pass/fail prediction.

    python eval.py --model outputs/score_qwen3-1.7b_s10/global_step_120/actor \
        --benchmark errcalbench --data data/ErrCalBench.jsonl \
        --diagnosis binary --output results/score_qwen3-1.7b_s10/errcalbench.json

--model accepts a Hugging Face id, a merged HF directory, or a verl FSDP actor directory
(merged into <dir>/merged_hf on first use).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from verl.trainer.ppo.score_utils import ERR_PASS, build_diagnosis_messages, parse_diagnosis, strip_thinking  # noqa: E402

SAMPLING = dict(temperature=0.6, top_p=0.95, max_tokens=8192)
DIAG_SAMPLING = dict(temperature=0.1, top_p=0.95, max_tokens=64)  # same as the training diagnosis pass


# ----------------------------------------------------------------------------- model
def resolve_model(path: str) -> str:
    if not os.path.isdir(path) or os.path.isfile(os.path.join(path, "config.json")):
        return path
    merged = os.path.join(path, "merged_hf")
    if not os.path.isfile(os.path.join(merged, "config.json")):
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger

        cfg = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=path,
            target_dir=merged,
            trust_remote_code=True,
            hf_model_config_path=os.path.join(path, "huggingface"),
        )
        merger = FSDPModelMerger(cfg)
        merger.merge_and_save()
        merger.cleanup()
    return merged


# ----------------------------------------------------------------------------- data
def load_problems(benchmark: str, data: str) -> list[dict]:
    """Each problem: messages (chat prompt), problem (text for the diagnosis prompt), ground_truth."""
    probs = []
    if benchmark == "lcb":
        import pandas as pd

        for _, row in pd.read_parquet(data).iterrows():
            msgs = [dict(m) for m in row["prompt"]]
            probs.append({"messages": msgs, "problem": msgs[-1]["content"], "ground_truth": row["reward_model"]["ground_truth"]})
    elif benchmark == "errcalbench":
        with open(data) as f:
            for line in f:
                item = json.loads(line)
                msgs = item["input"] if isinstance(item["input"], list) else [{"role": "user", "content": item["input"]}]
                label = item["label"] if isinstance(item["label"], dict) else json.loads(item["label"])
                label.setdefault("testtype", "stdin")
                label.setdefault("fn_name", "")
                probs.append({"messages": msgs, "problem": msgs[-1]["content"], "ground_truth": json.dumps(label)})
    elif benchmark == "apps":
        with open(data) as f:
            for line in f:
                item = json.loads(line)
                io = item["input_output"]
                io = json.loads(io) if isinstance(io, str) else io
                fn_name = io.get("fn_name") or ""
                if fn_name:
                    instr = "Implement the function as described. Output only the code, no explanations."
                    outputs = [json.dumps(o[0] if isinstance(o, list) and len(o) == 1 else o) for o in io["outputs"]]
                    gt = {"inputs": io["inputs"], "outputs": outputs, "testtype": "functional", "fn_name": fn_name}
                else:
                    instr = "Read input from stdin and print the output to stdout.\nOutput only the code, no explanations."
                    gt = {"inputs": io["inputs"], "outputs": io["outputs"], "testtype": "stdin", "fn_name": ""}
                text = f"Solve the following programming problem in Python.\n{instr}\n\n{item['question']}"
                probs.append({"messages": [{"role": "user", "content": text}], "problem": text, "ground_truth": json.dumps(gt)})
    elif benchmark == "humaneval_plus":
        from evalplus.data import get_human_eval_plus

        for task_id, p in sorted(get_human_eval_plus().items()):
            text = (
                "Complete the following Python function. Output only the code (including the function signature), "
                f"no explanations.\n\n```python\n{p['prompt']}\n```"
            )
            probs.append({"messages": [{"role": "user", "content": text}], "problem": text, "task_id": task_id,
                          "prompt": p["prompt"], "entry_point": p["entry_point"]})
    else:
        raise ValueError(benchmark)
    return probs


# ----------------------------------------------------------------------------- execution
def run_sandbox(probs, samples, workers):
    """(passed, error_type) per sample via the training reward function (10 s per test)."""
    from verl.utils.reward_score.feedback import compute_score

    def one(job):
        p_idx, text = job
        r = compute_score("code", text, probs[p_idx]["ground_truth"], {"split": "test"})
        return r["score"] == 1.0, int(r["error_type"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, samples))


def run_evalplus(probs, samples, out_path):
    """HumanEval+ verdicts from evalplus (base + 80x augmented tests)."""
    import re

    def code_of(text):
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", strip_thinking(text), re.DOTALL)
        return max(blocks, key=len).strip() if blocks else strip_thinking(text)

    work = tempfile.mkdtemp(prefix="heplus_", dir=os.path.dirname(os.path.abspath(out_path)))
    sample_file = os.path.join(work, "samples.jsonl")
    with open(sample_file, "w") as f:
        for p_idx, text in samples:
            p = probs[p_idx]
            code = code_of(text)
            if p["entry_point"] not in code:
                code = p["prompt"] + "\n" + code
            f.write(json.dumps({"task_id": p["task_id"], "solution": code}) + "\n")
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", sample_file], check=True)
    res_file = sample_file.replace(".jsonl", "_eval_results.json")
    evals = json.load(open(res_file))["eval"]
    seen, out = {}, []
    for p_idx, _ in samples:
        tid = probs[p_idx]["task_id"]
        k = seen.get(tid, 0)
        seen[tid] = k + 1
        ok = evals[tid][k]["plus_status"] == "pass"
        out.append((ok, ERR_PASS if ok else -1))  # evalplus gives no outcome type
    return out


# ----------------------------------------------------------------------------- metrics
def solved_within(passed, k):
    """1 if at least one of the first k samples of a problem passes (paper's pass@k)."""
    return float(any(passed[:k]))


def summarize(records, n):
    by_prob = {}
    for r in records:
        by_prob.setdefault(r["problem"], []).append(r["passed"])
    a = sum(r["passed"] and r["pred_pass"] for r in records)
    b = sum((not r["passed"]) and r["pred_pass"] for r in records)
    c = sum(r["passed"] and not r["pred_pass"] for r in records)
    d = sum((not r["passed"]) and not r["pred_pass"] for r in records)
    tpr, tnr = a / max(1, a + c), d / max(1, b + d)
    return {
        "pass@1": 100 * sum(sum(v) / len(v) for v in by_prob.values()) / len(by_prob),
        "pass@4": 100 * sum(solved_within(v, 4) for v in by_prob.values()) / len(by_prob),
        "bacc": 100 * 0.5 * (tpr + tnr),
        "bsr": 100 * b / max(1, b + d),
        "n_correct": sum(r["passed"] for r in records),
        "n_samples": len(records),
        "n_problems": len(by_prob),
        "n": n,
        "quadrants": {"A": a, "B": b, "C": c, "D": d},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True, choices=["lcb", "humaneval_plus", "apps", "errcalbench"])
    ap.add_argument("--data", default=None, help="parquet (lcb) / jsonl (apps, errcalbench); unused for humaneval_plus")
    ap.add_argument("--diagnosis", required=True, choices=["binary", "five_class"], help="binary for 1.7B, five_class for 4B/8B")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from vllm import LLM, SamplingParams

    model = resolve_model(args.model)
    probs = load_problems(args.benchmark, args.data)
    llm = LLM(model=model, tensor_parallel_size=args.tp, trust_remote_code=True, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096 + 8192)
    tok = llm.get_tokenizer()

    # 1) n samples per problem, thinking mode
    prompts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in probs]
    outs = llm.generate(prompts, SamplingParams(n=args.n, seed=args.seed, **SAMPLING))
    samples = [(i, c.text) for i, o in enumerate(outs) for c in o.outputs]

    # 2) execution
    if args.benchmark == "humaneval_plus":
        verdicts = run_evalplus(probs, samples, args.output)
    else:
        verdicts = run_sandbox(probs, samples, args.workers)

    # 3) self-prediction with the training diagnosis prompt
    binary = args.diagnosis == "binary"
    diag_prompts = [
        tok.apply_chat_template(build_diagnosis_messages(tok, probs[i]["problem"], text, binary), tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for i, text in samples
    ]
    diag = llm.generate(diag_prompts, SamplingParams(n=1, seed=args.seed, **DIAG_SAMPLING))

    records = []
    for (p_idx, _), (passed, err), d in zip(samples, verdicts, diag, strict=True):
        pred = parse_diagnosis(d.outputs[0].text, binary)
        records.append({"problem": p_idx, "passed": bool(passed), "error_type": err, "pred": int(pred), "pred_pass": pred == ERR_PASS})
    metrics = summarize(records, args.n)
    print(json.dumps({"model": args.model, "benchmark": args.benchmark, **metrics}, indent=2))
    with open(args.output, "w") as f:
        json.dump({"model": args.model, "benchmark": args.benchmark, "metrics": metrics, "records": records}, f)


if __name__ == "__main__":
    main()
