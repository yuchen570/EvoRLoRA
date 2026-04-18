"""训练后评测：在 fxmeng/pissa-dataset 测试集上生成并计算指标（对齐 PiSSA 官方）。

对齐方式：
  - 生成（vLLM 优先，HF generate 兜底）：等价于 PiSSA/utils/gen_vllm.py
  - gsm8k / math 评测：复用 PiSSA/utils/test_acc.py 中的答案抽取与等价判断
  - humaneval / mbpp 后处理：复用 PiSSA/utils/code_process.py
  - humaneval / mbpp pass@1：若安装了 ``evalplus``，则调用 ``evalplus.evaluate``

Usage
-----
    python scripts/eval_nlg_pissa.py \
        --model_dir artifacts/nlg/xxx_lora_seed42 \
        --sub_task metamath python \
        --method lora --seed 42 \
        --out_file artifacts/nlg/xxx_lora_seed42/eval_results.json

输出：一个 ``eval_results.json``（键：method/seed/model/task → 指标字典），
      加上原始 ``*_response.jsonl`` 便于事后检查。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import concatenate_datasets, load_dataset
from fractions import Fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) 文本答案抽取 & 判等（完整移植自 PiSSA/utils/test_acc.py）
# ---------------------------------------------------------------------------

def _remove_right_units(s: str) -> str:
    if "\\text{ " in s:
        return s.split("\\text{ ")[0]
    return s


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        if not p or p[0] != "{":
            out += "\\sqrt{" + p[0] + "}" + p[1:] if p else "\\sqrt"
        else:
            out += "\\sqrt" + p
    return out


def _fix_fracs(s: str) -> str:
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        else:
            if len(sub) < 2:
                return s
            a, b = sub[0], sub[1]
            rest = sub[2:] if len(sub) > 2 else ""
            if b != "{":
                out += "{" + a + "}{" + b + "}" + rest
            else:
                out += "{" + a + "}" + b + rest
    return out


def _fix_a_slash_b(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        a_i, b_i = int(a), int(b)
        if s == f"{a_i}/{b_i}":
            return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
    except Exception:
        pass
    return s


def _strip_math_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("\\%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s


def _is_equiv(a: Optional[str], b: Optional[str]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return _strip_math_string(a) == _strip_math_string(b)
    except Exception:
        return a == b


def _process_math(completion: str, answer: str) -> bool:
    parts = completion.split("The answer is: ")
    if len(parts) <= 1:
        return False
    cand = parts[-1].split(".\n")[0].strip()
    if cand and cand[-1] == ".":
        cand = cand[:-1]
    return _is_equiv(cand.strip(), answer)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _extract_gsm8k_number(completion: str) -> Optional[float]:
    parts = completion.split("The answer is: ")
    if len(parts) <= 1:
        return None
    extract = parts[-1].strip()
    m = re.search(r"[\-+]?\d*[\.,/]?\d+", extract)
    if not m:
        return None
    token = m.group().replace(",", "")
    if "/" in token:
        num, den = token.split("/")
        if _is_number(num) and _is_number(den):
            if den == "0":
                return round(float(num))
            try:
                f = Fraction(token)
                return round(float(f.numerator / f.denominator))
            except Exception:
                return None
        return None
    try:
        v = float(token)
        return None if v == float("inf") else round(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2) 代码抽取（完整移植自 PiSSA/utils/code_process.py）
# ---------------------------------------------------------------------------

def _extract_code(completion: str) -> str:
    completion = completion.replace("\r", "")
    if "```python" in completion:
        idx = completion.index("```python")
        completion = completion[idx:].strip().replace("```python", "")
        try:
            end = completion.index("\n```")
            completion = completion[:end].strip()
        except ValueError:
            pass
    if '__name__ == "__main__"' in completion:
        idx = completion.index('if __name__ == "__main__":')
        completion = completion[:idx].strip()
    if "# Example usage" in completion:
        idx = completion.index("# Example usage")
        completion = completion[:idx].strip()
    if "assert" in completion:
        idx = completion.index("assert")
        completion = completion[:idx].strip()
    return completion


# ---------------------------------------------------------------------------
# 3) 生成后端：vLLM 优先，HF 兜底
# ---------------------------------------------------------------------------

def _generate_vllm(model_dir: str, prompts: Sequence[str], max_new_tokens: int, tp: int) -> List[str]:
    from vllm import LLM, SamplingParams  # type: ignore
    model_dir_abs = os.path.abspath(model_dir)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    llm = LLM(model=model_dir_abs, tensor_parallel_size=max(1, tp), dtype="bfloat16")
    outs = llm.generate(list(prompts), sp)
    return [o.outputs[0].text for o in outs]


def _generate_hf(
    model_dir: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    dtype: torch.dtype,
) -> List[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 本地训练产出目录：强制 local_files_only 防止 transformers 将相对路径当作 Hub repo_id 查询
    _local_only = os.path.isdir(model_dir)

    # 检测 adapter-only 保存（由 --save_adapter_only 产生）
    _base_info_file = os.path.join(model_dir, "base_model_path.json")
    _adapter_cfg_file = os.path.join(model_dir, "adapter_config.json")
    _is_adapter_only = os.path.isfile(_base_info_file) and os.path.isfile(_adapter_cfg_file)

    tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True,
        local_files_only=_local_only or _is_adapter_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if _is_adapter_only:
        with open(_base_info_file) as f:
            _base_path = json.load(f)["base_model_path"]
        logger.info(f"Adapter-only save detected; loading base from {_base_path}")
        model = AutoModelForCausalLM.from_pretrained(
            _base_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
            local_files_only=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
            local_files_only=_local_only,
        )
    model.eval()

    out_texts: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i : i + batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_tail = gen[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen_tail, skip_special_tokens=True)
        out_texts.extend(texts)
        logger.info(f"generated {i + len(batch)}/{len(prompts)}")
    return out_texts


# ---------------------------------------------------------------------------
# 4) 评测主流程
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def _load_test_dataset(data_path: str, sub_tasks: Sequence[str], split: str, cache_dir: str):
    # 使用 hf_cache_utils 中带离线回退逻辑的函数，防止 datasets API 解析离线缓存失败
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from hf_cache_utils import load_fxmeng_pissa_split

    parts = []
    for t in sub_tasks:
        try:
            ds = load_fxmeng_pissa_split(data_path, data_dir=t, split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"load_fxmeng_pissa_split({data_path}, data_dir={t}, split={split}) 失败：{e!r}；跳过")
            continue
        parts.append(ds)
    if not parts:
        raise RuntimeError("无任何可用的测试子任务")
    return concatenate_datasets(parts)


def _score_gsm8k_math(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        t = r.get("type", "")
        if t == "gsm8k":
            pred = _extract_gsm8k_number(r["output"])
            try:
                ok = pred is not None and float(pred) == float(r["answer"])
            except Exception:
                ok = False
            buckets["gsm8k"].append(bool(ok))
        elif t == "math":
            buckets["math"].append(_process_math(r["output"], r["answer"]))
    out: Dict[str, Dict[str, float]] = {}
    for k, v in buckets.items():
        out[k] = {"acc": (sum(v) / len(v)) if v else 0.0, "n": len(v)}
    return out


def _write_code_jsonls(records: List[Dict[str, Any]], out_dir: str) -> Dict[str, str]:
    paths = {"humaneval": os.path.join(out_dir, "humaneval.jsonl"),
             "mbpp": os.path.join(out_dir, "mbpp.jsonl")}
    streams = {k: open(v, "w", encoding="utf-8") for k, v in paths.items()}
    counts = {"humaneval": 0, "mbpp": 0}
    try:
        for r in records:
            t = r.get("type", "")
            if t not in ("humaneval", "mbpp"):
                continue
            task_id = str(r.get("answer", ""))
            completion = _extract_code(r.get("output", ""))
            streams[t].write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
            counts[t] += 1
    finally:
        for f in streams.values():
            f.close()
    return {k: paths[k] for k, c in counts.items() if c > 0}


def _run_evalplus(dataset: str, samples_path: str) -> Dict[str, float]:
    """尝试调用 ``evalplus.evaluate`` 并解析 pass@1；失败则返回空字典。"""
    cmd = [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", samples_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        logger.warning("evalplus 未安装；已保存 jsonl，请后续手动评测。")
        return {}
    except Exception as e:
        logger.warning(f"evalplus 调用失败：{e!r}")
        return {}
    stdout = (res.stdout or "") + "\n" + (res.stderr or "")
    out: Dict[str, float] = {}
    # evalplus 输出形如 "pass@1:   0.423" / "pass@1 (base): 0.423"
    for line in stdout.splitlines():
        m = re.search(r"pass@1[^:]*:\s*([0-9.]+)", line)
        if m:
            try:
                v = float(m.group(1))
                key = "pass@1_base" if "base" in line.lower() else ("pass@1_plus" if "plus" in line.lower() else "pass@1")
                out.setdefault(key, v)
            except Exception:
                pass
    if not out:
        logger.warning(f"未能从 evalplus 输出解析 pass@1；原始输出:\n{stdout[-2000:]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="训练完保存模型的目录（含 config.json）")
    ap.add_argument("--data_path", default="fxmeng/pissa-dataset")
    ap.add_argument("--sub_task", nargs="+", default=["metamath"],
                    help="测试子任务名（如 metamath / python）。metamath 含 gsm8k+math 条目，python 含 humaneval+mbpp。")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--dataset_cache_dir", default="./datasets")
    ap.add_argument("--max_samples", type=int, default=0, help="0 表示用全部；>0 时只评前 N 条便于 smoke test")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16, help="仅 HF 后端使用")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="仅 vLLM 使用")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)

    ap.add_argument("--method", default="unknown")
    ap.add_argument("--model_tag", default="unknown")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_file", default=None, help="默认写 <model_dir>/eval_results.json")
    ap.add_argument("--response_file", default=None, help="默认写 <model_dir>/eval_response.jsonl")

    args = ap.parse_args()

    out_file = args.out_file or os.path.join(args.model_dir, "eval_results.json")
    resp_file = args.response_file or os.path.join(args.model_dir, "eval_response.jsonl")

    # 1) 加载测试集
    ds = _load_test_dataset(args.data_path, args.sub_task, args.dataset_split, args.dataset_cache_dir)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info(f"Loaded {len(ds)} test samples from sub_tasks={args.sub_task}")

    instructions: List[str] = ds["instruction"]
    answers: List[Any] = ds["output"]
    types: List[str] = ds["type"] if "type" in ds.column_names else ["" for _ in ds]
    prompts = [PROMPT_TEMPLATE.format_map({"instruction": q}) for q in instructions]

    # 2) 生成
    if args.use_vllm:
        try:
            outputs = _generate_vllm(args.model_dir, prompts, args.max_new_tokens, args.tensor_parallel_size)
        except Exception as e:
            logger.warning(f"vLLM 失败（{e!r}），回退到 HF generate。")
            outputs = _generate_hf(args.model_dir, prompts, args.max_new_tokens, args.batch_size,
                                   torch.bfloat16 if args.bf16 else torch.float32)
    else:
        outputs = _generate_hf(args.model_dir, prompts, args.max_new_tokens, args.batch_size,
                               torch.bfloat16 if args.bf16 else torch.float32)

    assert len(outputs) == len(prompts), f"生成条数不匹配：{len(outputs)} vs {len(prompts)}"

    # 3) 保存响应 jsonl（PiSSA 兼容字段：type/query/output/answer）
    records: List[Dict[str, Any]] = []
    with open(resp_file, "w", encoding="utf-8") as f:
        for t, q, o, a in zip(types, instructions, outputs, answers):
            rec = {"type": t, "query": q, "output": o, "answer": a}
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Responses saved to {resp_file}")

    # 4) 指标
    metrics: Dict[str, Any] = {}

    # gsm8k / math
    math_metrics = _score_gsm8k_math(records)
    metrics.update(math_metrics)

    # humaneval / mbpp
    code_paths = _write_code_jsonls(records, args.model_dir)
    for name, path in code_paths.items():
        logger.info(f"Running evalplus on {name} ({path}) ...")
        r = _run_evalplus(name, path)
        if r:
            metrics[name] = {"n": sum(1 for _ in open(path, "r", encoding="utf-8")), **r}
        else:
            # 没装 evalplus：仍登记样本数，便于人工
            metrics[name] = {"n": sum(1 for _ in open(path, "r", encoding="utf-8"))}

    # 5) 整理并写 eval_results.json
    result = {
        "method": args.method,
        "model_tag": args.model_tag,
        "seed": args.seed,
        "sub_task": args.sub_task,
        "dataset_split": args.dataset_split,
        "n_samples": len(records),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote metrics to {out_file}\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
