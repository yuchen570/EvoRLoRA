"""MT-Bench 评测：调用 FastChat llm_judge 对训练后模型打分。

流程：
  1. 使用 FastChat gen_model_answer 对 80 道 MT-Bench 问题生成回答
  2. 使用 GPT-4 作为裁判对回答打分 (需设置 OPENAI_API_KEY)
  3. 解析分数并写入 eval_results.json（与 eval_nlg_pissa.py 格式兼容）

Usage
-----
    python scripts/eval_mtbench.py \\
        --model_dir artifacts/nlg/xxx_lora_seed42 \\
        --method lora --model_tag Llama-2-7b-hf --seed 42

前置依赖:
    pip install "fschat[model_worker,llm_judge]==0.2.36"

未设置 OPENAI_API_KEY 时：只跑 ``gen_model_answer`` 生成 MT-Bench 回答 jsonl，跳过 ``gen_judgment``；
事后有 key 时再 ``pip install openai``，并在 FastChat ``llm_judge`` 目录下手动运行 ``gen_judgment.py`` 补分。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _find_fastchat_judge_dir() -> Optional[str]:
    """定位 fastchat.llm_judge 的安装路径。"""
    try:
        import fastchat.llm_judge
        return os.path.dirname(fastchat.llm_judge.__file__)
    except ImportError:
        return None


def _gen_model_answer(
    model_dir: str,
    model_id: str,
    judge_dir: str,
    max_new_tokens: int,
    num_gpus: int,
    question_begin: Optional[int] = None,
    question_end: Optional[int] = None,
) -> str:
    """调用 FastChat gen_model_answer.py 生成回答。

    question_begin / question_end 与 FastChat 一致，对 question 列表做 ``[begin:end]`` 切片（冒烟时可传 0/3）。
    """
    model_dir_abs = os.path.abspath(model_dir)
    script = os.path.join(judge_dir, "gen_model_answer.py")
    cmd = [
        sys.executable, script,
        "--model-path", model_dir_abs,
        "--model-id", model_id,
        "--max-new-token", str(max_new_tokens),
        "--num-gpus-per-model", str(num_gpus),
    ]
    if question_begin is not None:
        cmd.extend(["--question-begin", str(question_begin)])
    if question_end is not None:
        cmd.extend(["--question-end", str(question_end)])
    logger.info(f"Running: {' '.join(cmd)} (cwd={judge_dir})")
    result = subprocess.run(
        cmd, capture_output=False, text=True, timeout=7200, cwd=judge_dir
    )
    if result.returncode != 0:
        logger.error(f"gen_model_answer failed with code {result.returncode}")

    answer_file = os.path.join(judge_dir, "data", "mt_bench", "model_answer", f"{model_id}.jsonl")
    if not os.path.exists(answer_file):
        alt = os.path.join("data", "mt_bench", "model_answer", f"{model_id}.jsonl")
        if os.path.exists(alt):
            answer_file = alt
    return answer_file


def _gen_judgment(model_id: str, judge_dir: str) -> Optional[str]:
    """调用 FastChat gen_judgment.py（需要 OPENAI_API_KEY 与 openai 包）。"""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY 未设置：跳过 GPT-4 裁判打分，仅保留模型回答。"
            "补评：export/set OPENAI_API_KEY 后执行 pip install 'openai>=1.0.0'，"
            "再在 fastchat/llm_judge 下运行 python gen_judgment.py --model-list %s",
            model_id,
        )
        return None
    try:
        import openai  # noqa: F401
    except ImportError:
        logger.warning(
            "未安装 openai，无法调用 gen_judgment。请: pip install 'openai>=1.0.0' 后重试。"
        )
        return None

    script = os.path.join(judge_dir, "gen_judgment.py")
    cmd = [
        sys.executable, script,
        "--model-list", model_id,
        "--parallel", "4",
    ]
    logger.info(f"Running: {' '.join(cmd)} (cwd={judge_dir})")
    result = subprocess.run(
        cmd, capture_output=False, text=True, timeout=7200, cwd=judge_dir
    )
    if result.returncode != 0:
        logger.warning(f"gen_judgment failed with code {result.returncode}")

    judgment_file = os.path.join(judge_dir, "data", "mt_bench", "model_judgment", "gpt-4_single.jsonl")
    if not os.path.exists(judgment_file):
        alt = os.path.join("data", "mt_bench", "model_judgment", "gpt-4_single.jsonl")
        if os.path.exists(alt):
            judgment_file = alt
    return judgment_file


def _parse_scores(judgment_file: str, model_id: str) -> Dict[str, Any]:
    """从 judgment jsonl 中解析目标模型的分数。"""
    scores: List[float] = []
    turn1_scores: List[float] = []
    turn2_scores: List[float] = []
    category_scores: Dict[str, List[float]] = {}

    if not os.path.exists(judgment_file):
        return {}

    with open(judgment_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if rec.get("model") != model_id:
                continue
            score = rec.get("score")
            if score is None or score == -1:
                continue
            scores.append(float(score))
            turn = rec.get("turn", 1)
            if turn == 1:
                turn1_scores.append(float(score))
            else:
                turn2_scores.append(float(score))
            cat = rec.get("category", "unknown")
            category_scores.setdefault(cat, []).append(float(score))

    if not scores:
        return {}

    avg = sum(scores) / len(scores)
    result: Dict[str, Any] = {
        "score": avg,
        "n": len(scores),
        "turn1": sum(turn1_scores) / len(turn1_scores) if turn1_scores else 0.0,
        "turn2": sum(turn2_scores) / len(turn2_scores) if turn2_scores else 0.0,
    }
    for cat, vals in category_scores.items():
        result[f"cat_{cat}"] = sum(vals) / len(vals)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="训练后合并模型的目录")
    ap.add_argument("--method", default="unknown")
    ap.add_argument("--model_tag", default="unknown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument(
        "--question-begin",
        dest="question_begin",
        type=int,
        default=None,
        help="仅评测 MT-Bench 题目切片 [begin:end)，与 FastChat gen_model_answer 一致；冒烟建议 0",
    )
    ap.add_argument(
        "--question-end",
        dest="question_end",
        type=int,
        default=None,
        help="题目切片结束下标（不含）；冒烟建议 3 表示只跑前 3 条",
    )
    ap.add_argument("--out_file", default=None, help="默认写 <model_dir>/eval_results.json")
    args = ap.parse_args()

    judge_dir = _find_fastchat_judge_dir()
    if judge_dir is None:
        logger.error(
            "FastChat 未安装。请运行：pip install 'fschat[model_worker,llm_judge]'\n"
            "或参考 https://github.com/lm-sys/FastChat"
        )
        sys.exit(1)

    model_id = f"{args.model_tag}_{args.method}_seed{args.seed}"

    # Step 1: 生成回答
    answer_file = _gen_model_answer(
        args.model_dir,
        model_id,
        judge_dir,
        args.max_new_tokens,
        args.num_gpus,
        question_begin=args.question_begin,
        question_end=args.question_end,
    )
    logger.info(f"Answers saved to {answer_file}")

    # Step 2: GPT-4 裁判打分
    judgment_file = _gen_judgment(model_id, judge_dir)

    # Step 3: 解析分数
    metrics: Dict[str, Any] = {}
    if judgment_file:
        mt_scores = _parse_scores(judgment_file, model_id)
        if mt_scores:
            metrics["mt_bench"] = mt_scores
            logger.info(f"MT-Bench score: {mt_scores.get('score', 'N/A')}")
        else:
            logger.warning("未能从 judgment 文件中解析到分数。")
    else:
        logger.info("跳过分数解析（无 judgment 文件）。")
        if not os.environ.get("OPENAI_API_KEY"):
            n_ans = 0
            if answer_file and os.path.isfile(answer_file):
                with open(answer_file, "r", encoding="utf-8") as af:
                    n_ans = sum(1 for _ in af)
            metrics["mt_bench"] = {
                "status": "answers_only",
                "answer_file": answer_file,
                "n": n_ans,
                "note": "未设置 OPENAI_API_KEY；无 MT-Bench 分数。汇总表中 MT-Bench 列为 '-'。",
            }

    # Step 4: 写 eval_results.json
    out_file = args.out_file or os.path.join(args.model_dir, "eval_results.json")

    existing: Dict[str, Any] = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    result = {
        "method": args.method,
        "model_tag": args.model_tag,
        "seed": args.seed,
        "sub_task": ["conversation"],
        "dataset_split": "mt_bench",
        "n_samples": int(metrics.get("mt_bench", {}).get("n") or 0),
        "metrics": {**existing.get("metrics", {}), **metrics},
    }

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote eval_results.json to {out_file}")


if __name__ == "__main__":
    main()
