"""
Hugging Face 本地缓存路径解析（与仓库内 ``models/``、``datasets/`` 实际布局一致）。

实际布局
--------
- **模型**：``<model_cache_dir>/models--<org>--<name>/snapshots/<rev>/``
  （即 ``model_cache_dir`` 本身就是 hub_root，其下直接存放 ``models--*`` 目录）
- **数据集**：``<cache_dir>/<dataset_path>/<config_or_version>/0.0.0/<hash>/``

  - GLUE: ``datasets/glue/<task>/0.0.0/<hash>/``
  - CNN/DailyMail: ``datasets/cnn_dailymail/3.0.0/0.0.0/<hash>/``
  - fxmeng/pissa-dataset: ``datasets/fxmeng___pissa-dataset/default-<hash>/0.0.0/<hash>/``
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def hub_roots_for_model_cache(model_cache_dir: str) -> List[str]:
    """
    可能的 Hugging Face **Hub** 根目录（其下为 ``models--org--name/snapshots/``）。

    首先搜索 ``model_cache_dir`` 自身（实际仓库布局），然后搜索常见的
    ``<model_cache_dir>/hub``、并列目录 ``./hub``、``./hf_home/hub`` 以及用户默认缓存。
    """
    mcd = os.path.abspath(model_cache_dir)
    parent = os.path.dirname(mcd)
    candidates = [
        mcd,                                                          # models/ 自身即为 hub root
        os.path.join(mcd, "hub"),                                     # models/hub/
        os.path.join(parent, "hub"),                                  # 与 models/ 并列的 hub/
        os.path.join(parent, "hf_home", "hub"),                       # hf_home/hub/
        os.path.join(mcd, "hf_home", "hub"),                          # models/hf_home/hub/
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    out: List[str] = []
    seen = set()
    for c in candidates:
        n = os.path.normpath(c)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def snapshot_dir_with_config(hub_root: str, repo_slug: str) -> Optional[str]:
    """在 ``hub_root/models--<slug>/snapshots/<rev>/`` 下找到含 ``config.json`` 的快照目录。"""
    snap_root = os.path.join(hub_root, f"models--{repo_slug}", "snapshots")
    if not os.path.isdir(snap_root):
        return None
    try:
        for name in sorted(os.listdir(snap_root)):
            cand = os.path.join(snap_root, name)
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    except OSError:
        return None
    return None


def resolve_pretrained_model_source(model_name_or_path: str, model_cache_dir: str) -> Tuple[str, bool]:
    """
    解析 HF 模型 id 或本地路径，供 ``AutoTokenizer`` / ``AutoModel.from_pretrained`` 使用。

    若能在本地找到 ``config.json``，则返回**快照目录的绝对路径**并 ``local_files_only=True``，
    避免仍以远程 id 触发 Hub API（如无网 DNS 失败、或 DeBERTa tokenizer 的额外 ``model_info`` 调用）。

    返回值
    ------
    (load_id, local_files_only) : Tuple[str, bool]
        - 找到本地快照 → (快照目录路径, True)
        - 未找到 → (原始 model_name_or_path, False)
    """
    # 1. 如果已经是本地目录路径，直接使用
    if os.path.isdir(model_name_or_path):
        return model_name_or_path, True

    slug = model_name_or_path.replace("/", "--")
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        try_to_load_from_cache = None  # type: ignore[misc, assignment]

    # 2. 在所有候选 hub root 下搜索快照
    for hub in hub_roots_for_model_cache(model_cache_dir):
        if not os.path.isdir(hub):
            continue
        # 方式 A：通过 huggingface_hub API 查找
        if try_to_load_from_cache is not None:
            try:
                p = try_to_load_from_cache(model_name_or_path, "config.json", cache_dir=hub)
            except Exception:
                p = None
            if isinstance(p, str) and os.path.isfile(p):
                return os.path.dirname(p), True
        # 方式 B：直接遍历 snapshots 目录
        snap = snapshot_dir_with_config(hub, slug)
        if snap is not None:
            return snap, True

    # 3. 最后检查 model_cache_dir 下是否有 models--slug 目录（即使未找到含 config.json 的快照，
    #    只要目录存在就说明曾经下载过——仍然返回快照路径而非 hub ID，以避免触发网络调用）
    mcd = os.path.abspath(model_cache_dir)
    for base in [mcd] + hub_roots_for_model_cache(model_cache_dir):
        model_dir = os.path.join(base, f"models--{slug}")
        if not os.path.isdir(model_dir):
            continue
        # 尝试返回 snapshots 下任意一个目录（即使没有 config.json）
        snap_root = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snap_root):
            try:
                revs = sorted(os.listdir(snap_root))
                for rev in revs:
                    rev_path = os.path.join(snap_root, rev)
                    if os.path.isdir(rev_path):
                        return rev_path, True
            except OSError:
                pass
        # 目录存在但无可用 snapshots（如仅有 refs/ 的不完整下载），继续搜索其它候选

    return model_name_or_path, False


def check_local_model(model_name_or_path: str, cache_dir: str = "models") -> bool:
    """检查模型是否已存在于本地缓存中，以决定是否开启 offline 模式。"""
    _resolved, local_only = resolve_pretrained_model_source(model_name_or_path, cache_dir)
    return local_only


def _dataset_dir_has_artifacts(d: str) -> bool:
    """目录存在且非空：含数据文件或版本子目录即视为已缓存。"""
    if not os.path.isdir(d):
        return False
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and (
                    e.name.endswith((".arrow", ".parquet", ".json", ".lock"))
                    or e.name == "dataset_info.json"
                ):
                    return True
                if e.is_dir():
                    return True
    except OSError:
        return False
    return False


def _recursive_has_arrow(d: str, max_depth: int = 4) -> bool:
    """递归搜索目录树（最多 max_depth 层），检查是否存在 .arrow/.parquet 数据文件。"""
    if max_depth <= 0 or not os.path.isdir(d):
        return False
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and e.name.endswith((".arrow", ".parquet")):
                    return True
                if e.is_dir() and _recursive_has_arrow(e.path, max_depth - 1):
                    return True
    except OSError:
        return False
    return False


def check_local_dataset(path: str, name: Optional[str] = None, cache_dir: str = "datasets") -> bool:
    """
    检查 HuggingFace ``datasets`` 是否已在 ``cache_dir`` 下缓存。

    实际布局示例（以 ``cache_dir="datasets"`` 为例）::

        datasets/glue/sst2/0.0.0/<hash>/glue-train.arrow
        datasets/cnn_dailymail/3.0.0/0.0.0/<hash>/cnn_dailymail-train-*.arrow
        datasets/fxmeng___pissa-dataset/default-<hash>/0.0.0/<hash>/*.arrow

    本函数对以上布局做精确匹配探测，并兜底做递归搜索。
    """
    if os.path.isdir(path):
        return True
    root = os.path.abspath(cache_dir)
    candidates: List[str] = []

    if path == "glue" and name:
        # 实际: datasets/glue/<task>/0.0.0/<hash>/
        candidates.extend(
            [
                os.path.join(root, "glue", name),
                os.path.join(root, "glue", name, "0.0.0"),
                os.path.join(root, "glue", name, "default"),
                os.path.join(root, "datasets", "glue", name),
            ]
        )
    elif path == "cnn_dailymail" and name:
        # 实际: datasets/cnn_dailymail/3.0.0/0.0.0/<hash>/
        candidates.extend(
            [
                os.path.join(root, "cnn_dailymail", name),
                os.path.join(root, "cnn_dailymail", name, "0.0.0"),
                os.path.join(root, "cnn_dailymail"),
                os.path.join(root, "datasets", "cnn_dailymail", name),
            ]
        )
    elif path == "xsum" and name is None:
        candidates.extend(
            [
                os.path.join(root, "xsum"),
                os.path.join(root, "datasets", "xsum"),
            ]
        )
    else:
        # 通用 HuggingFace datasets 缓存：slug 使用 ___ 分隔符（三个下划线）
        slug_triple = path.replace("/", "___")
        slug_double = path.replace("/", "--")
        candidates.append(os.path.join(root, slug_triple))
        candidates.append(os.path.join(root, slug_double))
        candidates.append(os.path.join(root, path))
        if name:
            candidates.append(os.path.join(root, slug_triple, name))
            candidates.append(os.path.join(root, slug_double, name))

    # 先做浅层检查
    for c in candidates:
        if _dataset_dir_has_artifacts(c):
            return True

    # 兜底：对顶层候选目录做递归搜索（处理 0.0.0/<hash>/ 深层嵌套）
    checked: set = set()
    for c in candidates:
        norm = os.path.normpath(c)
        if norm in checked:
            continue
        checked.add(norm)
        if _recursive_has_arrow(c):
            return True

    return False
