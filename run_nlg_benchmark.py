import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import transformers
from datasets import concatenate_datasets
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

# 导入 EvoRank 和其它对比算法的注入函数
from train_integration import inject_evo_lora, train_evo_lora_step
from adalora_utils import (
    adalora_update_and_allocate,
    compute_adalora_orthogonal_loss,
    get_adalora_orth_reg_weight,
    normalize_adalora_schedule,
)
from sora_inject import inject_sora, SparseAdamW
from flatlora_inject import FlatLoRAHookManager
from toplora_inject import inject_toplora
from hf_cache_utils import load_fxmeng_pissa_split, resolve_pretrained_model_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 统一 HuggingFace 缓存到当前工程目录
os.makedirs("./models", exist_ok=True)
os.makedirs("./datasets", exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.abspath("./models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("./models"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("./datasets"))

IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """对字符串列表进行分词"""
    tokenized_list = [
        tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        for text in strings
    ]
    input_ids = [np.array(tokenized["input_ids"]) for tokenized in tokenized_list]
    input_ids_lens = [len(tokenized["input_ids"]) for tokenized in tokenized_list]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """预处理数据：将原始输入和目标拼接并分词，计算 labels 中需要 ignored 的长度 (Instruction 部分)"""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    # 将 Instruction 部分的 label 设为 IGNORE_INDEX
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    """Dataset map 函数：将数据集行映射为 tokenized 字典"""
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

class DataCollatorForSupervisedDataset:
    """用于动态进行 padding 的 Collator"""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance["labels"], dtype=torch.long) for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

class DictFeatureClassifier(nn.Module):
    """用于将 CausalLM 封装为单个特征字典输入并输出 logits 的结构，以适配 train_evo_lora_step"""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.inner(**features).logits

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5):
    """余弦学习率衰减"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def _merge_evorank_into_base(model: nn.Module) -> None:
    """将所有 EvoRankLoRAWrapper 中的 LoRA 旁路合并回 base_layer 权重，然后用 base_layer 替换 wrapper。"""
    from train_integration import EvoRankLoRAWrapper, _set_module_by_path
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, EvoRankLoRAWrapper):
            module.lora_layer.merge(module.base_layer.weight)
            replacements.append((name, module.base_layer))
    for name, base_layer in replacements:
        _set_module_by_path(model, name, base_layer)
    logger.info(f"Merged {len(replacements)} EvoRank layers into base model.")


def _merge_sora_into_base(model: nn.Module) -> None:
    """将所有 SoRALinear 中的 ΔW = scaling * B @ diag(gate) @ A 合并回 base_layer 权重。"""
    from sora_inject import SoRALinear
    from train_integration import _set_module_by_path
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SoRALinear):
            with torch.no_grad():
                gate_diag = module.gate.squeeze(0).to(module.lora_A.dtype)
                delta_W = (module.lora_B @ torch.diag(gate_diag) @ module.lora_A) * module.scaling
                module.base_layer.weight.data += delta_W.to(module.base_layer.weight.dtype)
            replacements.append((name, module.base_layer))
    for name, base_layer in replacements:
        _set_module_by_path(model, name, base_layer)
    logger.info(f"Merged {len(replacements)} SoRA layers into base model.")


def _merge_toplora_into_base(model: nn.Module) -> None:
    """将 TopLoRALinear 近似合并回 base_layer。

    TopLoRA 的 λ(x) = exp(RMSNorm(x @ W_λ)) 是 token-dependent 的，无法精确合并为
    静态矩阵。这里使用 mean-field 近似：将 λ 视为全 1 向量（RMSNorm 输出均值趋近 0，
    exp(0) = 1），等价于标准 LoRA 合并 ΔW = scaling * B @ A。
    这是训练后评测的合理近似——在 RMSNorm 归一化下，λ 的 token 间方差通常较小。
    """
    from toplora_inject import TopLoRALinear
    from train_integration import _set_module_by_path
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, TopLoRALinear):
            with torch.no_grad():
                delta_W = (module.lora_B @ module.lora_A) * module.scaling
                module.base_layer.weight.data += delta_W.to(module.base_layer.weight.dtype)
            replacements.append((name, module.base_layer))
    for name, base_layer in replacements:
        _set_module_by_path(model, name, base_layer)
    logger.info(f"Merged {len(replacements)} TopLoRA layers into base model (mean-field approx).")


def build_model_and_peft(args, method: str, total_train_steps: Optional[int] = None):
    """构建 CausalLM 模型并注入对应的 PEFT 模块。

    total_train_steps
        AdaLoRA（peft>=0.11）要求 ``AdaLoraConfig(total_step=...)`` 在构造时即 >0；
        由 ``run_sft_training`` 在得到 ``train_loader`` 后传入实际总优化步数。
    """
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    # 强制在单卡模式下使用 device_map="auto"；如果是 DDP 则先放置在分配的 gpu 上。
    # 为了避免与 deepspeed 冲突，DDP 环境使用空 cache 和 manual to(device)。
    # 自动探测本地 Hub 缓存（含 models/hub、并列 ./hub、./hf_home/hub），存在则 local_files_only 并解析快照路径
    model_load_id, use_local = resolve_pretrained_model_source(args.model_name_or_path, "./models")
    if use_local:
        logger.info(
            f"Model {args.model_name_or_path} detected in local cache (load id={model_load_id!r}). "
            "Enabling local_files_only."
        )

    # transformers>=4.45 的 tokenizer/model __init__ 内部可能调用 model_info() 发起 HTTP 请求
    _prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    _prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if use_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 在 DDP 环境下，让 Rank 0 先加载（触发下载），其它 Rank 等待，防止缓存竞争
    if dist.is_initialized():
        if dist.get_rank() != 0:
            dist.barrier()

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_load_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir="./models",
            local_files_only=use_local,
        )
    finally:
        if _prev_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _prev_hub_offline
        if _prev_tf_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf_offline

    if dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = args.target_modules.split(",")
    controller = None
    meta = {
        "trainable_params": 0,
        "extra_params": 0,
        "target_modules": target_modules,
        "effective_dropout": args.lora_dropout,
    }

    if method in ["lora", "lora_kaiming", "pissa", "adalora"]:
        # 支持使用 HuggingFace 官方实现的四种算法
        init_weights = True
        if method == "lora":
            init_weights = True
        elif method == "lora_kaiming":
            # PEFT 不接受字符串 "kaiming"；init_lora_weights=True 即官方 LoRA 默认：
            # lora_A 为 Kaiming uniform、lora_B 全零（与 microsoft/LoRA loralib 一致）。
            init_weights = True
        elif method == "pissa":
            init_weights = getattr(args, "pissa_init_method", "pissa_niter_16") 
        elif method == "adalora":
            init_weights = True # AdaLoRA 之后再由其自带 hook 拦截（不过这里直接用 PEFT 的 AdaLoRA 结构不符 EvoRLoRA fair 设计）
            # 注意：公平比较中我们使用 EvoRLoRA 内部的 adalora_inject，但由于是 CausalLM，可以直接手写或者使用 PEFT
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout if method != "pissa" else 0.0, # PiSSA 论文要求 dropout=0
            init_lora_weights=init_weights,
        )
        
        # AdaLoRA 特殊处理：使用 HuggingFace 的 AdaLoRA 并同步超参
        if method == "adalora":
            from peft import AdaLoraConfig
            _ts = total_train_steps
            if _ts is None:
                _ts = getattr(args, "adalora_total_steps", None)
            if _ts is None or int(_ts) < 1:
                raise ValueError(
                    "AdaLoRA 需要有效的 total_train_steps（由数据与 batch 决定）。"
                    "请确保在构造模型前已创建 train_loader 并传入 total_train_steps，"
                    "或设置 --adalora_total_steps 为正整数。"
                )
            _ts = int(_ts)
            _ti, _tf, _sched_warn = normalize_adalora_schedule(
                total_steps=_ts,
                adalora_tinit=int(args.adalora_tinit),
                adalora_tfinal=int(args.adalora_tfinal),
            )
            _dt = max(1, min(int(args.adalora_delta_t), _ts))
            _adalora_sched_warn = (not dist.is_initialized()) or (dist.get_rank() == 0)
            if _adalora_sched_warn and _sched_warn is not None:
                logger.warning(_sched_warn)
            if _adalora_sched_warn and int(args.adalora_delta_t) != _dt:
                logger.warning(
                    f"AdaLoRA deltaT 已按 total_step={_ts} 收紧: {_dt} (was {args.adalora_delta_t})"
                )
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=args.lora_rank * 2, # init_r (通常比 target_r 大)
                target_r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                init_r=args.lora_rank * 2,
                tinit=_ti,
                tfinal=_tf,
                deltaT=_dt,
                beta1=0.85,
                beta2=0.85,
                orth_reg_weight=0.1,
                total_step=_ts,
            )
            
        logger.info(f"Initialize {method} adapter from base model with r={args.lora_rank}")
        model = get_peft_model(model, peft_config)
        
    elif method == "evorank":
        # EvoRank 我们的实现
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={
                "r_max": args.lora_rank,
                "r_init": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            },
            controller_kwargs={
                "rho": args.evorank_rho,
                "p_g": 1.0,
                "p_p": 0.05,
                "H_g": 4,
                "H_p": 4,
                "cooldown_steps": 100,
                "r_min": 2,
                "r_max": args.lora_rank,
                "alpha_u": 1.0,
                "beta_u": 0.5,
                "allow_reallocation": True,
                "expand_init_mode": "gradient",
                "max_reallocate_candidates": 16,
            },
        )
    elif method == "sora":
        inject_sora(
            model=model,
            target_modules=target_modules,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif method == "toplora":
        inject_toplora(
            model=model,
            target_modules=target_modules,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif method == "flatlora":
        # Flat-LoRA 基础是标准 LoRA；PEFT 的 True 即 Kaiming A + 零 B。
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=True,
        )
        model = get_peft_model(model, peft_config)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 将 layer 转换为 bf16
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module.to(torch.float32)
        elif 'lora_A' in name or 'lora_B' in name or 'lora_E' in name:
            module.to(dtype)

    meta["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, controller, meta


def run_sft_training(args, method: str):
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    model_load_id, use_local = resolve_pretrained_model_source(args.model_name_or_path, "./models")

    # transformers>=4.45 的 tokenizer __init__ 内部 _patch_mistral_regex 会调用
    # model_info(model_id) 发起 HTTP 请求，仅靠 local_files_only=True 无法阻止。
    _prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    _prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if use_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_load_id,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            cache_dir="./models",
            local_files_only=use_local,
        )
    finally:
        if _prev_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _prev_hub_offline
        if _prev_tf_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf_offline
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ======= 加载与预处理数据（须在 AdaLoRA 建模型前完成，以便计算 total_step）=======
    all_training_dataset = []
    for task in args.sub_task:
        if ":" in task: # e.g. metamath:100000
            cur_task, num_split = task.split(":")
            cur_split = f"{args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, args.dataset_split

        # DDP 屏障：确保数据下载同步
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()

        # 不在此处设置 HF_DATASETS_OFFLINE（理由见上）；另见 hf_cache_utils.load_fxmeng_pissa_split：
        # Hub 间歇不可达时 datasets 会走「仅缓存」路径并查找 default-data_dir=<task>，而磁盘常见为
        # default-<hash>（config_name=default），会抛 Couldn't find cache；该 helper 在校验后回退加载。
        ds = load_fxmeng_pissa_split(args.data_path, cur_task, cur_split, cache_dir="./datasets")
        
        if dist.is_initialized() and dist.get_rank() == 0:
            dist.barrier()
        if is_main_process:
            logger.info(f"Loaded {args.data_path}/{cur_task}/{cur_split}: {ds.num_rows} rows")
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if args.seed is not None:
        raw_train_datasets = raw_train_datasets.shuffle(seed=args.seed)

    if dist.is_initialized():
        dist.barrier()

    _n_rows = len(raw_train_datasets)
    _env_map = os.environ.get("NLG_DATASET_MAP_NUM_PROC")
    if _env_map is not None:
        _map_num_proc = max(1, int(_env_map))
    elif sys.platform == "win32":
        # Windows：datasets 多进程会 spawn 子进程并各自 import torch/CUDA，
        # 易触发 WinError 1455（页面文件太小，无法加载 fbgemm/cufft 等 DLL）。
        _map_num_proc = 1
    else:
        _map_num_proc = min(32, max(1, min(_n_rows, os.cpu_count() or 8)))
    if is_main_process:
        logger.info(f"Dataset map num_proc={_map_num_proc} (rows={_n_rows}, platform={sys.platform})")

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=_map_num_proc if is_main_process else 1,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0] if len(args.dataset_field) > 0 else "instruction", 
            "response": args.dataset_field[1] if len(args.dataset_field) > 1 else "output"
        }
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    sampler = DistributedSampler(train_dataset, seed=args.seed) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=data_collator,
        pin_memory=True,
    )

    total_steps = max(
        1,
        len(train_loader) * args.epochs // max(1, args.gradient_accumulation_steps),
    )

    model, controller, meta = build_model_and_peft(args, method, total_train_steps=total_steps)
    model.to(device)

    if method == "evorank":
        model = DictFeatureClassifier(model)

    # 抽取 Mini-Validation Set 用于 EvoRank 的适应度评估
    mini_val_batches = []
    if method == "evorank":
        for i, batch in enumerate(train_loader):
            if i >= args.mini_val_k:
                break
            batch = {k: v.cpu() for k, v in batch.items()}
            mini_val_batches.append(batch)

    # ======= 定义优化器 =======
    warmup_steps = int(total_steps * args.warmup_ratio)

    opt_class = SparseAdamW if method == "sora" else torch.optim.AdamW
    
    # 遵循 PiSSA 的配置
    optimizer = opt_class(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    sparse_optimizer = None
    if method == "sora":
        sora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".gate")]
        normal_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate")]
        optimizer = torch.optim.AdamW(normal_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        sparse_optimizer = SparseAdamW(sora_params, lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sparse_optimizer is not None:
        sparse_lr_scheduler = get_cosine_schedule_with_warmup(sparse_optimizer, warmup_steps, total_steps)

    # ======= DDP 包装 =======
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    # FlatLoRA Injection Hook
    flatlora_manager = None
    if method == "flatlora":
        flatlora_manager = FlatLoRAHookManager(model, args.flatlora_rho, total_steps)

    # ======= 训练循环 =======
    global_step = 0
    start_time = time.time()
    
    if is_main_process:
        logger.info(f"*** Starting training for {method} ***")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}, Batch/dev: {args.per_device_train_batch_size}")
        
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        model.train()
        accum_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if flatlora_manager is not None:
                flatlora_manager.prepare_step(global_step)
                
            if method == "evorank":
                # EvoRank 的自定义步（仅限前向和反向）
                def compute_loss(logits, targets):
                    # logits: [B, S, V], targets: [B, S]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=IGNORE_INDEX)

                # 将 inputs 重封装为 train_evo_lora_step 所需的元组
                inputs_for_evo = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
                
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
                    out = train_evo_lora_step(
                        model=model,
                        controller=controller,
                        optimizer=optimizer,
                        train_batch=(inputs_for_evo, batch["labels"]),
                        val_batch=[({"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}, b["labels"]) for b in mini_val_batches],
                        loss_fn=compute_loss,
                        step=global_step,
                        warmup_steps=warmup_steps,
                        T_es=args.T_es,
                        lambda_c=0.0,
                        include_noop_candidate=True,
                    )
                loss = torch.tensor(out["train_loss"], device=device)
            else:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
                    if flatlora_manager is not None:
                        flatlora_manager.perturb_before_forward()
                    outputs = model(**batch)
                    loss = outputs.loss
                
                if method == "adalora" and hasattr(model, "peft_config"):
                    # 使用 Huggingface PEFT Adalora 内部机制，它没有外置正交损失调用。
                    pass
                elif method == "sora":
                    # L1 Penalty
                    l1_penalty = sum(p.abs().sum() for n, p in model.named_parameters() if n.endswith(".gate"))
                    gate_numel = max(1, sum(p.numel() for n, p in model.named_parameters() if n.endswith(".gate")))
                    loss = loss + 1e-3 * l1_penalty / gate_numel
                    
                try:
                    (loss / args.gradient_accumulation_steps).backward()
                finally:
                    if flatlora_manager is not None:
                        flatlora_manager.restore_after_backward()

            accum_loss += loss.detach().item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                
                if method != "evorank":
                    optimizer.step()
                if method == "sora" and sparse_optimizer is not None:
                    sparse_optimizer.step()

                # AdaLoRA 需在 zero_grad 之前调用（依赖本步仍存在的 p.grad 统计 IPT）
                if method == "adalora":
                    model_to_update = model.module if hasattr(model, "module") else model
                    if hasattr(model_to_update, "peft_config") and hasattr(model_to_update, "update_and_allocate"):
                        model_to_update.update_and_allocate(global_step)

                lr_scheduler.step()
                if method == "sora" and sparse_optimizer is not None:
                    sparse_lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                if method == "sora" and sparse_optimizer is not None:
                    sparse_optimizer.zero_grad(set_to_none=True)

                global_step += 1
                
                if is_main_process and global_step % 10 == 0:
                    logger.info(f"Epoch {epoch} | Step {global_step}/{total_steps} | Loss: {accum_loss / args.gradient_accumulation_steps:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

    # ======= 保存合并模型 =======
    if is_main_process:
        logger.info(f"Training completed. Saving to {args.output_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        if method == "evorank":
            model_to_save = model_to_save.inner

        os.makedirs(args.output_dir, exist_ok=True)

        _is_peft = method in ["lora", "lora_kaiming", "pissa", "flatlora", "adalora"]

        if getattr(args, "save_adapter_only", False) and _is_peft:
            # ---- 仅保存 adapter 权重（~10MB），跳过 merge + 全量写盘（~13GB/7B） ----
            logger.info("Saving adapter only (skip merge) ...")
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            # 记录 base model 实际路径，供 eval 阶段自动重建完整模型
            with open(os.path.join(args.output_dir, "base_model_path.json"), "w") as f:
                json.dump({"base_model_path": os.path.abspath(model_load_id)}, f)
        else:
            # ---- 合并 adapter 到 base → 保存完整模型 ----
            if _is_peft:
                logger.info("Merging adapter into base model (PEFT merge_and_unload)...")
                model_to_save = model_to_save.merge_and_unload()
            elif method == "evorank":
                logger.info("Merging EvoRank LoRA into base model...")
                _merge_evorank_into_base(model_to_save)
            elif method == "sora":
                logger.info("Merging SoRA into base model...")
                _merge_sora_into_base(model_to_save)
            elif method == "toplora":
                logger.info("Merging TopLoRA into base model...")
                _merge_toplora_into_base(model_to_save)

            # 将模型移到 CPU 并释放 GPU 显存，防止 save_pretrained 期间 OOM
            model_to_save.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        
        # 记录元数据
        meta["total_time_sec"] = time.time() - start_time
        with open(os.path.join(args.output_dir, "benchmark_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--method", type=str, required=True, choices=["lora", "lora_kaiming", "pissa", "evorank", "adalora", "sora", "toplora", "flatlora"])
    parser.add_argument("--output_dir", type=str, default="artifacts/nlg_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16) # (1*16)*8 = 128 effective
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # LoRA / PEFT 结构参数
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # 数据集
    parser.add_argument("--data_path", type=str, default="fxmeng/pissa-dataset")
    parser.add_argument("--sub_task", nargs="+", default=["metamath:100000"])
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_field", nargs="+", default=["instruction", "output"])
    parser.add_argument("--model_max_length", type=int, default=512)
    
    # 算法特有参数
    parser.add_argument("--T_es", type=int, default=50) # EvoRank
    parser.add_argument("--mini_val_k", type=int, default=2) # EvoRank
    parser.add_argument("--evorank_rho", type=float, default=0.0)
    parser.add_argument("--adalora_tinit", type=int, default=100)
    parser.add_argument("--adalora_tfinal", type=int, default=500)
    parser.add_argument("--adalora_delta_t", type=int, default=10)
    parser.add_argument("--flatlora_rho", type=float, default=0.05)
    parser.add_argument("--pissa_init_method", type=str, default="pissa_niter_16", choices=["pissa", "pissa_niter_16"])
    parser.add_argument("--save_adapter_only", action="store_true", default=False,
                        help="PEFT 方法仅保存 adapter 权重（跳过 merge + 全量写盘），大幅加速冒烟测试")

    args = parser.parse_args()

    # torchrun / torch.distributed.run 在 PyTorch 2.x 起通过环境变量 LOCAL_RANK 分配进程与 GPU，
    # 不再向子进程传入 --local_rank。若仍用默认值 -1，则 dist 不会初始化、device 退化为 cuda:0，
    # 多进程会全部挤在同一张卡上（与 nvidia-smi 上两个 python 均占 GPU0 的现象一致）。
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    run_sft_training(args, args.method)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
