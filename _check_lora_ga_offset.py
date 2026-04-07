"""检查 LoRA-GA stable_gamma 初始化后 offset 的幅度"""
import torch
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from lora_ga_init import run_lora_ga_init_pipeline

model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

# 准备一个小 dataloader
ds = load_dataset("glue", "rte", split="train[:64]")
def tokenize(ex):
    out = tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, max_length=128, padding="max_length")
    out["labels"] = ex["label"]
    return out
ds = ds.map(tokenize, batched=True, remove_columns=["sentence1", "sentence2", "idx", "label"])
ds.set_format("torch")
loader = DataLoader(ds, batch_size=8, shuffle=False)

target_modules = ["query_proj", "key_proj", "value_proj", "intermediate.dense", "output.dense"]

for gamma in [16, 64, 256]:
    init_by_key = run_lora_ga_init_pipeline(
        base_model=copy.deepcopy(model),
        train_loader=loader,
        target_modules=target_modules,
        lora_r=8,
        lora_ga_batches=8,
        task_type="nlu",
        device=torch.device("cpu"),
        is_main_process=True,
        ddp_enabled=False,
        stable_gamma=gamma,
    )
    # 检查 A/B 幅度和 offset 幅度
    scaling = 16.0 / 8.0  # lora_alpha / r, use_rslora=False
    print(f"\n=== stable_gamma={gamma}, scaling={scaling} ===")
    for key in list(init_by_key.keys())[:3]:  # 只看前3层
        A, B = init_by_key[key]
        offset = (B.float() @ A.float()) * scaling
        # 找到对应的原始权重
        for n, p in model.named_parameters():
            if key in n and n.endswith(".weight") and "lora" not in n:
                w_max = p.float().abs().max().item()
                w_norm = p.float().norm().item()
                break
        else:
            w_max, w_norm = 0, 0
        print(f"  {key}: A_norm={A.float().norm():.4f} B_norm={B.float().norm():.4f} "
              f"offset_max={offset.abs().max():.6f} offset_norm={offset.norm():.4f} "
              f"weight_max={w_max:.6f} weight_norm={w_norm:.4f} "
              f"ratio_max={offset.abs().max()/max(w_max,1e-12):.4f}")
