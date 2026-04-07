from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)
for n, p in model.named_parameters():
    if 'query_proj' in n and 'weight' in n:
        print(f'{n}: dtype={p.dtype}, shape={p.shape}, norm={p.float().norm().item():.4f}, max={p.float().abs().max().item():.6f}')
        break
for n, p in model.named_parameters():
    if 'classifier' in n and 'weight' in n:
        print(f'{n}: dtype={p.dtype}, shape={p.shape}, norm={p.float().norm().item():.4f}, max={p.float().abs().max().item():.6f}')
        break
