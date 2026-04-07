import os
import csv
from collections import defaultdict
import glob

def format_metric(val):
    """格式化指标值为百分比字符串 (保留两位小数)。"""
    if val == "" or val is None or val == "N/A":
        return "-"
    try:
        # 尝试转换为浮点数
        v = float(val)
        return f"{v * 100:.2f}"
    except ValueError:
        return str(val)

def main():
    # 查找所有可能的 CSV 结果文件
    csv_files = glob.glob("artifacts/*.csv") + glob.glob("*.csv")
    target_csvs = ["results_fair_glue_deberta_ddp.csv", "results_fair_glue_deberta_rte_ddp.csv"]
    valid_csvs = [f for f in target_csvs if os.path.exists(f)]
    
    if not valid_csvs:
        print(f"未找到任何核心结果文件 {target_csvs}。请先运行测试脚本。")
        print("当前存在的 CSV 文件:", csv_files)
        return

    # 数据结构: data[method][task] = {指标键: 指标值}
    data = defaultdict(lambda: defaultdict(dict))
    params = {}

    for target_csv in valid_csvs:
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.get("method", "Unknown")
                task = row.get("task", "Unknown")
                
                # 记录参数量 (转换为 M 并在字典中去重保存)
                if method not in params and row.get("trainable_params"):
                    try:
                        pcount = int(row["trainable_params"])
                        params[method] = f"{pcount / 1_000_000:.2f}M"
                    except:
                        params[method] = row["trainable_params"]

                # 存储数据行
                # 如果存在 'mean' (均值) 行，则由于我们支持多种子实验，优先读取均值行进行展示
                if row.get("seed") == "mean":
                    data[method][task] = row 
                elif row.get("seed") == "std":
                    pass # 跳过标准差行
                else:
                    # 如果没有 mean 行，则读取普通数据行
                    if task not in data[method] or data[method][task].get("seed") != "mean":
                        data[method][task] = row

    # 指定显示的方法顺序（覆盖当前公平脚本中的全方法集合）
    methods_order = ["lora", "adalora", "evorank", "sora", "toplora", "flatlora", "pissa"]
    present_methods = []
    for m in methods_order:
        if m in data:
            present_methods.append(m)
    for m in data.keys():
        if m not in present_methods:
            present_methods.append(m)

    # 打印表头 (完全对齐用户要求的 UI 格式)
    print("| Method | # Params | MNLI (m/mm) | SST-2 (Acc) | CoLA (Mcc) | QQP (Acc/F1) | QNLI (Acc) | RTE (Acc) | MRPC (Acc) | STS-B (Corr) | All (Ave.) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    for method in present_methods:
        c_params = params.get(method, "-")
        
        # 内部数值提取函数
        def get_val(task, key, multiplier=100.0):
            try:
                v = data[method][task].get(key, "")
                if v == "" or v == "N/A" or v is None:
                    return None
                return float(v) * multiplier
            except:
                return None

        # --- MNLI (m/mm) ---
        mnli_m = get_val("mnli", "accuracy_m")
        mnli_mm = get_val("mnli", "accuracy_mm")
        if mnli_m is None and mnli_mm is None:
            # 如果没有专门的 m/mm 指标，退而求其次读取通用 accuracy
            mnli_acc = get_val("mnli", "accuracy")
            mnli_str = f"{mnli_acc:.2f}/-" if mnli_acc else "-"
            mnli_score = mnli_acc if mnli_acc else None
        else:
            mnli_str = f"{mnli_m if mnli_m else 0:.2f}/{mnli_mm if mnli_mm else 0:.2f}"
            mnli_score = ((mnli_m or 0) + (mnli_mm or 0)) / 2.0

        # --- SST-2 (Acc) ---
        sst2 = get_val("sst2", "accuracy")
        sst2_str = f"{sst2:.2f}" if sst2 is not None else "-"
        sst2_score = sst2

        # --- CoLA (Mcc) ---
        cola = get_val("cola", "matthews_corrcoef")
        cola_str = f"{cola:.2f}" if cola is not None else "-"
        cola_score = cola

        # --- QQP (Acc/F1) ---
        qqp_acc = get_val("qqp", "accuracy")
        qqp_f1 = get_val("qqp", "f1")
        if qqp_acc is None and qqp_f1 is None:
            qqp_str = "-"
            qqp_score = None
        else:
            qqp_str = f"{qqp_acc if qqp_acc else 0:.2f}/{qqp_f1 if qqp_f1 else 0:.2f}"
            qqp_score = ((qqp_acc or 0) + (qqp_f1 or 0)) / 2.0

        # --- QNLI (Acc) ---
        qnli = get_val("qnli", "accuracy")
        qnli_str = f"{qnli:.2f}" if qnli is not None else "-"
        qnli_score = qnli

        # --- RTE (Acc) ---
        rte = get_val("rte", "accuracy")
        rte_str = f"{rte:.2f}" if rte is not None else "-"
        rte_score = rte

        # --- MRPC (Acc) ---
        mrpc = get_val("mrpc", "accuracy")
        mrpc_str = f"{mrpc:.2f}" if mrpc is not None else "-"
        mrpc_score = mrpc

        # --- STS-B (Corr Ave) ---
        stsb = get_val("stsb", "pearson_spearman_mean")
        stsb_str = f"{stsb:.2f}" if stsb is not None else "-"
        stsb_score = stsb

        # --- All (Ave.) 计算综合平均分 ---
        scores = [mnli_score, sst2_score, cola_score, qqp_score, qnli_score, rte_score, mrpc_score, stsb_score]
        valid_scores = [s for s in scores if s is not None]
        if len(valid_scores) > 0:
            avg_score = sum(valid_scores) / len(valid_scores)
            avg_str = f"**{avg_score:.2f}**"
        else:
            avg_str = "-"

        # 打印生成的表格行
        row_str = f"| {method} | {c_params} | {mnli_str} | {sst2_str} | {cola_str} | {qqp_str} | {qnli_str} | {rte_str} | {mrpc_str} | {stsb_str} | {avg_str} |"
        print(row_str)

if __name__ == "__main__":
    main()
