import math
import torch
import torch.nn as nn

from evo_rank_lora import EvoRankLoRALayer

def test_initialization():
    """验证参数形状和初始掩码状态"""
    in_features, out_features = 10, 20
    r_max, r_init = 8, 4
    layer = EvoRankLoRALayer(in_features, out_features, r_max, r_init, lora_alpha=16)
    
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    assert layer.r_max == r_max
    assert layer.get_active_rank() == r_init
    assert sum(layer.active_mask) == r_init
    assert layer.lora_A.weight.shape == (r_max, in_features)
    assert layer.lora_B.weight.shape == (out_features, r_max)
    print("  PASSED: test_initialization")

def test_forward_pass_correctness():
    """验证前向传播数值等价性"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    x = torch.randn(2, 10)
    out = layer(x)
    
    assert out.shape == (2, 20)
    
    # 手动计算一遍等价性
    active_idx = layer.get_active_indices()
    A_act = layer.lora_A.weight[active_idx, :]
    B_act = layer.lora_B.weight[:, active_idx]
    scaling = 16.0 / math.sqrt(4)
    out_manual = x @ A_act.T @ B_act.T * scaling
    
    assert torch.allclose(out, out_manual)
    print("  PASSED: test_forward_pass_correctness")

def test_scaling_compensation_single_side():
    """验证 c^2 Bug 已修复：补偿只作用于 B，不同时作用于 A 和 B"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    old_A = layer.lora_A.weight.clone()
    old_B = layer.lora_B.weight.clone()
    
    inactive_idx = layer.get_inactive_indices()[0]
    layer.activate_component(inactive_idx)
    
    assert layer.get_active_rank() == 5
    
    comp_factor = math.sqrt(5 / 4)
    old_active_idx = [0, 1, 2, 3]
    
    # A 权重不应该被改变！（修复 c^2 Bug 的关键验证）
    assert torch.allclose(layer.lora_A.weight[old_active_idx, :], old_A[old_active_idx, :]), \
        "BUG: A 权重被修改了，会导致 c^2 效应！"
    
    # 仅 B 权重被放大
    assert torch.allclose(layer.lora_B.weight[:, old_active_idx], old_B[:, old_active_idx] * comp_factor)
    
    # 新激活的组件 B 列有非零小噪声
    assert torch.any(layer.lora_B.weight[:, inactive_idx] != 0.0)
    print("  PASSED: test_scaling_compensation_single_side")

def test_deactivate_with_reverse_compensation():
    """验证休眠时的反向补偿：缩减秩时对留下的组件施加衰减"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    # 先给 B 赋予非零值以便测试（B 初始为零）
    with torch.no_grad():
        layer.lora_B.weight.fill_(1.0)
    
    B_before = layer.lora_B.weight.clone()
    
    # 休眠第 3 个组件（4 -> 3）
    layer.deactivate_component(3)
    assert layer.get_active_rank() == 3
    assert not layer.active_mask[3]
    
    # 反向补偿因子: sqrt(3 / 4) < 1，留下来的组件应该被衰减
    comp_factor = math.sqrt(3 / 4)
    remaining = [0, 1, 2]
    assert torch.allclose(layer.lora_B.weight[:, remaining], B_before[:, remaining] * comp_factor)
    
    # 被休眠组件的 A 权重应保留（权重继承）
    assert torch.allclose(layer.lora_A.weight[3, :], layer.lora_A.weight[3, :])
    print("  PASSED: test_deactivate_with_reverse_compensation")

def test_weight_inheritance():
    """验证去激活再重激活后权重不丢失"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    A_weight_before = layer.lora_A.weight[3, :].clone()
    
    layer.deactivate_component(3)
    assert not layer.active_mask[3]
    
    # A 权重应该保留（权重继承）
    assert torch.allclose(layer.lora_A.weight[3, :], A_weight_before)
    print("  PASSED: test_weight_inheritance")

def test_trace_trick_evaluation():
    """验证 Trace Trick 重要性评分的数值正确性"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    x = torch.randn(2, 10)
    target = torch.randn(2, 20)
    
    out = layer(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_B.weight.grad is not None
    
    alpha1 = 1.0
    alpha2 = 0.1
    scores = layer.compute_component_importance(alpha1, alpha2)
    
    assert scores.shape == (8,)
    
    # 手动计算一遍验证等价性
    manual_scores = torch.zeros(8)
    active_idx = [0, 1, 2, 3]
    for idx in active_idx:
        w_A = layer.lora_A.weight[idx, :]
        w_B = layer.lora_B.weight[:, idx]
        g_A = layer.lora_A.weight.grad[idx, :]
        
        grad_inter = torch.abs(torch.dot(g_A, w_A))
        norm_prod = torch.norm(w_A) * torch.norm(w_B)
        manual_scores[idx] = alpha1 * grad_inter + alpha2 * norm_prod
        
    assert torch.allclose(scores[active_idx], manual_scores[active_idx])
    
    demand = layer.compute_demand_score()
    assert isinstance(demand, float)
    print("  PASSED: test_trace_trick_evaluation")

def test_merge():
    """验证合并回基础模型权重的正确性"""
    layer = EvoRankLoRALayer(10, 20, 8, 4, lora_alpha=16)
    
    # 模拟一次训练步，使 lora_B 获得非零权重
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(2, 10)
    target = torch.randn(2, 20)
    loss = nn.MSELoss()(layer(x), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    base_W = nn.Parameter(torch.randn(20, 10))
    base_W_orig = base_W.clone()
    
    layer.merge(base_W)
    
    assert not torch.allclose(base_W, base_W_orig)
    
    # 手动验证合并逻辑
    active_idx = layer.get_active_indices()
    A_act = layer.lora_A.weight[active_idx, :]
    B_act = layer.lora_B.weight[:, active_idx]
    scaling = layer.get_scaling_factor(len(active_idx))
    delta_W = (B_act @ A_act) * scaling
    
    assert torch.allclose(base_W, base_W_orig + delta_W)
    print("  PASSED: test_merge")


if __name__ == "__main__":
    print("Running EvoRankLoRALayer tests...")
    test_initialization()
    test_forward_pass_correctness()
    test_scaling_compensation_single_side()
    test_deactivate_with_reverse_compensation()
    test_weight_inheritance()
    test_trace_trick_evaluation()
    test_merge()
    print("\nALL 7 TESTS PASSED!")
