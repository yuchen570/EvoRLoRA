import math
import torch
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from evo_rank_lora import EvoRankLoRALayer

# ==========================================
# 变异动作抽象 (Command Pattern)
# ==========================================

class ModuleMutation(ABC):
    """
    变异动作的基类 (Command 模式)。
    隔离了外部 ES 的验证集探索 (Trial) 与最终落实 (Commit)。
    - apply(): 临时修改模型参数以计算 Reward。
    - undo(): 完全回滚刚才的参数修改。
    """
    @abstractmethod
    def apply(self):
        pass
        
    @abstractmethod
    def undo(self):
        pass

    @abstractmethod
    def clear_cache(self):
        """释放 apply() 产生的临时缓存，避免对象被外部持有时造成显存泄漏。"""
        pass

class ExpandMutation(ModuleMutation):
    def __init__(self, layer_name: str, layer: EvoRankLoRALayer, index: int):
        self.layer_name = layer_name
        self.layer = layer
        self.index = index
        # 缓存扩张前的 A 和 B 的克隆 (为了完全精确地 undo compensate 的效果)
        self.cached_A = None
        self.cached_B = None
        
    def apply(self):
        self.cached_A = self.layer.lora_A.weight.data.clone()
        self.cached_B = self.layer.lora_B.weight.data.clone()
        self.layer.activate_component(self.index)
        
    def undo(self):
        if self.cached_A is not None and self.cached_B is not None:
            self.layer.active_mask[self.index] = False
            self.layer.lora_A.weight.data.copy_(self.cached_A)
            self.layer.lora_B.weight.data.copy_(self.cached_B)
            self.cached_A, self.cached_B = None, None

    def clear_cache(self):
        self.cached_A = None
        self.cached_B = None

class PruneMutation(ModuleMutation):
    def __init__(self, layer_name: str, layer: EvoRankLoRALayer, index: int):
        self.layer_name = layer_name
        self.layer = layer
        self.index = index
        self.cached_B = None # 去激活只补偿 B
        
    def apply(self):
        self.cached_B = self.layer.lora_B.weight.data.clone()
        self.layer.deactivate_component(self.index)
        
    def undo(self):
        if self.cached_B is not None:
            self.layer.active_mask[self.index] = True
            self.layer.lora_B.weight.data.copy_(self.cached_B)
            self.cached_B = None

    def clear_cache(self):
        self.cached_B = None

class ReallocateMutation(ModuleMutation):
    def __init__(self, prune_mut: PruneMutation, expand_mut: ExpandMutation):
        self.prune_mut = prune_mut
        self.expand_mut = expand_mut
        
    def apply(self):
        # 先缩减，再扩张（顺序无所谓，只要 undo 时严格逆序即可）
        self.prune_mut.apply()
        self.expand_mut.apply()
        
    def undo(self):
        # 严格的原子对称性逆序撤销
        self.expand_mut.undo()
        self.prune_mut.undo()

    def clear_cache(self):
        self.expand_mut.clear_cache()
        self.prune_mut.clear_cache()

# ==========================================
# 演化结构控制器大脑
# ==========================================

class RankEvolutionController:
    """
    控制整个模型维度演化的核心大脑。
    分为严格的执行阶段：
    1. update_statistics(): 仅收集 EMA 分数
    2. compute_thresholds(): 只算分位数阈值
    3. tick_evolution_state(): 推进时间和计数器
    4. generate_mutations(): 快照提取动作闭包 (Command对象)
    5. commit_mutation(): 全局永久生效，处理 Rolling Candidacy
    """
    
    def __init__(
        self,
        lora_layers: Dict[str, EvoRankLoRALayer],
        rho: float = 0.9,
        p_g: float = 0.8,
        p_p: float = 0.1,
        r_min: int = 2,
        r_max: int = 16,
        H_g: int = 2,
        H_p: int = 3,
        cooldown_steps: int = 2,
        # 与论文动态阈值小节capacity score一致：u_ℓ = α_g · g̃_ℓ + β_s · s̃̄_ℓ（层间 min-max 归一化）
        alpha_u: float = 1.0,
        beta_u: float = 1.0,
        # ===== ES 候选限流（避免 Reallocate 组合爆炸）=====
        max_expand_candidates: Optional[int] = None,
        max_prune_candidates: Optional[int] = None,
        max_reallocate_candidates: Optional[int] = None,
        reallocate_strategy: str = "topk_cross",
    ):
        if not lora_layers:
            raise ValueError("lora_layers 不能为空字典")

        self.layers = lora_layers
        self.rho = rho
        self.p_g = p_g
        self.p_p = p_p
        self.r_min = r_min
        self.r_max = r_max
        self.H_g = H_g
        self.H_p = H_p
        self.cooldown_steps = cooldown_steps
        self.alpha_u = float(alpha_u)
        self.beta_u = float(beta_u)

        device = next(iter(self.layers.values())).lora_A.weight.device
        
        # 内部状态变量
        self.ema_u: Dict[str, float] = {name: 0.0 for name in self.layers}
        self.ema_s: Dict[str, torch.Tensor] = {name: torch.zeros(r_max, device=device) for name in self.layers}
        
        self.count_g: Dict[str, int] = {name: 0 for name in self.layers}
        self.count_p: Dict[str, torch.Tensor] = {name: torch.zeros(r_max, dtype=torch.long, device=device) for name in self.layers}
        
        # 记录每个组件还剩几步解禁 (大于0表示正在冷却)
        self.cooldowns: Dict[str, torch.Tensor] = {name: torch.zeros(r_max, dtype=torch.long, device=device) for name in self.layers}
        
        # 是否完成了冷启动第一帧 (Step 0 全零陷阱的防御标志)
        self._is_initialized = False

        self.max_expand_candidates = max_expand_candidates
        self.max_prune_candidates = max_prune_candidates
        self.max_reallocate_candidates = max_reallocate_candidates
        self.reallocate_strategy = reallocate_strategy

    def cleanup_uncommitted_mutations(
        self,
        mutations: List["ModuleMutation"],
        committed: Optional["ModuleMutation"] = None,
    ) -> None:
        """
        ES 每轮结束后，主动清理那些未被 commit 的 Mutation 缓存引用。

        说明：当前实现中 mutation.undo() 一般会把 cached_* 置空，但外部 ES
        有时会不小心持有 mutation 对象（例如写日志/历史记录），因此增加
        显式 cleanup 可以进一步降低“落选动作缓存滞留导致 OOM”的工程风险。
        """
        if not mutations:
            return

        for m in mutations:
            if committed is not None and m is committed:
                continue
            try:
                m.clear_cache()
            except Exception:
                # cleanup 只做安全兜底，避免影响主流程
                pass

    def update_statistics(self):
        """收集 g_ℓ、组件重要性，组合成 u_ℓ（论文式 217）并对 s_{ℓ,i} 做 EMA。"""
        raw_g: Dict[str, float] = {}
        raw_bar_s: Dict[str, float] = {}
        curr_s_by_name: Dict[str, torch.Tensor] = {}

        for name, layer in self.layers.items():
            g_val = layer.compute_demand_score(use_cached=True)
            curr_s = layer.compute_component_importance(use_cached=True)
            curr_s_by_name[name] = curr_s
            active = layer.active_mask
            raw_g[name] = g_val
            if active.any():
                raw_bar_s[name] = float(curr_s[active].mean().item())
            else:
                raw_bar_s[name] = 0.0

        g_max = max(raw_g.values())
        s_max = max(raw_bar_s.values())
        eps = 1e-12

        for name, layer in self.layers.items():
            g_t = raw_g[name] / (g_max + eps)
            s_t = raw_bar_s[name] / (s_max + eps)
            curr_u = self.alpha_u * g_t + self.beta_u * s_t
            if not self._is_initialized:
                self.ema_u[name] = curr_u
            else:
                self.ema_u[name] = self.rho * self.ema_u[name] + (1 - self.rho) * curr_u

            curr_s = curr_s_by_name[name]
            active = layer.active_mask
            if not self._is_initialized:
                self.ema_s[name][active] = curr_s[active]
            else:
                self.ema_s[name][active] = self.rho * self.ema_s[name][active] + (1 - self.rho) * curr_s[active]

        self._is_initialized = True

    def compute_thresholds(self) -> Tuple[float, float]:
        """计算全网统一的扩张与修剪阈值。"""
        if not self._is_initialized:
            return float('inf'), float('-inf')
            
        # 扩张阈值 tau_grow
        all_u = torch.tensor(list(self.ema_u.values()), dtype=torch.float32)
        if all_u.numel() > 0:
            tau_grow = torch.quantile(all_u, self.p_g).item()
        else:
            tau_grow = float('inf')
            
        # 修剪阈值 tau_prune
        # 仅收集所有 活跃 且 不在冷却期 的组件
        valid_s_list = []
        for name, layer in self.layers.items():
            active_and_not_cool = layer.active_mask & (self.cooldowns[name] == 0)
            if active_and_not_cool.any():
                valid_s_list.append(self.ema_s[name][active_and_not_cool])
                
        if len(valid_s_list) > 0:
            all_valid_s = torch.cat(valid_s_list)
            tau_prune = torch.quantile(all_valid_s, self.p_p).item()
        else:
            tau_prune = float('-inf') # 防崩溃
            
        return tau_grow, tau_prune

    def tick_evolution_state(self, tau_grow: float, tau_prune: float):
        """
        [独立时间线步进] 推进计数器和冷却机制，发生永久全局状态变更的唯一入口（除了 commit）。
        """
        for name, layer in self.layers.items():
            act_rank = layer.get_active_rank()
            
            # 1. 更新扩张计数 count_g
            if self.ema_u[name] > tau_grow and act_rank < self.r_max:
                self.count_g[name] += 1
            else:
                self.count_g[name] = 0
                
            # 2. 更新修剪计数 count_p
            active_mask = layer.active_mask
            cooling_mask = self.cooldowns[name] > 0
            
            # 只有 活跃、非冷却、且容量够大、分数低的才涨 count_p
            eligible_for_prune = active_mask & (~cooling_mask) & (self.ema_s[name] < tau_prune)
            
            # tensor 运算，满足条件位置 +1，不满足条件立刻清零
            if act_rank > self.r_min:
                self.count_p[name] = torch.where(
                    eligible_for_prune,
                    self.count_p[name] + 1,
                    torch.zeros_like(self.count_p[name])
                )
            else:
                self.count_p[name].zero_()
                
            # 3. 冷却期倒计时 -1
            self.cooldowns[name] = torch.clamp(self.cooldowns[name] - 1, min=0)

    def generate_mutations(self) -> List[ModuleMutation]:
        """
        纯粹收集候选，不改任何计数或状态。
        返回所有达标候选，让外层 ES 在验证集上决策赢家。
        """
        mutations: List[ModuleMutation] = []

        # 1) 收集扩张候选（按 u_l 分数排序后再限流）
        expand_candidates: List[Tuple[float, ExpandMutation]] = []
        for name, layer in self.layers.items():
            if self.count_g[name] < self.H_g:
                continue
            inactive_indices = layer.get_inactive_indices()
            if not inactive_indices:
                continue
            # 保持一个稳定策略：优先激活最小未激活索引
            e_mut = ExpandMutation(name, layer, inactive_indices[0])
            expand_candidates.append((float(self.ema_u[name]), e_mut))

        expand_candidates.sort(key=lambda x: x[0], reverse=True)
        if self.max_expand_candidates is not None:
            expand_candidates = expand_candidates[: self.max_expand_candidates]

        expand_muts: List[ExpandMutation] = [m for _, m in expand_candidates]

        # 2) 收集修剪候选（按 s_{l,i} 从小到大排序后再限流）
        prune_candidates: List[Tuple[float, PruneMutation]] = []
        for name, layer in self.layers.items():
            valid_p_mask = self.count_p[name] >= self.H_p
            if not valid_p_mask.any():
                continue
            valid_indices = torch.where(valid_p_mask)[0].tolist()
            for idx in valid_indices:
                p_score = float(self.ema_s[name][idx].item())
                prune_candidates.append((p_score, PruneMutation(name, layer, idx)))

        prune_candidates.sort(key=lambda x: x[0], reverse=False)
        if self.max_prune_candidates is not None:
            prune_candidates = prune_candidates[: self.max_prune_candidates]

        prune_muts: List[PruneMutation] = [m for _, m in prune_candidates]

        # 3) 单操作候选
        mutations.extend(expand_muts)
        mutations.extend(prune_muts)

        # 4) Reallocate 候选：避免组合爆炸
        if not expand_muts or not prune_muts:
            return mutations

        reallocate_count = 0
        for e_mut in expand_muts:
            for p_mut in prune_muts:
                if e_mut.layer_name == p_mut.layer_name:
                    continue

                if self.reallocate_strategy not in {"topk_cross"}:
                    raise ValueError(f"未知 reallocate_strategy: {self.reallocate_strategy}")

                if self.max_reallocate_candidates is not None and reallocate_count >= self.max_reallocate_candidates:
                    return mutations

                mutations.append(
                    ReallocateMutation(
                        PruneMutation(p_mut.layer_name, p_mut.layer, p_mut.index),
                        ExpandMutation(e_mut.layer_name, e_mut.layer, e_mut.index),
                    )
                )
                reallocate_count += 1

        return mutations

    def commit_mutation(self, mutation: ModuleMutation):
        """
        [全局唯一永久生效口]
        ES 在几组动作中完成了 eval -> undo 的试探并选出了这个 mutation。
        正式落实参数变化，并仅针对赢家 清零计数器 与 施加冷却，
        落选者状态完整保留以参与下周期的 Rolling Candidacy。
        """
        # Reallocate 需要一次性 apply，然后只重置其涉及子动作的计数器。
        if isinstance(mutation, ReallocateMutation):
            mutation.apply()
            expand_mut = mutation.expand_mut
            prune_mut = mutation.prune_mut
            self.count_g[expand_mut.layer_name] = 0
            self.cooldowns[expand_mut.layer_name][expand_mut.index] = self.cooldown_steps
            self.count_p[prune_mut.layer_name][prune_mut.index] = 0
            mutation.clear_cache()
            return

        # 第一步：物理参数落实
        mutation.apply()

        # 第二步：针对性状态重置
        if isinstance(mutation, ExpandMutation):
            name = mutation.layer_name
            idx = mutation.index
            self.count_g[name] = 0
            self.cooldowns[name][idx] = self.cooldown_steps
            
        elif isinstance(mutation, PruneMutation):
             name = mutation.layer_name
             idx = mutation.index
             self.count_p[name][idx] = 0

        # 提交后不再需要 undo 缓存，主动释放以避免外部日志持有时的显存泄漏。
        mutation.clear_cache()
