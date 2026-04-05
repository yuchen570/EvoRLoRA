import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from evo_rank_lora import EvoRankLoRALayer
from rank_evolution_controller import (
    ExpandMutation,
    ModuleMutation,
    PruneMutation,
    RankEvolutionController,
    ReallocateMutation,
)


class _PaddingTrialMutation(ModuleMutation):
    """DDP 专用：无 apply/undo 语义，仅用于对齐各 rank 上 ES trial 的 all_reduce 次数。"""

    def apply(self) -> None:
        pass

    def undo(self) -> None:
        pass

    def clear_cache(self) -> None:
        pass


class EvoRankLoRAWrapper(nn.Module):
    """
    将冻结的原线性层与可演化 LoRA 旁路组合（论文 Eq. 81: W' = W + ΔW）。
    前向输出: base_layer(x) + lora_layer(x)
    """

    def __init__(self, base_layer: nn.Linear, lora_layer: EvoRankLoRALayer):
        super().__init__()
        self.base_layer = base_layer
        self.lora_layer = lora_layer

        # 原始预训练权重冻结，只训练 LoRA 旁路参数。
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_layer(x)


def _set_module_by_path(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """按 named_modules 路径替换子模块。"""
    if "." not in module_path:
        setattr(root, module_path, new_module)
        return

    parent_path, child_name = module_path.rsplit(".", 1)
    parent_module = root.get_submodule(parent_path)
    setattr(parent_module, child_name, new_module)


def inject_evo_lora(
    model: nn.Module,
    target_modules: List[str],
    layer_kwargs: Dict[str, Any],
    controller_kwargs: Optional[Dict[str, Any]] = None,
) -> RankEvolutionController:
    """
    将命中后缀的 Linear 替换为 EvoRankLoRAWrapper，并返回控制这些 LoRA 层的 controller。

    参数:
    - model: 待注入模型
    - target_modules: 目标后缀列表，例如 ["q_proj", "v_proj"]
    - layer_kwargs: 构造 EvoRankLoRALayer 的超参（例如 r_max, r_init, lora_alpha）
    - controller_kwargs: 构造 RankEvolutionController 的超参（例如 rho, p_g, p_p）
    """
    if not target_modules:
        raise ValueError("target_modules 不能为空")

    if "r_max" not in layer_kwargs or "r_init" not in layer_kwargs:
        raise ValueError("inject_evo_lora 需要 layer_kwargs 至少包含 r_max 与 r_init")
    controller_kwargs = dict(controller_kwargs or {})

    # 先冻结整个基座模型，避免未注入模块在训练中被意外更新。
    for p in model.parameters():
        p.requires_grad = False

    # 先收集路径，避免边遍历边修改模型结构。
    to_inject: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules):
            to_inject.append((name, module))

    if not to_inject:
        raise ValueError("未匹配到任何可注入的 nn.Linear，请检查 target_modules 后缀")

    lora_layers: Dict[str, EvoRankLoRALayer] = {}
    for name, base_linear in to_inject:
        lora_layer = EvoRankLoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            **layer_kwargs,
        )
        # LoRA 旁路放在与原线性层相同设备，避免 forward 时设备不一致。
        lora_layer = lora_layer.to(base_linear.weight.device)
        wrapped = EvoRankLoRAWrapper(base_layer=base_linear, lora_layer=lora_layer)
        _set_module_by_path(model, name, wrapped)
        lora_layers[name] = lora_layer

    # 避免重复传参冲突：controller.r_max 默认对齐 layer_kwargs["r_max"]，若显式给出需一致。
    layer_r_max = int(layer_kwargs["r_max"])
    controller_r_max = int(controller_kwargs.get("r_max", layer_r_max))
    if controller_r_max != layer_r_max:
        raise ValueError("controller_kwargs['r_max'] 必须与 layer_kwargs['r_max'] 一致")
    controller_kwargs["r_max"] = controller_r_max

    controller = RankEvolutionController(lora_layers=lora_layers, **controller_kwargs)
    return controller


def train_evo_lora_step(
    *,
    model: nn.Module,
    controller: RankEvolutionController,
    optimizer: torch.optim.Optimizer,
    train_batch: Tuple[torch.Tensor, torch.Tensor],
    val_batch: Optional[
        Union[
            Tuple[torch.Tensor, torch.Tensor],
            Iterable[Tuple[torch.Tensor, torch.Tensor]],
        ]
    ],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    step: int,
    warmup_steps: int,
    T_es: int = 200,
    lambda_c: float = 0.0,
    complexity_mode: str = "rank_sum",
    lambda_pop: Optional[int] = None,
    population_strategy: str = "all",
    random_seed: Optional[int] = None,
    max_grad_norm: Optional[float] = None,
    include_noop_candidate: bool = True,
) -> Dict[str, Any]:
    """
    论文 Alg. 1：双时间尺度训练步。

    Inner loop（Alg. 1 line 4–6）：每步对 active LoRA 参数做梯度下降。
    Outer loop（Alg. 1 line 7–12, Alg. 2）：每 T_es 步做结构演化：
      1. 缓存 g_ℓ 与 s^red（Eq. 138, 150）——在 clip_grad_norm 之前
      2. EMA + 阈值 + 计数器（Alg. 2 全流程）
      3. 生成候选 N(z_t)（Sec 3.4 三类变异）
      4. 逐候选 apply → D_val eval → undo（Alg. 1 line 9–10）
      5. Elitist selection（Eq. 167）：z_{t+1} = argmax R(z)
         - R(z') = -L_val(Θ;z') - λ_c C(z')  …Eq. 163
         - C(z): rank_sum（Eq. 127）或 size_aware（Eq. 131）
      6. no-op（z_t）始终参与竞选，保证 R 单调不减（Theorem 4.4）

    Warmup（Alg. 2 line 2–3, Sec 3.7 line 268）：前 10% steps 仅做参数更新。
    """
    def _iter_val_batches(
        batch_or_batches: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                Iterable[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if batch_or_batches is None:
            return []
        if isinstance(batch_or_batches, tuple) and len(batch_or_batches) == 2:
            return [batch_or_batches]
        return list(batch_or_batches)

    def _sync_controller_state_device(ctrl: RankEvolutionController) -> None:
        """
        controller 在注入阶段可能构建在 CPU，但模型后续会被搬到 CUDA。
        这里在每步前做一次轻量对齐，避免状态张量与 LoRA 参数跨设备导致 RuntimeError。
        """
        layer_device = next(iter(ctrl.layers.values())).lora_A.weight.device
        for name in ctrl.layers:
            if ctrl.ema_s[name].device != layer_device:
                ctrl.ema_s[name] = ctrl.ema_s[name].to(layer_device)
            if ctrl.count_p[name].device != layer_device:
                ctrl.count_p[name] = ctrl.count_p[name].to(layer_device)
            if ctrl.cooldowns[name].device != layer_device:
                ctrl.cooldowns[name] = ctrl.cooldowns[name].to(layer_device)

    def _compute_complexity(ctrl: RankEvolutionController, mode: str) -> float:
        """论文 Eq. 127 / 131：结构复杂度 C(z)。"""
        if mode == "rank_sum":
            # Eq. 127: C(z) = Σ_ℓ r_ℓ
            return float(sum(layer.get_active_rank() for layer in ctrl.layers.values()))
        if mode == "size_aware":
            # Eq. 131: C(z) = Σ_ℓ (d_ℓ + k_ℓ) r_ℓ
            return float(
                sum((layer.in_features + layer.out_features) * layer.get_active_rank() for layer in ctrl.layers.values())
            )
        raise ValueError(f"未知 complexity_mode: {mode}")

    def _select_population(
        mutations: List[ModuleMutation],
        pop_size: Optional[int],
        strategy: str,
    ) -> List[ModuleMutation]:
        if pop_size is None or pop_size <= 0 or pop_size >= len(mutations):
            return mutations
        if strategy == "all":
            return mutations[:pop_size]
        if strategy == "random":
            rng = random.Random(random_seed)
            return rng.sample(mutations, k=pop_size)
        raise ValueError(f"未知 population_strategy: {strategy}")

    def _reset_optimizer_state_for_component(
        opt: torch.optim.Optimizer,
        layer: EvoRankLoRALayer,
        index: int,
    ) -> None:
        def _zero_matching_state(param: nn.Parameter, selector) -> None:
            state = opt.state.get(param, None)
            if not state:
                return
            for value in state.values():
                if torch.is_tensor(value) and value.shape == param.shape:
                    selector(value)

        _zero_matching_state(layer.lora_A.weight, lambda tensor: tensor[index, :].zero_())
        _zero_matching_state(layer.lora_B.weight, lambda tensor: tensor[:, index].zero_())

    def _reset_optimizer_state_for_mutation(
        opt: torch.optim.Optimizer,
        mutation: ModuleMutation,
    ) -> None:
        if isinstance(mutation, ExpandMutation):
            _reset_optimizer_state_for_component(opt, mutation.layer, mutation.index)
            return
        if isinstance(mutation, PruneMutation):
            _reset_optimizer_state_for_component(opt, mutation.layer, mutation.index)
            return
        if isinstance(mutation, ReallocateMutation):
            _reset_optimizer_state_for_mutation(opt, mutation.prune_mut)
            _reset_optimizer_state_for_mutation(opt, mutation.expand_mut)

    model.train()
    _sync_controller_state_device(controller)
    inputs, targets = train_batch
    model_device = next(model.parameters()).device

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    train_loss = loss_fn(logits, targets)
    train_loss.backward()

    result: Dict[str, Any] = {
        "train_loss": float(train_loss.detach().item()),
        "did_evolution": False,
        "num_mutations": 0,
        "best_reward": None,
        "best_mutation": None,
    }

    should_evolve = (
        step >= warmup_steps and T_es > 0 and step % T_es == 0
    )
    did_cache_stats = False
    if should_evolve:
        # 在梯度裁剪之前缓存 g_ℓ 与分量分数，对应 ∂L/∂ΔW（与论文一致）；裁剪会改变范数口径。
        # cache_statistics_from_current_gradients 内对 g_ℓ 与各分量评分只各算一次并写入缓存，供 update_statistics 复用。
        for layer in controller.layers.values():
            layer.cache_statistics_from_current_gradients()
        did_cache_stats = True

    # 梯度裁剪在反传后、优化器步之前执行。
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )

    # 1) Warmup（论文 Alg. 2 line 2–3, Sec 3.7: 前 10% steps 禁止结构演化）
    if step < warmup_steps:
        optimizer.step()
        return result

    # 2) 外环结构演化（论文 Alg. 1 line 7–12; 统计已在裁剪前写入 layer 缓存）
    if should_evolve:
        controller.update_statistics()
        # DDP 下为避免统计量在各卡之间出现微小差异，进一步对 EMA 统计做全局一致化。
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for name in controller.layers:
                u = torch.tensor([controller.ema_u[name]], device=model_device, dtype=torch.float32)
                dist.all_reduce(u, op=dist.ReduceOp.SUM)
                controller.ema_u[name] = (u / world_size).item()
                dist.all_reduce(controller.ema_s[name], op=dist.ReduceOp.SUM)
                controller.ema_s[name] = controller.ema_s[name] / world_size
        tau_grow, tau_prune = controller.compute_thresholds()
        controller.tick_evolution_state(tau_grow=tau_grow, tau_prune=tau_prune)
        mutations = controller.generate_mutations()
        mutations = _select_population(mutations, lambda_pop, population_strategy)

        result["did_evolution"] = True
        result["num_mutations"] = len(mutations)

        # Trial 阶段：逐个 apply -> (mini-val-set 平均评估) -> undo，禁止污染训练图。
        # DDP：各 rank 上 generate_mutations 得到的列表长度可能不一致（浮点/采样边界），
        # 若仍用「if mutations and val_batches」整段跳过，则会出现部分 rank 少做 all_reduce，
        # NCCL 集体次数不一致 -> watchdog 超时 / SIGABRT。此处用 MAX 对齐长度并用占位 mutation 补齐。
        val_batches = _iter_val_batches(val_batch)
        if val_batches:
            n_local = len(mutations)
            if dist.is_available() and dist.is_initialized():
                n_tensor = torch.tensor([n_local], device=model_device, dtype=torch.long)
                dist.all_reduce(n_tensor, op=dist.ReduceOp.MAX)
                n_max = int(n_tensor.item())
            else:
                n_max = n_local
            while len(mutations) < n_max:
                mutations.append(_PaddingTrialMutation())

            was_training = model.training
            model.eval()

            rewards: List[Tuple[float, Optional[ModuleMutation]]] = []
            with torch.no_grad():
                base_eval_losses: List[torch.Tensor] = []
                for val_inputs, val_targets in val_batches:
                    val_inputs_dev = {k: v.to(model_device) for k, v in val_inputs.items()}
                    val_targets_dev = val_targets.to(model_device)
                    base_logits = model(val_inputs_dev)
                    base_eval_losses.append(loss_fn(base_logits, val_targets_dev).detach())
                base_eval_loss = torch.stack(base_eval_losses).mean()
                if dist.is_available() and dist.is_initialized():
                    reduced_base = base_eval_loss.clone()
                    dist.all_reduce(reduced_base, op=dist.ReduceOp.SUM)
                    base_eval_loss = reduced_base / dist.get_world_size()
                if include_noop_candidate:
                    # 论文 Eq. 167 elitist selection: z_t (no-op) 始终在候选集中
                    # 论文 Eq. 163: R(z) = -L_val(Θ; z) - λ_c · C(z)
                    base_reward = -float(base_eval_loss.item()) - lambda_c * _compute_complexity(controller, complexity_mode)
                    rewards.append((base_reward, None))  # no-op reward

                for mutation in mutations:
                    if isinstance(mutation, _PaddingTrialMutation):
                        eval_loss = base_eval_loss.detach().clone()
                        committed_candidate: Optional[ModuleMutation] = None
                    else:
                        mutation.apply()
                        eval_losses: List[torch.Tensor] = []
                        for val_inputs, val_targets in val_batches:
                            val_inputs_dev = {k: v.to(model_device) for k, v in val_inputs.items()}
                            val_targets_dev = val_targets.to(model_device)
                            val_logits = model(val_inputs_dev)
                            batch_eval_loss = loss_fn(val_logits, val_targets_dev)
                            eval_losses.append(batch_eval_loss.detach())

                        eval_loss = torch.stack(eval_losses).mean()
                        committed_candidate = mutation
                    if dist.is_available() and dist.is_initialized():
                        reduced = eval_loss.clone()
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        eval_loss = reduced / dist.get_world_size()

                    reward = -float(eval_loss.item()) - lambda_c * _compute_complexity(controller, complexity_mode)
                    rewards.append((reward, committed_candidate))
                    if not isinstance(mutation, _PaddingTrialMutation):
                        mutation.undo()

            if was_training:
                model.train()

            if rewards:
                # 论文 Eq. 167: z_{t+1} = argmax_{z ∈ {z_t} ∪ N(z_t)} R(z)
                # Theorem 4.4 保证 R(z_{t+1}) ≥ R(z_t)（单调不减）
                best_reward, best_mutation = max(rewards, key=lambda x: x[0])
                if best_mutation is not None:
                    controller.commit_mutation(best_mutation)
                    _reset_optimizer_state_for_mutation(optimizer, best_mutation)
                # ES 轮次结束后，主动回收未提交候选的缓存引用，降低潜在显存滞留风险。
                controller.cleanup_uncommitted_mutations(mutations, committed=best_mutation)
                result["best_reward"] = best_reward
                result["best_mutation"] = "noop" if best_mutation is None else best_mutation.__class__.__name__

    # 3) 参数更新（论文 Alg. 1 line 5–6: 每步对 active Θ 做梯度下降）
    optimizer.step()
    if did_cache_stats:
        for layer in controller.layers.values():
            layer.clear_statistics_cache()
    return result
