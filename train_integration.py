from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from evo_rank_lora import EvoRankLoRALayer
from rank_evolution_controller import ModuleMutation, RankEvolutionController


class EvoRankLoRAWrapper(nn.Module):
    """
    将冻结的原线性层与可演化 LoRA 旁路组合在一起。
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
) -> Dict[str, Any]:
    """
    双时间尺度训练模板：
    - Inner loop: 每步都做参数梯度下降（optimizer.step）
    - Outer loop: 每隔 T_es 步做一次结构演化（ES trial/commit）

    该函数只演示单步的执行顺序，外层可在 epoch/datalaoder 中反复调用。
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

    model.train()
    inputs, targets = train_batch

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

    # 1) Warmup：严格跳过结构演化，避免 Step 早期零梯度陷阱污染统计。
    if step < warmup_steps:
        optimizer.step()
        return result

    # 2) 每隔 T_es 步触发一次外环结构演化。
    should_evolve = (T_es > 0) and (step % T_es == 0)
    if should_evolve:
        # 必须在 optimizer.step() 之前调用，保证 LoRA 梯度仍可用于评分。
        controller.update_statistics()
        tau_grow, tau_prune = controller.compute_thresholds()
        controller.tick_evolution_state(tau_grow=tau_grow, tau_prune=tau_prune)
        mutations = controller.generate_mutations()

        result["did_evolution"] = True
        result["num_mutations"] = len(mutations)

        # Trial 阶段：逐个 apply -> (mini-val-set 平均评估) -> undo，禁止污染训练图。
        val_batches = _iter_val_batches(val_batch)
        if mutations and val_batches:
            was_training = model.training
            model.eval()

            rewards: List[Tuple[float, ModuleMutation]] = []
            with torch.no_grad():
                for mutation in mutations:
                    mutation.apply()
                    eval_losses: List[torch.Tensor] = []
                    for val_inputs, val_targets in val_batches:
                        val_logits = model(val_inputs)
                        batch_eval_loss = loss_fn(val_logits, val_targets)
                        eval_losses.append(batch_eval_loss.detach())

                    eval_loss = torch.stack(eval_losses).mean()
                    # DDP 下必须对 reward 来源的 loss 做跨卡一致化，否则会出现不同卡提交不同 mutation。
                    if dist.is_available() and dist.is_initialized():
                        reduced = eval_loss.clone()
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        eval_loss = reduced / dist.get_world_size()

                    # reward 越大越好，这里采用 reward = -eval_loss
                    reward = -float(eval_loss.item())
                    rewards.append((reward, mutation))
                    mutation.undo()

            if was_training:
                model.train()

            if rewards:
                best_reward, best_mutation = max(rewards, key=lambda x: x[0])
                controller.commit_mutation(best_mutation)
                result["best_reward"] = best_reward
                result["best_mutation"] = best_mutation.__class__.__name__

    # 3) 参数更新始终发生在本步末尾，保证训练主干稳定推进。
    optimizer.step()
    return result
