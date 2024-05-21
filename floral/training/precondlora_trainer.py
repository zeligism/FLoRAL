import torch
import torch.nn as nn
from typing import Any
from .trainer import Trainer
from flwr.common.logger import logger
from floral.floral import Floral
from floral.floral.floral import (
    MODULAR_IMPL, LoraExperts, LoraLinearExperts, LoraConv2dExperts,
    LoRA, LoRAList, LinearLoRA, ConvLoRA, ConvTransposeLoRA
)

PRECOND_EPS = 1e-5
INVERSE_FREE = True
MATRIX_PRECOND = True
ROUTER_PRECOND = False


class PrecondLoRATrainer(Trainer):
    def __init__(self,
                 *args,
                 precond_eps=PRECOND_EPS,
                 inverse_free=INVERSE_FREE,
                 matrix_precond=MATRIX_PRECOND,
                 router_precond=ROUTER_PRECOND,
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.precond_eps = precond_eps
        self.inverse_free = inverse_free
        self.matrix_precond = matrix_precond
        self.router_precond = router_precond

    def train_step(self, batch: Any) -> dict[str, float]:
        data, target = self.batch_preprocess(batch)
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target).mean()
        reg = self.regularizer(self.model).to(self.device)
        (loss + reg).backward()
        # Preconditioning step here
        self.precond_(self.model,
                      eps=self.precond_eps,
                      inverse_free=self.inverse_free,
                      matrix_precond=self.matrix_precond,
                      router_precond=self.router_precond)
        # TODO(refactor): Can we wrap precond_ in optimizer.step?
        #                 problem is knowledge of model structure is required.
        self.optimizer.step()

        return {"loss": loss.item(), **self.regularizer.as_dict()}

    @staticmethod
    @torch.no_grad()
    def precond_(model: Floral,
                 eps: float = PRECOND_EPS,
                 inverse_free: bool = INVERSE_FREE,
                 matrix_precond: bool = MATRIX_PRECOND,
                 router_precond: bool = ROUTER_PRECOND,
                 ) -> None:
        assert isinstance(model, Floral)
        for m in model.base_model.modules():
            module_ref = model.get_ref(m)
            if module_ref is None or module_ref not in model.lora_modules:
                continue
            lora_module = model.lora_modules[module_ref]
            if MODULAR_IMPL:
                lora_list: LoRAList = lora_module
                if len(lora_list) == 0:
                    logger.warning(f"LoRAList is empty!")
                    continue
                if not isinstance(lora_list[0], (LinearLoRA, ConvLoRA, ConvTransposeLoRA)):
                    # logger.debug(f"LoRA preconditioning for '{type(lora_list[0])}' is not implemented")
                    continue
                if router_precond:
                    lora_norms = [
                        torch.linalg.norm(torch.einsum(
                            "ma...,an...->mn...", lora.layer_out.weight, lora.layer_in.weight))
                        for lora in lora_list
                    ]
                # --- Preconditioning step here --- #
                for lora in lora_list:
                    precond_lora_(lora, eps=eps, inverse_free=inverse_free, matrix_precond=matrix_precond)
                # --------------------------------- #
            else:
                lora_experts: LoraExperts = lora_module
                if not isinstance(lora_experts, (LoraLinearExperts, LoraConv2dExperts)):
                    # logger.debug(f"LoRA preconditioning for '{type(lora_experts)}' is not implemented")
                    continue
                if lora_experts.weight_in.grad is None or lora_experts.weight_out.grad is None:
                    continue
                if router_precond:
                    lora_fused = torch.einsum("cma...,can...->cmn...", lora.layer_out.weight, lora.layer_in.weight)
                    lora_norms = torch.stack([torch.linalg.norm(lora_fused[c]) for c in range(len(lora_fused))]).view(-1)
                precond_lora_experts_(lora_experts, eps=eps, inverse_free=inverse_free)

        # TODO(experimental): preconditioning the router.
        if router_precond and not model.router_per_layer and model.router.weight.grad is not None:
            for c in range(model.num_clusters):
                model.router.weight.grad[c].div_(lora_norms[c].add(eps))


@torch.no_grad()
def precond_lora_(lora: LoRA,
                  eps: float = PRECOND_EPS,
                  inverse_free: bool = INVERSE_FREE,
                  matrix_precond: bool = MATRIX_PRECOND,
                  ) -> None:
    if not matrix_precond:
        if len(lora.layer_in.weight.size()) > 1:
            precond_tensor_lora_(lora.layer_in.weight, lora.layer_out.weight, eps=eps)
    else:
        if isinstance(lora, LinearLoRA):
            precond_matrix_lora_(lora.layer_in.weight, lora.layer_out.weight, eps=eps, inverse_free=inverse_free)
        elif isinstance(lora, (ConvLoRA, ConvTransposeLoRA)):
            # TODO(experimental): for conv/convtranspose loras, how to invert properly?
            precond_tensor_lora_(lora.layer_in.weight, lora.layer_out.weight, eps=eps)


@torch.no_grad
def precond_matrix_lora_(W_in: torch.Tensor,
                         W_out: torch.Tensor,
                         eps: float = PRECOND_EPS,
                         inverse_free: bool = INVERSE_FREE,
                         ) -> None:
    """
    precond = U.T U for _V_, and V.T V for _U_ (assuming U, V: dim x rank):
        g_U = g_W V -> g_U (V.T V)^-1 = g_W V (V.T V)^-1
        g_V = g_W.T U -> g_V (U.T U)^-1 = g_W.T U (U.T U)^-1
    """
    if W_in.grad is None or W_out.grad is None:
        return
    P_in = W_out.T @ W_out  # preconditioner for W_in.grad
    P_out = W_in @ W_in.T  # preconditioner for W_out.grad
    diag = torch.diag(torch.ones(len(P_in)).mul(eps)).to(P_in)
    # option for inverse-free implementation of preconditioning
    if inverse_free:
        W_in.grad.copy_(torch.linalg.solve(P_in.add(diag), W_in.grad))
        W_out.grad.copy_(torch.linalg.solve((P_out.add(diag)).T, W_out.grad.T).T)
    else:
        W_in.grad.copy_(torch.linalg.inv(P_in.add(diag)) @ W_in.grad)
        W_out.grad.copy_((torch.linalg.inv(P_out.add(diag)).T @ W_out.grad.T).T)


@torch.no_grad
def precond_tensor_lora_(W_in: torch.Tensor,
                         W_out: torch.Tensor,
                         eps: float = PRECOND_EPS,
                         ) -> None:
    if W_in.grad is None or W_out.grad is None:
        return
    P_in = torch.einsum("ma...,mb...->ab...", W_out, W_out)
    P_out = torch.einsum("an...,bn...->ab...", W_in, W_in)
    W_in.grad.div_(torch.linalg.norm(P_in).add(eps))
    W_out.grad.div_(torch.linalg.norm(P_out).add(eps))


@torch.no_grad()
def precond_lora_experts_(lora: LoraExperts,
                          eps: float = PRECOND_EPS,
                          inverse_free: bool = INVERSE_FREE
                          ) -> None:
    if lora.weight_in.grad is None or lora.weight_out.grad is None:
        return
    # precond = X.T X, where X is the other matrix
    in_precond = torch.einsum("cma...,cmb...->cab...", lora.weight_out, lora.weight_out)
    out_precond = torch.einsum("can...,cbn...->cab...", lora.weight_in, lora.weight_in)
    if len(in_precond.size()) == 3:  # XXX
        # linear
        for i in range(len(lora.weight_in)):
            diag = torch.diag(torch.ones(len(in_precond[i])).mul(eps)).to(in_precond)
            # g_U = g_W V -> g_U (V.T V)^-1 = g_W V (V.T V)^-1
            # g_V = g_W.T U -> g_V (U.T U)^-1 = g_W.T U (U.T U)^-1
            if inverse_free:
                lora.weight_in.grad[i] = torch.linalg.solve(in_precond[i].add(diag), lora.weight_in.grad[i])
                lora.weight_out.grad[i] = torch.linalg.solve(out_precond[i].add(diag).T, lora.weight_out.grad[i].T).T
            else:
                lora.weight_in.grad[i] = torch.linalg.inv(in_precond[i].add(diag)) @ lora.weight_in.grad[i]
                lora.weight_out.grad[i] = (torch.linalg.inv(out_precond[i].add(diag)).T @ lora.weight_out.grad[i].T).T
    else:
        # TODO(experimental): conv, how to invert?
        for i in range(len(lora.weight_in)):
            lora.weight_in.grad[i].div_(torch.linalg.norm(in_precond[i]).add(eps))
            lora.weight_out.grad[i].div_(torch.linalg.norm(out_precond[i]).add(eps))

