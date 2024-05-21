from typing import Union, Optional, Mapping, Callable
from functools import partial
import torch
from torch import nn
from flwr.common.logger import logger

from floral.model import Router
from .base import LoRA, LoRAList, Conv, ConvTranspose, BatchNorm, InstanceNorm
from .linear import LinearLoRA
from .embedding import EmbeddingLoRA
from .conv import ConvLoRA
from .conv_transpose import ConvTransposeLoRA
from .batchnorm import BatchNormLoRA
from .instancenorm import InstanceNormLoRA
from .layernorm import LayerNormLoRA
from .groupnorm import GroupNormLoRA

from .linear import LoraLinearExperts
from .conv import LoraConv2dExperts
from .batchnorm import LoraBatchNormExperts
from .embedding import LoraEmbeddingExperts
LoraExperts = Union[
    LoraLinearExperts, LoraConv2dExperts, LoraBatchNormExperts, LoraEmbeddingExperts
]

MODULAR_IMPL = True  # TODO: deprecate False
MODULE_NAME_SEP = '/'


class Floral(nn.Module):
    base_model: nn.Module
    lora_modules: Mapping[str, Union[LoRAList, LoraExperts]]
    router: Router

    def __init__(self,
                 base_model: nn.Module,
                 rank: float = 1,
                 num_clusters: int = 2,
                 num_clusters_mult: float = 1.0,
                 alpha: float = 1.,
                 use_linearlora: bool = True,
                 use_embeddinglora: bool = False,
                 use_convlora: bool = False,
                 use_normlora: bool = False,
                 min_rank: int = 1,
                 bias: bool = True,
                 init_strategy: Optional[str] = None,
                 convlora_method: Optional[str] = None,
                 normlora_reparam: bool = False,
                 fuse_params: bool = False,
                 router_opts: dict = {},
                 router_per_layer: bool = False,
                 ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.num_clusters = num_clusters
        self.num_clusters = round(self.num_clusters * num_clusters_mult)
        self.min_rank = min_rank
        self.alpha = alpha
        self.bias = bias
        self.use_linearlora = use_linearlora
        self.use_embeddinglora = use_embeddinglora
        self.use_convlora = use_convlora
        self.use_normlora = use_normlora
        self.init_strategy = init_strategy
        self.convlora_method = convlora_method
        # XXX: overrides use_convlora and sets it to False (mainly for sweeps)
        if self.convlora_method == "none":
            self.use_convlora = False
        self.normlora_reparam = normlora_reparam
        self.fuse_params = fuse_params
        self.router_per_layer = router_per_layer  # XXX: experimental
        self._patch_methods_from_base_model()
        self._init_module_refs()
        self._init_router(router_opts)
        self._init_lora()

    def _patch_methods_from_base_model(self):
        # "import" all public base_model methods that do not conflict with self's methods
        # (this will make all non-conflicting user-defined methods in base_model directly callable from self)
        for method_name, method_fn in self.base_model.__class__.__dict__.items():
            if not method_name.startswith('_') and method_name not in self.__class__.__dict__.keys():
                self.__setattr__(method_name, partial(method_fn, self.base_model))

    def _init_module_refs(self):
        # Gives each module a global name wrt base_model for seamless ModuleDict integration
        self._global_refs = {}
        for name, module in self.base_model.named_modules():
            # '.' not allowed in module names, any unambiguous char would do
            # self._global_refs[module] = "lora(" + MODULE_NAME_SEP.join("base_model", *name.split('.')) + ")"
            if len(name) == 0:
                self._global_refs[module] = f"lora(base_model)"
            else:
                self._global_refs[module] = f"lora({MODULE_NAME_SEP.join(['base_model'] + name.split('.'))})"
            # module._module_ref_from_base = MODULE_NAME_SEP + name.replace('.', MODULE_NAME_SEP)

    def get_ref(self, module: nn.Module):
        return self._global_refs.get(module)
        # return module._module_ref_from_base

    def _init_router(self, router_opts):
        layers = None
        if self.router_per_layer:
            # XXX: only implemented for linear and conv2d
            layers = [self.get_ref(m) for m in self.base_model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        self.router = Router(num_clusters=self.num_clusters, layers=layers, **router_opts)

    def _init_lora(self):
        self.lora_modules = {}
        self.base_model.apply(self._add_lora)
        self.lora_modules = nn.ModuleDict(self.lora_modules)  # register modules in a dict

        def _add_lora_forward_hooks(module: nn.Module):
            if self.get_ref(module) in self.lora_modules:
                module.register_forward_hook(self._lora_forward_hook)

        self.base_model.apply(_add_lora_forward_hooks)

    def _add_lora(self, m: nn.Module):
        module_ref = self.get_ref(m)
        if self.num_clusters == 0:
            return  # no need to add anything
        if module_ref is None:
            return  # could not find reference of module
        if module_ref in self.lora_modules:
            return  # a lora module already exist

        # Create a LoRA module (which is a list of LoRAs)
        if MODULAR_IMPL:
            instantiate_lora = self.create_lora_instantiator(m)
            if instantiate_lora is not None:
                self.lora_modules[module_ref] = LoRAList(
                    instantiate_lora(m, self.rank) for _ in range(self.num_clusters)
                )
        else:
            lora_experts = self.create_lora_experts(m)
            if lora_experts is not None:
                self.lora_modules[module_ref] = lora_experts

    def create_lora_instantiator(self, m) -> Optional[Callable[..., LoRA]]:
        lora_opts = {"min_rank": self.min_rank, "bias": self.bias, "init_strategy": None}
        if isinstance(m, nn.Linear) and self.use_linearlora:
            LoRALayer = LinearLoRA
        elif isinstance(m, nn.Embedding) and self.use_embeddinglora:
            LoRALayer = EmbeddingLoRA
        elif isinstance(m, Conv) and self.use_convlora:
            LoRALayer = ConvLoRA
            lora_opts["method"] = self.convlora_method
        elif isinstance(m, ConvTranspose) and self.use_convlora:
            LoRALayer = ConvTransposeLoRA
            lora_opts["method"] = self.convlora_method
        elif isinstance(m, BatchNorm) and self.use_normlora:
            LoRALayer = BatchNormLoRA
            lora_opts["reparam"] = self.normlora_reparam
        elif isinstance(m, InstanceNorm) and self.use_normlora:
            LoRALayer = InstanceNormLoRA
            lora_opts["reparam"] = self.normlora_reparam
        elif isinstance(m, nn.LayerNorm) and self.use_normlora:
            LoRALayer = LayerNormLoRA
        elif isinstance(m, nn.GroupNorm) and self.use_normlora:
            LoRALayer = GroupNormLoRA
        else:
            return None

        def instantiate_lora(*args):
            return LoRALayer(*args, **lora_opts)

        return instantiate_lora

    def create_lora_experts(self, m) -> Optional[LoraExperts]:
        # XXX: this is kept for backward compatibility
        lora_opts = {"min_rank": self.min_rank, "bias": self.bias, "fuse_params": self.fuse_params}
        if isinstance(m, nn.Linear) and self.use_linearlora:
            lora_experts = LoraLinearExperts(m, self.rank, self.num_clusters, **lora_opts)
        elif isinstance(m, nn.Embedding) and self.use_embeddinglora:
            lora_experts = LoraEmbeddingExperts(m, self.rank, self.num_clusters, **lora_opts)
        elif isinstance(m, Conv) and self.use_convlora:
            lora_opts["method"] = self.convlora_method
            lora_experts = LoraConv2dExperts(m, self.rank, self.num_clusters, **lora_opts)
        elif isinstance(m, BatchNorm) and self.use_normlora:
            lora_experts = LoraBatchNormExperts(m, self.rank, self.num_clusters, **lora_opts)
        elif isinstance(m, (ConvTranspose, InstanceNorm, nn.LayerNorm)) and self.use_normlora:
            logger.warning(f"LoRAExperts layer for module of type '{type(m)}' not implemented")
        else:
            lora_experts = None

        return lora_experts

    def fuse_loras(self):
        if self.fuse_params:
            for lora_module in self.lora_modules.values():
                lora_module.fuse()

    @torch.no_grad()
    def defuse_loras(self):
        if self.fuse_params:
            for lora_module in self.lora_modules.values():
                lora_module.defuse(self.router.routes)

    # Add lora forward hooks for the original modules only
    def _lora_forward_hook(self, inner_m, args, output):
        probs = self.router.routes
        if probs is None:
            return output
        if self.router_per_layer:
            probs = probs[self.get_ref(inner_m)]
        # '_lora_forward_hook' is ony added for modules in 'lora_modules'
        lora_module = self.lora_modules[self.get_ref(inner_m)]
        return output + self.alpha * lora_module(*args, probs)

    def forward(self, *args, **kwargs):
        self.router.reset()
        self.router()
        output = self.base_model(*args, **kwargs)
        return output

    def print_stats(self):
        count_params = lambda m: sum(p.numel() for p in m.parameters())
        base_params = count_params(self.base_model)
        lora_params = count_params(self.lora_modules)
        router_params = count_params(self.router)
        total_params = base_params + lora_params + router_params
        logger.debug(f"FLoRAL: rank = {self.rank}")
        logger.debug(f"FLoRAL: num of clusters = {self.num_clusters}")
        logger.debug(f"FLoRAL: base model parameters = {base_params}")
        logger.debug(f"FLoRAL: lora model parameters = {lora_params}")
        logger.debug(f"FLoRAL: routing model parameters = {router_params}")
        logger.debug(f"FLoRAL: total model parameters = {total_params}")
        logger.debug(f"FLoRAL: increase in parameters = {100. * (total_params / base_params - 1.):.2f}%")
        
