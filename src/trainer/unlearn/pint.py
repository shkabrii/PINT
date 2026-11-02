"""PINT: loss = lambda * HSIC(last hidden state, next token) on the forget set + retain representation MSE."""

import copy
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as grad_all_gather
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trainer.unlearn.base import UnlearnTrainer


def _median_bw(Z, max_pts=4096):
    if Z.shape[0] > max_pts:
        Z = Z[torch.randperm(Z.shape[0], device=Z.device)[:max_pts]]
    sq = torch.cdist(Z, Z) ** 2
    return sq[sq > 0].median()


def _hsic(K, L):
    """Biased HSIC estimator, (1/n^2) tr(KHLH)."""
    n = K.shape[0]
    if n < 2:
        return K.sum() * 0.0
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    return (Kc * L).sum() / (n * n)


def _gather(t, pad_value, grad=False):
    length = torch.tensor([t.shape[1]], device=t.device)
    lengths = [torch.empty_like(length) for _ in range(dist.get_world_size())]
    dist.all_gather(lengths, length)
    pad = [0, 0] * (t.dim() - 2) + [0, int(max(lengths)) - t.shape[1]]
    t = F.pad(t, pad, value=pad_value)
    if grad:
        return torch.cat(grad_all_gather(t.contiguous()), dim=0)
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, dim=0)


class PINT(UnlearnTrainer):
    def __init__(
        self,
        trainable_params_regex=None,
        freeze_untrained_params=False,
        *args,
        **kwargs,
    ):
        self.lam = float(kwargs.pop("lambda", 100.0))
        super().__init__(*args, **kwargs)
        self.trainable_params_regex = trainable_params_regex
        self.freeze_untrained_params = bool(freeze_untrained_params)
        self.bw = None
        self.ref_model = self._prepare_ref_model(self.model)
        self.distributed = dist.is_initialized() and dist.get_world_size() > 1
        self._log_terms = {}

    def _ref_copy(self, model):
        if not is_deepspeed_zero3_enabled():
            return copy.deepcopy(model)
        path = getattr(model.config, "_name_or_path", None)
        if not path:
            return copy.deepcopy(model)
        return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=model.dtype,
            attn_implementation=model.config._attn_implementation,
        )

    def _prepare_ref_model(self, model):
        ref_model = self._ref_copy(model)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            cfg = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if cfg["zero_optimization"]["stage"] == 3:
                return self._prepare_deepspeed(ref_model)
            return self._prepare_deepspeed(ref_model.to(self.accelerator.device))
        return self.accelerator.prepare_model(
            ref_model.to(self.accelerator.device), evaluation_mode=True
        )

    def create_optimizer(self):
        if not self.trainable_params_regex:
            return super().create_optimizer()
        for name, p in self.model.named_parameters():
            p.requires_grad = any(
                re.fullmatch(r, name) for r in self.trainable_params_regex
            )
        opt = super().create_optimizer()
        if not self.freeze_untrained_params:
            for p in self.model.parameters():
                p.requires_grad = True
        elif self.args.gradient_checkpointing:
            self.model.enable_input_require_grads()
        return opt

    @staticmethod
    def _last_hidden(model, inputs):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            logits_to_keep=1,
        )
        return out.hidden_states[-1]

    @staticmethod
    def _gram_points(Z, ids, labels):
        pred = torch.zeros_like(labels, dtype=torch.bool)
        pred[:, :-1] = labels[:, 1:] != -100
        tok = pred.reshape(-1).nonzero().squeeze(1)
        return Z.reshape(-1, Z.shape[-1])[tok].float(), ids.reshape(-1)[tok + 1]

    def _forget_hsic(self, Z, nxt):
        if self.bw is None:
            bw = _median_bw(Z.detach())
            if self.distributed:
                dist.broadcast(bw, src=0)
            self.bw = bw
        K = torch.exp(-(torch.cdist(Z, Z) ** 2) / self.bw.to(Z.device))
        L = (nxt[:, None] == nxt[None, :]).float()
        return _hsic(K, L)

    def _retain_loss(self, model, retain_inputs):
        Hs = self._last_hidden(model, retain_inputs)
        with torch.no_grad():
            Hr = self._last_hidden(self.ref_model, retain_inputs)
        mask = (retain_inputs["labels"] != -100).to(Hs.dtype)
        sq = F.mse_loss(Hs, Hr, reduction="none").mean(-1)
        return ((sq * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget, retain = inputs["forget"], inputs["retain"]
        Z = self._last_hidden(model, forget)
        ids, labels = forget["input_ids"], forget["labels"]
        if self.distributed:
            Z, ids, labels = (
                _gather(Z, 0.0, grad=True),
                _gather(ids, 0),
                _gather(labels, -100),
            )
        hsic = self._forget_hsic(*self._gram_points(Z, ids, labels))
        retain_loss = self._retain_loss(model, retain)
        loss = self.lam * hsic + retain_loss
        self._log_terms = {
            "forget_hsic": hsic.item(),
            "retain_loss": retain_loss.item(),
        }
        return (loss, {"last_hidden_state": Z}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs.update({k: round(v, 6) for k, v in self._log_terms.items()})
        return super().log(logs, *args, **kwargs)
