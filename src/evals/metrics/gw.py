"""Gaussian-poison metrics: x = -<dCE/d(input embeddings), w> / (sigma ||g||) is N(0, 1) under the null."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from evals.metrics.base import unlearning_metric


def _ce_input_grad(model, ids):
    """d(mean-token CE)/d(input embeddings) for one sequence, [seq, D]."""
    ids = ids.to(model.device).unsqueeze(0)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb).logits[:, :-1]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1)
    )
    (g,) = torch.autograd.grad(ce, emb)
    return g[0].float()


@unlearning_metric(name="gw_detection")
def gw_detection(model, **kwargs):
    """AUROC of the poison detection statistic on the poisoned sequences (0.5 = undetectable)."""
    data = kwargs["data"]
    sigma = float(kwargs.get("sigma", 0.075))
    x, by_index = [], {}
    for i in range(len(data)):
        item = data[i]
        g = _ce_input_grad(model, item["input_ids"])
        w = item["noise_vector"].to(g.device)
        xi = float(-(g * w).sum() / (sigma * g.norm() + 1e-12))
        x.append(xi)
        by_index[i] = {"x": xi}
    phi = 0.5 * (
        1.0 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / math.sqrt(2.0))
    )
    return {"agg_value": float(phi.mean()), "value_by_index": by_index}


@unlearning_metric(name="gw_perplexity")
def gw_perplexity(model, **kwargs):
    """Perplexity on held-out sequences; per-sequence losses kept for paired comparisons."""
    data = kwargs["data"]
    losses, by_index = [], {}
    with torch.no_grad():
        for i in range(len(data)):
            ids = data[i]["input_ids"].to(model.device).unsqueeze(0)
            loss = float(model(input_ids=ids, labels=ids).loss)
            losses.append(loss)
            by_index[i] = {"loss": loss}
    return {"agg_value": float(np.exp(np.mean(losses))), "value_by_index": by_index}
