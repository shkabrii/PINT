"""bioS metric: accuracy at the position that predicts the single-token label."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.utils import IGNORE_INDEX
from evals.metrics.base import unlearning_metric


def _label_positions(labels):
    """Index of the first supervised token per row."""
    return (labels != IGNORE_INDEX).int().argmax(1)


def _batches(model, data, collator, batch_size):
    """Yield (indices, logits at the predicting position, target ids)."""
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    for batch in loader:
        idx = batch.pop("index").tolist()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        pos = _label_positions(batch["labels"])
        rows = torch.arange(len(idx), device=model.device)
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        yield idx, logits[rows, pos - 1].float(), batch["input_ids"][rows, pos]


@unlearning_metric(name="label_accuracy")
def label_accuracy(model, **kwargs):
    """Accuracy of the most likely of the eight attribute labels at the answer position."""
    data, collator = kwargs["data"], kwargs["collators"]
    batch_size = kwargs.get("batch_size", 32)
    choices = sorted({t for ids in data.label_ids.values() for t in ids.values()})
    choice_ids = torch.tensor(choices, device=model.device)
    by_index = {}
    for idx, logits, target in _batches(model, data, collator, batch_size):
        choice = choice_ids[logits[:, choice_ids].argmax(-1)]
        for j, i in enumerate(idx):
            by_index[i] = {"correct": bool(choice[j] == target[j])}
    return {
        "agg_value": float(np.mean([v["correct"] for v in by_index.values()])),
        "value_by_index": by_index,
    }
