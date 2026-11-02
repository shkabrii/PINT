"""Gaussian-poison datasets: OLMo-2 pretraining sequences (poisoned band, clean retain / utility)."""

import glob
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pds
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from data.utils import load_hf_dataset

RETAIN_N = 150
UTILITY_N = 1000


def _train_item(ids):
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
    }


class GWForgetDataset(Dataset):
    """Poisoned sequences of the final band, batch_idx in (step - fresh_band, step]; detect mode adds their noise."""

    def __init__(self, hf_args, step, fresh_band=10000, mode="train", **kwargs):
        local = snapshot_download(
            hf_args["path"], repo_type="dataset", allow_patterns=["*.parquet"]
        )
        files = sorted(
            glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True)
        )
        band = (pc.field("batch_idx") > step - fresh_band) & (
            pc.field("batch_idx") <= step
        )
        table = pds.dataset(files, format="parquet").to_table(filter=band)
        self.ids = [
            torch.as_tensor(np.asarray(x, dtype=np.int64))
            for x in table.column("first_sequence").to_pylist()
        ]
        self.noise = (
            table.column("gaussian_noise")
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            if mode == "detect"
            else None
        )
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = self.ids[i]
        if self.mode == "train":
            return _train_item(ids)
        w = self.noise[i]
        if w.dtype == object:
            w = np.stack([np.asarray(r, dtype=np.float32) for r in w])
        noise = torch.from_numpy(np.asarray(w, dtype=np.float32)).reshape(len(ids), -1)
        return {"input_ids": ids, "noise_vector": noise, "index": i}


class TokenSequenceDataset(Dataset):
    """Pre-tokenized held-out sequences: the first RETAIN_N in train mode, UTILITY_N of the rest otherwise."""

    def __init__(self, hf_args, text_key="tokens", mode="train", **kwargs):
        ds = load_hf_dataset(**hf_args)
        if mode == "train":
            rows = range(RETAIN_N)
        else:
            rows = np.random.default_rng(0).choice(
                np.arange(RETAIN_N, ds.num_rows), UTILITY_N, replace=False
            )
            rows = sorted(rows.tolist())
        self.rows = ds.select(rows)
        self.text_key = text_key
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = torch.as_tensor(np.asarray(self.rows[i][self.text_key], dtype=np.int64))
        if self.mode == "train":
            return _train_item(ids)
        return {"input_ids": ids, "index": i}
