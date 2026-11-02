"""bioS: 1,024 fictitious people x (occupation to forget, major to keep), 16 people per (occupation, major) cell."""

import itertools
import os

import datasets
import numpy as np

from data.qa import QADataset
from data.utils import IGNORE_INDEX

FIELDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bios_fields")

LABELS = {
    "occupation": [
        "Teacher",
        "Doctor",
        "Manager",
        "Chef",
        "Writer",
        "Painter",
        "Actor",
        "Coach",
    ],
    "major": [
        "Finance",
        "History",
        "Physics",
        "Law",
        "Music",
        "Architecture",
        "Statistics",
        "Engineering",
    ],
}

TEMPLATES = {
    "occupation": {
        "train": [
            "What is {name}'s occupation?",
            "What does {name} do for a living?",
            "Which profession does {name} practice?",
            "What job does {name} hold?",
        ],
        "eval": [
            "What is the profession of {name}?",
            "In which occupation does {name} work?",
        ],
    },
    "major": {
        "train": [
            "What did {name} major in?",
            "What was {name}'s field of study at university?",
            "Which subject did {name} study?",
            "What is {name}'s academic major?",
        ],
        "eval": [
            "What was the major of {name}?",
            "Which field did {name} study at university?",
        ],
    },
}


FORGET, RETAIN = "occupation", "major"
N_PER_CELL, TARGET_PER_CELL = 16, 4

SPLITS = {
    "full": ([("both", "all", {})], "train", False),
    "forget": ([("forget", "target", {})], "train", True),
    "retain": ([("retain", "mix", {"n_people": 256, "overlap": 0.5})], "train", True),
    "forget_eval": ([("forget", "target", {})], "eval", False),
    "retain_eval": ([("retain", "target", {})], "eval", False),
}


def _pool(name):
    with open(os.path.join(FIELDS, name + ".txt")) as f:
        return [line.strip() for line in f if line.strip()]


def make_people(seed):
    rng = np.random.default_rng(seed)
    pools = [_pool(k) for k in ("first_name", "middle_name", "last_name")]
    cells = list(itertools.product(LABELS[FORGET], LABELS[RETAIN]))
    names, seen = [], set()
    while len(names) < N_PER_CELL * len(cells):
        name = " ".join(rng.choice(p) for p in pools)
        if name not in seen:
            seen.add(name)
            names.append(name)
    people = [
        {
            "name": names[c * N_PER_CELL + r],
            "cell": c,
            "rank": r,
            FORGET: cells[c][0],
            RETAIN: cells[c][1],
            "target": r < TARGET_PER_CELL,
        }
        for c in range(len(cells))
        for r in range(N_PER_CELL)
    ]
    rng.shuffle(people)
    for i, p in enumerate(people):
        p["id"] = i
    return people


def _ordered(people, target):
    group = [p for p in people if p["target"] == target]
    return sorted(group, key=lambda p: (p["rank"], p["cell"]))


def select_people(people, which, n_people=None, overlap=None):
    if which == "all":
        return list(people)
    if which == "target":
        return _ordered(people, True)
    k = round(overlap * n_people)
    return _ordered(people, True)[:k] + _ordered(people, False)[: n_people - k]


def build_rows(people, blocks, templates):
    roles = {"forget": FORGET, "retain": RETAIN}
    rows = []
    for attr, which, kwargs in blocks:
        chosen = select_people(people, which, **kwargs)
        for role in ("forget", "retain") if attr == "both" else (attr,):
            field = roles[role]
            for person in chosen:
                for t, template in enumerate(TEMPLATES[field][templates]):
                    rows.append(
                        {
                            "question": template.format(name=person["name"]),
                            "answer": person[field],
                            "person": person["id"],
                            "attribute": field,
                            "template": t,
                        }
                    )
    return rows


class BiosQADataset(QADataset):
    def __init__(
        self,
        template_args,
        tokenizer,
        split,
        seed=0,
        max_length=512,
        predict_with_generate=False,
    ):
        blocks, templates, self.label_only = SPLITS[split]
        rows = build_rows(make_people(seed), blocks, templates)
        super().__init__(
            hf_args=None,
            template_args=template_args,
            tokenizer=tokenizer,
            max_length=max_length,
            predict_with_generate=predict_with_generate,
            data=datasets.Dataset.from_list(rows),
        )
        self.label_ids = {
            field: {label: self._single_token(label) for label in LABELS[field]}
            for field in set(self.data["attribute"])
        }

    def _single_token(self, label):
        item = self._process_sample("Q", label)
        supervised = item["input_ids"][item["labels"] != IGNORE_INDEX].tolist()
        special = set(self.tokenizer.all_special_ids)
        core = [t for t in supervised if t not in special]
        if len(core) != 1 or self.tokenizer.decode(core).strip() != label:
            raise ValueError(
                f"label {label!r} is not a single token under this tokenizer: {core}"
            )
        return core[0]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.label_only:
            positions = (item["labels"] != IGNORE_INDEX).nonzero().squeeze(1)
            item["labels"][positions[1:]] = IGNORE_INDEX
        return item
