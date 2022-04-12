import os
from typing import Dict, List, Tuple

from datasets import load_dataset
from lineflow.core import IterableDataset


def get_at_dataset() -> Tuple[IterableDataset, IterableDataset]:
    """Load AT dataset in form of IterableDataset for train, val
    Returns:
        Tuple[IterableDataset, IterableDataset]: datasets for train and val.
    """

    train = load_dataset(
        "umanlp/xscitldr", "ja", use_auth_token=True, split="train"
    )
    val = load_dataset(
        "umanlp/xscitldr", "ja", use_auth_token=True, split="validation"
    )

    train = IterableDataset([{"source": x["source"], "target": x["target"][0]} for x in train])
    val = IterableDataset([{"source": x["source"], "target": x["target"][0]} for x in val])

    return train, val


def get_scitldr_dataset() -> Tuple[IterableDataset, IterableDataset]:
    """Load original scitldr parallel dataset in form of IterableDataset for train, val
    Returns:
        Tuple[IterableDataset, IterableDataset]: datasets for train and val.
    """

    train = load_dataset("scitldr", "Abstract", split="train")
    val = load_dataset("scitldr", "Abstract", split="validation")

    train = IterableDataset([{"source": " ".join(x["source"]), "target": x["target"][0]} for x in train])
    val = IterableDataset([{"source": " ".join(x["source"]), "target": x["target"][0]} for x in val])

    return train, val


def get_cl_scitldr_dataset(lang: str) -> Tuple[IterableDataset, IterableDataset]:
    """Load CL-SciTLDR parallel dataset in form of IterableDataset for train, val
    Args:
        lang (str): de, it or zh.
    Returns:
        Tuple[IterableDataset, IterableDataset]: datasets for train and val.
    """
    # TODO: use huggingface/datasets.
    train = load_dataset(
        "sobamchan/xscitldr_staging", lang, use_auth_token=True, split="train"
    )
    val = load_dataset(
        "sobamchan/xscitldr_staging", lang, use_auth_token=True, split="validation"
    )

    train = IterableDataset([{"source": x["source"], "target": x["target"][0]} for x in train])
    val = IterableDataset([{"source": x["source"], "target": x["target"][0]} for x in val])

    return train, val


def get_abst_mt_dataset(lang: str) -> Tuple[IterableDataset, IterableDataset]:
    """Load enja mt parallel dataset in form of IterableDataset for train, val
    Returns:
        Tuple[IterableDataset, IterableDataset]: datasets for train and val.
    """
    ddir = f"./data/abst_mt/{lang}"
    with open(os.path.join(ddir, "train.source")) as f:
        train_src_lines = [line for line in f.readlines()]
    with open(os.path.join(ddir, "train.target")) as f:
        train_tgt_lines = [line for line in f.readlines()]
    with open(os.path.join(ddir, "val.source")) as f:
        val_src_lines = [line for line in f.readlines()]
    with open(os.path.join(ddir, "val.target")) as f:
        val_tgt_lines = [line for line in f.readlines()]

    train = IterableDataset(
        [{"source": s, "target": t} for s, t in zip(train_src_lines, train_tgt_lines)]
    )
    val = IterableDataset(
        [{"source": s, "target": t} for s, t in zip(val_src_lines, val_tgt_lines)]
    )
    return train, val
