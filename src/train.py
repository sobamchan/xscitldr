import os
from argparse import ArgumentParser
from functools import partial
from itertools import product

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data import (get_abst_mt_dataset, get_at_dataset, get_cl_scitldr_dataset,
                  get_scitldr_dataset)
from model import Seq2SeqModule
from utils import collate_fn, get_tokenizer


def train(
    module: Seq2SeqModule,
    train_dl: DataLoader,
    val_dl: DataLoader,
    lr: float,
    seed: int,
    grad_accm: int,
    default_root_dir: str,
    max_epochs: int,
    freeze_embs: bool,
    prefix: str,
) -> str:
    """Train model Gith given hparams and return path to best checkpoint."""
    gpus = 1
    exp_name = f"{prefix}_lr_{lr}_seed_{seed}_ga_{grad_accm}"
    default_root_dir = os.path.join(default_root_dir, exp_name)
    module.do_freeze_embs_dec(freeze_embs)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.1,
        accumulate_grad_batches=grad_accm,
    )
    trainer.fit(module, train_dl, val_dl)
    return checkpoint_callback.best_model_path


def run(
    ip_type: str,
    lang: str,
    bs: int,
    default_root_dir: str,
    ip_max_epochs: int,
    max_epochs: int,
):
    """Run intermediate pretraining and finetune on target SciTLDR lang dataset sequentially.
    Train models on every combination of hparams.
    For each hparam set, keep only the best model regarding to val loss.
    Args:
        ip_type (str): type of intermediate pretraining, `lang` or `task`
        lang (str): language for lang-ip, and final
        bs (int): batch size
        default_root_dir (str): base dir to save checkpoints
        ip_max_epochs (int): num of max epochs for intermediate finetuning
        max_epochs (int): num of max epochs for final finetuning
    """
    li_lr = [1e-5, 3e-5]
    li_seed = [1122, 22]
    li_grad_accm = [8]

    #
    # -- INTERMEDIATE PRETRAINING PREPARATION --
    #

    # Load dataset for intermediate pretraining
    ip_train_ds, ip_val_ds = None, None
    if ip_type == "task":
        ip_train_ds, ip_val_ds = get_scitldr_dataset()
    elif ip_type == "abst_mt":
        ip_train_ds, ip_val_ds = get_abst_mt_dataset(lang)
    elif ip_type == "none":
        pass
    else:
        raise ValueError(f"`ip_type` needs to be `task`, `abst_mt` or `none`")

    # Get tokenizers
    src_tokenizer = get_tokenizer("en")
    tgt_tokenizer = src_tokenizer
    if ip_type == "task":
        # Target language is English too.
        pass
    elif ip_type == "abst_mt":
        tgt_tokenizer = get_tokenizer(lang)
    elif ip_type == "none":
        pass
    else:
        raise ValueError(f"`ip_type` needs to be `task`, `lang`, `abst_mt` or `none`")

    pad_token_id = 1

    # Set DataLoader
    ip_train_dl, ip_val_dl = None, None
    if ip_type != "none":
        collate_fner = partial(collate_fn, pad_token_id, src_tokenizer, tgt_tokenizer)
        ip_train_dl = DataLoader(
            ip_train_ds, collate_fn=collate_fner, batch_size=bs, shuffle=True
        )
        ip_val_dl = DataLoader(ip_val_ds, collate_fn=collate_fner, batch_size=bs, shuffle=False)

    #
    # -- FINETUNING PREPARATION --
    #

    # Load dataset for final finetuning.
    if lang == "ja":
        train_ds, val_ds = get_at_dataset()
    else:
        train_ds, val_ds = get_cl_scitldr_dataset(
            lang,
        )

    # Get tokenizers
    tgt_tokenizer = get_tokenizer(lang)
    collate_fner = partial(collate_fn, pad_token_id, src_tokenizer, tgt_tokenizer)
    train_dl = DataLoader(
        train_ds, collate_fn=collate_fner, batch_size=bs, shuffle=True
    )
    val_dl = DataLoader(val_ds, collate_fn=collate_fner, batch_size=bs, shuffle=False)  # type: ignore

    #
    # -- TRAIN --
    #

    for lr, seed, grad_accm in product(li_lr, li_seed, li_grad_accm):
        module = Seq2SeqModule(lr, freeze_embs_dec=True)  # Freeze for IP

        # Intermediate Pretraining
        if ip_type != "none":
            prefix = f"ip_{ip_type}_lang_{lang}"
            best_ckpt_path = train(
                module,
                ip_train_dl,
                ip_val_dl,
                lr,
                seed,
                grad_accm,
                default_root_dir,
                max_epochs=ip_max_epochs,
                freeze_embs=True,  # Freeze some parameters during Intermediate Pretraining.
                prefix=prefix,
            )

        # Finetune
        if ip_type != "none":
            module = Seq2SeqModule.load_from_checkpoint(best_ckpt_path)
        prefix = f"finetune_{lang}"
        best_ckpt_path = train(
            module,
            train_dl,
            val_dl,
            lr,
            seed,
            grad_accm,
            default_root_dir,
            max_epochs=max_epochs,
            freeze_embs=False,  # Update evreything for finetuning
            prefix=prefix,
        )

        print(f"ip_type: {ip_type}")
        print(f"lr: {lr}, seed: {seed}, grad_accm: {grad_accm}")
        print(f"best model is dumped to {best_ckpt_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ip-type", type=str, required=True, help="task or abst_mt or none")
    parser.add_argument("--lang", type=str, required=True, help="de, ja, it, or zh")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument(
        "--default-root-dir",
        type=str,
        required=True,
        help="base dir to save checkpoints",
    )
    parser.add_argument("--ip-max-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=5)
    args = parser.parse_args()

    run(
        args.ip_type,
        args.lang,
        args.bs,
        args.default_root_dir,
        args.ip_max_epochs,
        args.max_epochs,
    )
