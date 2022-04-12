from argparse import ArgumentParser
from functools import partial

from datasets.arrow_dataset import Dataset
from lineflow.core import IterableDataset
from torch.utils.data import DataLoader

from data import get_at_dataset, get_cl_scitldr_dataset
from model import Seq2SeqModule
from train import train
from utils import collate_fn, get_tokenizer


def run(
    lr: float,
    seed: int,
    grad_accm: int,
    max_epochs: int,
    lang: str,
    default_root_dir: str,
    ckpt_path: str = None,
    bs: int = 1,
):
    """Take pretrained mBART or finetuned checkpoint,
    and futher finetune it on limited size of CL-SciTLDR data (30% and 60%).
    Args:
        lr (float): learning rate
        seed (int): random seed
        grad_accm (int): gradient accumuration
        max_epochs (int): pl param
        lang (str): target language (one of [de, it, ja, zh])
        default_root_dir (str): base dir to save checkpoints.
        ckpt_path (str): path to checkpoint, if None, use pretrained mBART.
        bs (int): batch size
    """
    li_train_size = [0.01, 0.05, 0.30]

    # Load dataset
    if lang == "ja":
        train_ds, val_ds = get_at_dataset()
    else:
        train_ds, val_ds = get_cl_scitldr_dataset(lang)

    # Load tokenizer
    src_tokenizer = get_tokenizer("en")
    tgt_tokenizer = get_tokenizer(lang)

    pad_token_id = 1

    for train_size in li_train_size:
        # Slice train dataset
        n_samples = int(len(train_ds) * train_size)
        print(f"Using {train_size * 100}% of samples ({n_samples}/{len(train_ds)}).")

        if isinstance(train_ds, Dataset):
            sliced_train_ds = train_ds.filter(
                lambda _, idx: idx < n_samples, with_indices=True
            )
        else:
            sliced_train_ds = train_ds[:n_samples]

        # Prepare DataLoader
        collate_fner = partial(collate_fn, pad_token_id, src_tokenizer, tgt_tokenizer)
        train_dl = DataLoader(
            sliced_train_ds, collate_fn=collate_fner, batch_size=bs, shuffle=True
        )
        val_dl = DataLoader(
            val_ds, collate_fn=collate_fner, batch_size=bs, shuffle=False
        )

        # Prepare model
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            module = Seq2SeqModule.load_from_checkpoint(ckpt_path, strict=False)
        else:
            print("Loading just pretrained mBART")
            module = Seq2SeqModule(lr=lr, freeze_embs_dec=False)

        # Train
        prefix = f"datasize_{train_size}"
        train(
            module=module,
            train_dl=train_dl,
            val_dl=val_dl,
            lr=lr,
            seed=seed,
            grad_accm=grad_accm,
            default_root_dir=default_root_dir,
            max_epochs=max_epochs,
            freeze_embs=False,  # No freezing for fine-tuning.
            prefix=prefix,
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Take vanilla or IP mBART, finetune on scaled dataset (30, 60%).
        For hparam, use best combination from one corresponding setting with full dataset."""
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grad-accm", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="target language (one of [de, it, ja, zh])",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        required=True,
        help="base dir to save checkpoints",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="path to checkpoint, if not given, use just pretrained mBART.",
    )
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    run(
        lr=args.lr,
        seed=args.seed,
        grad_accm=args.grad_accm,
        max_epochs=args.max_epochs,
        lang=args.lang,
        default_root_dir=args.default_root_dir,
        ckpt_path=args.ckpt_path,
        bs=args.bs,
    )
