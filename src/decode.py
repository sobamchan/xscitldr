import glob
import os
from argparse import ArgumentParser
from itertools import product
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBartTokenizer

from model import Seq2SeqModule
from utils import LANG_CODE_TABLE, get_tokenizer


def get_name_fpath(base_dir):
    fpaths = glob.glob(f"{base_dir}/**/*.ckpt", recursive=True)
    fpaths = [fpath for fpath in fpaths if not os.path.isdir(fpath)]
    exp_names = []
    target_fpaths = []
    for fpath in fpaths:
        for split in os.path.normpath(fpath).split(os.sep):
            if split.startswith("finetune_") or split.startswith("datasize_"):
                exp_names.append(split)
                target_fpaths.append(fpath)
    return target_fpaths, exp_names


def module_to_model(module: Seq2SeqModule) -> MBartForConditionalGeneration:
    """pytorch_lightning module to hf model"""
    model = MBartForConditionalGeneration(module.bart.config)
    model.model = module.bart
    model.lm_head = module.lm_head
    model.final_logits_bias = module.final_logits_bias
    return model


def summarize(
    model: MBartForConditionalGeneration,
    tokenizer: MBartTokenizer,
    tgt_lang: str,
    src_sents: List[str],
    num_beams: int,
    repetition_penalty: float,
    use_gpu: bool,
    bs: int,
) -> List[str]:
    """
    Generate hypo for given senteneces on given decoding hparams.
    Args:
        model (MBartForConditionalGeneration): trained model to use
        tokenizer (MBart50TokenizerFast): tokenizer to use
        tgt_lang (str): language code to generate summaries in (ja, de, zh, it)
        src_sents (List[str]): source sents to summarize
        num_beams (int): number of beams
        repetition_penalty (float): penalty weight for repetition
        use_gpu (bool): if use gpu or not
        bs (int): batch size
    Returns:
        List[str]: Hypos
    """
    gen_max_len = 64
    early_stopping = True
    no_repeat_ngram_size = 3

    model.eval()
    model = model.to("cuda") if use_gpu else model

    lang_code = LANG_CODE_TABLE[tgt_lang]  # eg. en -> en_XX

    with torch.no_grad():
        hypos = []
        for i in tqdm(range(0, len(src_sents), bs), total=int(len(src_sents) / bs)):
            sents = src_sents[i : i + bs]
            batch_inputs = tokenizer(
                sents, padding="max_length", truncation=True, return_tensors="pt"
            )

            if use_gpu:
                batch_inputs = batch_inputs.to("cuda")

            summary_ids = model.generate(
                batch_inputs["input_ids"],
                max_length=gen_max_len,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                repetition_penalty=repetition_penalty,
                decoder_start_token_id=tokenizer.lang_code_to_id[lang_code],
            )
            with tokenizer.as_target_tokenizer():
                _hypos = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            hypos += _hypos

    return hypos


def summarize_grid(
    model: MBartForConditionalGeneration,
    tokenizer: MBartTokenizer,
    tgt_lang: str,
    src_sents: List[str],
    use_gpu: bool,
    num_beams: Optional[int],
    rp: Optional[float],
    bs: int,
) -> Dict[str, List[str]]:
    """
    Given model, generate hypos on every hparam combinations
    Args:
        model (MBartForConditionalGeneration): trained model to use
        tokenizer (MBart50TokenizerFast): tokenizer to use
        tgt_lang (str): language code to generate summaries in (ja, de, zh, it)
        src_sents (List[str]): source sents to summarize
        use_gpu (bool): if use gpu or not
        num_beams (int): num of beams to use, if None, run grid search.
        rp (int): repetition peanalty to use, if None, run grid search.
    Returns:
        Dict[str, List[str]]: hparam config string as key, and corresponding hypos
    """
    # params to search
    li_num_beams = list(range(2, 3 + 1)) if not num_beams else [num_beams]
    li_repetition_penalty = [0.8, 1.0] if not rp else [rp]

    results: Dict[str, List[str]] = {}

    for num_beams, repetition_penalty in product(li_num_beams, li_repetition_penalty):
        hypos = summarize(
            model,
            tokenizer,
            tgt_lang,
            src_sents,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            use_gpu=use_gpu,
            bs=bs,
        )
        config_name = f"num_beams_{num_beams}_repetition_penalty_{repetition_penalty}"
        results[config_name] = hypos

    return results


def run(
    src_path: str,
    base_dir: str,
    save_dir: str,
    tgt_lang: str,
    use_gpu: bool,
    num_beams: Union[int, None],
    rp: Union[float, None],
    bs: int,
):
    """
    Collect all the checkpoints under `base_dir`, for each model, for each decoding param
    generate hypos and save.
    Args:
        src_path (str): path to src file (eg. val.source), each line for each sample.
        base_dir (str): base dir path to search checkpoints, all checkpoints under this will be tried, if it's a file evaluate single model.
        save_dir (str): dir to save generated hypos
        tgt_lang (str): target language to generate
        use_gpu (bool): if use gpu or not
    """
    # Get paths to all the models to evaluate
    if base_dir is not None:
        if os.path.isdir(base_dir):
            module_paths, module_names = get_name_fpath(base_dir)
        else:
            module_paths = [base_dir]
            module_names = [
                [
                    split
                    for split in os.path.normpath(base_dir).split(os.sep)
                    if split.startswith("finetune_")
                    or split.startswith("datasize_")
                    or split.startswith("ip_")
                ][0]
            ]
    else:
        # If `base_dir` is None, run zero-shot summarization by just pretrained mBART.
        module_paths = ["mbart-large-cc25"]
        module_names = ["mbart-large-cc25"]

    # tokenizer
    tokenizer = get_tokenizer(tgt_lang)

    # Load source sentences to summarize
    with open(src_path, "r", encoding="utf-8") as _f:
        src_sents = [line.strip() for line in _f.readlines()]

    for module_path, module_name in tqdm(
        zip(module_paths, module_names), total=len(module_paths)
    ):
        # Load a model and convert it into MBartForConditionalGeneration class.
        if module_path != "mbart-large-cc25":
            module = Seq2SeqModule.load_from_checkpoint(module_path, strict=False)
            model = module_to_model(module)
        else:
            module = Seq2SeqModule(1e-5, freeze_embs_dec=True)  # Freeze for IP
            model = module_to_model(module)

        # Generate hypos for each dec hparams
        conf2hypos = summarize_grid(
            model,
            tokenizer,
            tgt_lang,
            src_sents,
            use_gpu,
            num_beams,
            rp,
            bs,
        )

        # Save
        for conf_str, hypos in conf2hypos.items():
            save_fname = f"{module_name}_{conf_str}.hypo"
            save_path = os.path.join(save_dir, save_fname)
            with open(save_path, "w", encoding="utf-8") as _f:
                _f.write("\n".join(hypos))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""For all the checkpoints under `base_dir`,
                       generate hypos with all the combination of decoding hparams"""
    )
    parser.add_argument(
        "--src-path",
        type=str,
        required=True,
        help="path to src file (eg. val.source), each line for each sample.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default=None,
        help="""base dir path to search checkpoints, all checkpoints under this will be tried.
                If it's a file evaluate single model.
                If not give, use just pretrained mbart.
        """,
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="dir to save generated hypos"
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        required=True,
        help="target language to generate (de, it, zh, ja)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="If to use gpu or not, recommended."
    )
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--rp", type=float, default=None)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()
    run(
        args.src_path,
        args.base_dir,
        args.save_dir,
        args.tgt_lang,
        args.use_gpu,
        args.num_beams,
        args.rp,
        args.bs,
    )
