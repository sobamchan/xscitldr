import torch
from transformers import AutoTokenizer

LANG_CODE_TABLE = {
    "en": "en_XX",
    "de": "de_DE",
    "ja": "ja_XX",
    "it": "it_IT",
    "zh": "zh_CN",
}


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Taken from huggingface/transformers
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


def collate_fn(
    pad_token_id,
    src_tokenizer,
    tgt_tokenizer,
    batch,
):
    src = [b["source"] for b in batch]
    tgt = [b["target"] for b in batch]
    src_inp = src_tokenizer(
        src, padding=True, max_length=512, return_tensors="pt", truncation=True
    )
    with tgt_tokenizer.as_target_tokenizer():
        tgt_inp = tgt_tokenizer(
            tgt, padding=True, max_length=512, return_tensors="pt", truncation=True
        )
    labels = tgt_inp["input_ids"]
    decoder_input_ids = shift_tokens_right(labels, pad_token_id)
    # input_ids, labels, attention_mask, decoder_input_ids
    return (src_inp["input_ids"], labels, src_inp["attention_mask"], decoder_input_ids)


def get_tokenizer(lang: str) -> AutoTokenizer:
    tgt_lang = LANG_CODE_TABLE[lang]
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang=tgt_lang
    )
    return tokenizer
