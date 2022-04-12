import pytorch_lightning as pl
from transformers import AutoModel, AdamW
import torch
import torch.nn as nn


class Seq2SeqModule(pl.LightningModule):
    def __init__(self, lr: float, freeze_embs_dec: bool):
        super().__init__()
        model_name = "facebook/mbart-large-cc25"
        self.bart = AutoModel.from_pretrained(model_name)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.bart.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            self.bart.config.d_model, self.bart.shared.num_embeddings, bias=False
        )
        # tie
        self.lm_head.weight = self.bart.shared.weight
        self.lm_head.out_features = self.bart.shared.num_embeddings

        self.do_freeze_embs_dec(freeze_embs_dec)

        self.save_hyperparameters()

    def do_freeze_embs_dec(self, do_freeze: bool) -> None:
        print("freeze embs" if do_freeze else "update embs")
        requires_grad = not do_freeze
        for param in self.bart.encoder.embed_tokens.parameters():
            param.requires_grad = requires_grad
        for param in self.bart.decoder.embed_tokens.parameters():
            param.requires_grad = requires_grad
        for param in self.bart.shared.parameters():
            param.requires_grad = requires_grad
        for param in self.lm_head.parameters():
            param.requires_grad = requires_grad
        for param in self.bart.get_decoder().parameters():
            param.requires_grad = requires_grad

    def forward(self, batch):
        input_ids, labels, attention_mask, decoder_input_ids = batch
        out = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = self.lm_head(out[0]) + self.final_logits_bias
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.bart.config.vocab_size), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bart_params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(bart_params, lr=self.hparams["lr"], eps=1e-8, weight_decay=0.01)
