import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()

        if transformers not in TRANSFORMERS_MAP:
            raise ValueError(f"Unsupported transformer type: {transformers}")

        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]

        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
        self.transformers = transformers

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text: [batch_size, 3, seq_len]
        3 channels:
            0 -> input_ids
            1 -> attention_mask
            2 -> token_type_ids / segment_ids
        """
        if text.dim() != 3 or text.size(1) != 3:
            raise ValueError(
                f"BertTextEncoder expects input shape [B, 3, L], but got {tuple(text.shape)}"
            )

        input_ids = text[:, 0, :].long()
        attention_mask = text[:, 1, :].long()
        token_type_ids = text[:, 2, :].long()

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # RoBERTa usually does not use token_type_ids
        if self.transformers == 'bert':
            model_inputs["token_type_ids"] = token_type_ids

        if self.use_finetune:
            outputs = self.model(**model_inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**model_inputs)

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_states = outputs.last_hidden_state
        else:
            last_hidden_states = outputs[0]

        return last_hidden_states
