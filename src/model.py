import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from transformers import BertPreTrainedModel
from transformers import BertModel

# from gensim.models import Word2Vec


class MLP(nn.Module):
    """KUCI用のMLPモデル
    """

    def __init__(self, embedding_matrix: torch.Tensor):
        print('Creating MLP model')
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)       # (batch, 4, seq, dim)
        x = torch.mean(x, dim=2)    # (batch, 4, dim)
        x = self.dropout(x)
        x = self.fc1(x)             # (batch, 4, 100)
        x = F.relu(x)
        x = self.fc2(x)             # (batch, 4, 1)
        x = torch.squeeze(x, dim=2)
        return x


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, my_config: dict):  # args: [config] kwargs: {'hoge': config}
        super().__init__(config)
        self.num_labels: int = my_config['num_labels']
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def main():
    from dataset import KUCIDatasetForBert
    # import ipdb;ipdb.set_trace()

    model_path = 'data/bert/kurohashi_bert_wwm'
    config = {'num_labels': 4}
    model = BertForSequenceClassification.from_pretrained(model_path, my_config=config)

    data_path = 'data/sample/train.jsonl'
    dataset = KUCIDatasetForBert(data_path)
    data, label = dataset[0]
    input_ = torch.LongTensor([data['context']['input_ids']])
    output = model(input_)
    print(output)


if __name__ == '__main__':
    main()
