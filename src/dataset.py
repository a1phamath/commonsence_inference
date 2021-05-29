import json

import torch
import torch.nn.utils.rnn as rnn
import pandas as pd
import numpy as np
import gensim
import MeCab
from transformers import BertTokenizer


class KUCIDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        print('Preparing KUCI Dataset')
        df = pd.read_json(data_path, orient='records', lines=True)
        self.sentences, self.labels = self.preprocess(df)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return sentence, label

    @staticmethod
    def preprocess(df):
        ''' データセットをMLPモデルに入力できるように整形する
        Args:
            df: DataFrame

        Returns:
            sentences: torch.LongTensor (data_size, 4, length_max)
                       4 word sequences in each data (translated into word ids)
            labels: torch.Tensor (data_size, 4)
                    one-hot expression
        '''

        # combine contexts and choices
        data_size = len(df)
        sentences = np.empty((data_size, 4), dtype=object)
        choices = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
        for j in range(4):
            choice_values = df['context'].values + ' ' + df[choices[j]].values
            choice_values = np.array(list(map(lambda x: x.split(), choice_values)), dtype=object)
            sentences[:, j] = choice_values

        # calculate max length of sentence
        length_max = 0
        length_max = max(max(len(words) for words in sentence) for sentence in sentences)
        print('max length of word sequence: {}'.format(length_max))

        # make sentences as word id sequences
        w2v_path = 'data/w2v.midasi.256.100K.bin'
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        index_to_key = word2vec.index_to_key
        key_to_index = word2vec.key_to_index

        sentences_id = torch.zeros((data_size, 4, length_max), dtype=torch.long)
        unk_id = key_to_index['<UNK>']
        for i in range(data_size):
            for j in range(4):
                # print(i, j)
                # print(sentences[i][j])
                sequence = torch.LongTensor(list(map(lambda x: key_to_index.get(x, unk_id), sentences[i][j])))
                padding_size = length_max - len(sequence)
                sequence = torch.cat((sequence, torch.zeros(padding_size, dtype=torch.long)))
                # print(sequence)
                # print(len(sequence))
                # for id in sequence:
                #     print(index_to_key[id])
                sequence.reshape((length_max, -1))
                sentences_id[i][j] = sequence
        print(sentences[0])

        labels = torch.zeros(data_size, dtype=torch.long)
        if 'label' in df.columns:
            answers = ['a', 'b', 'c', 'd']
            for i in range(data_size):
                answer_id = answers.index(df['label'][i])
                labels[i] = answer_id
            print(labels[0])

        return sentences_id, labels


class KUCIDatasetForBert(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer_path):
        print('Preparing KUCI Dataset for BERT')
        data_list = []
        with open(data_path, 'r') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        self.inputs = self.preprocess(data_list, tokenizer_path)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        # inputs = self.inputs[idx]
        inputs = {k: v[idx] for k, v in self.inputs.items()}
        return inputs

    @staticmethod
    def preprocess(data_list, tokenizer_path):
        ''' データセットをMLPモデルに入力できるように整形する
        Args:
            data_list: list consists of dict data
            tokenizer_path: path to BERT model for tokinization

        Returns:
            sentences: torch.LongTensor (data_size, 4, length_max)
                       4 word sequences in each data (translated into word ids)
            labels: torch.Tensor (data_size, 4)
                    one-hot expression
        '''

        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        keys = ('choice_a', 'choice_b', 'choice_c', 'choice_d')

        first_sentences = [data['context'] for data in data_list]
        second_sentences = [' '.join([data[key] for key in keys]) for data in data_list]

        inputs = tokenizer(first_sentences, second_sentences, padding=True)
        inputs = {k: torch.LongTensor(v) for k, v in inputs.items()}

        labels = [-1] * len(data_list)
        if 'label' in data_list[0].keys():
            answers = ['a', 'b', 'c', 'd']
            labels = torch.LongTensor([answers.index(data['label']) for data in data_list])
        inputs['labels'] = labels

        return inputs


def main():
    data_path = 'data/KUCI/train.jsonl'
    with open(data_path, 'r') as f:
        data_list = f.readlines()
        data_list = [json.loads(data) for data in data_list]
    print(data_list[0]['context'])

    model_path = 'data/bert/kurohashi_bert_wwm'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    keys = ['context', 'choice_a', 'choice_b', 'choice_c', 'choice_d']
    inputs = [{key: tokenizer(data[key])['input_ids'] for key in keys} for data in data_list]
    print(inputs[0])

    answers = ['a', 'b', 'c', 'd']
    labels = [answers.index(data['label']) for data in data_list]
    print(labels[:10])


if __name__ == '__main__':
    main()