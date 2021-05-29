import argparse
import csv

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm

from model import MLP
from dataset import KUCIDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='path to MLP model')
    parser.add_argument('--data-test', type=str, required=True, help='path to test data')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='cuda id')
    args = parser.parse_args()

    # prepare dataset
    test_ds = KUCIDataset(args.data_test)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        print('cuda is unavailable')
        device = torch.device('cpu')
    print(f'device: {device}')

    # make embedding matrix from pretrained data
    print('Making embedding matrix')
    w2v_path = 'data/w2v.midasi.256.100K.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    index_to_key = word2vec.index_to_key
    bocab_size = len(index_to_key)  # 100001
    w2v_dim = len(word2vec[0])  # 256
    embedding_matrix = np.zeros((bocab_size, w2v_dim))
    for id, word in enumerate(index_to_key):
        embedding_matrix[id] = word2vec[word]
    embedding_matrix = torch.FloatTensor(embedding_matrix)

    model = MLP(embedding_matrix)
    model.load_state_dict(torch.load(args.model_path), strict=False).to(device)
    # model = torch.nn.DataParallel(model).to(device)

    print('Start prediction')
    model.eval()
    predictions = torch.LongTensor([]).to(device)
    with torch.no_grad():
        for j, (sequences, label) in tqdm(enumerate(test_dl)):
            sequences = sequences.to(device)
            y = model(sequences)            # shape (batch, 4)
            prediction = torch.argmax(y, dim=1)     # shape (batch, 1)
            predictions = torch.cat((predictions, prediction), dim=0)

    print(predictions)
    output_path = 'data/prediction/mlp_pred.csv'
    choices = ['a', 'b', 'c', 'd']
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow(choices[pred])


if __name__ == '__main__':
    main()
