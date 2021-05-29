import argparse

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
    parser.add_argument('--data-train', type=str, required=True, help='path to training data')
    parser.add_argument('--data-val', type=str, required=True, help='path to validation data')
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='cuda id')
    args = parser.parse_args()

    # make embedding matrix from pretrained data
    print('Making embedding matrix')
    w2v_path = 'data/w2v.midasi.256.100K.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    index_to_key = word2vec.index_to_key
    bocab_size = len(index_to_key)      # 100001
    w2v_dim = len(word2vec[0])      # 256
    embedding_matrix = np.zeros((bocab_size, w2v_dim))
    for id, word in enumerate(index_to_key):
        embedding_matrix[id] = word2vec[word]
    embedding_matrix = torch.FloatTensor(embedding_matrix)

    # prepare dataset
    train_ds = KUCIDataset(args.data_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ds = KUCIDataset(args.data_val)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        print('cuda is unavailable')
        device = torch.device('cpu')
    print(f'device: {device}')

    model = MLP(embedding_matrix)
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_func = nn.CrossEntropyLoss()

    print('Start training')
    for i in range(args.epoch):
        print(f'epoch: {i+1} / {args.epoch}')
        total_loss = 0.0
        model.train()
        for j, (sequences, label) in tqdm(enumerate(train_dl), postfix='hoge'):
            sequences = sequences.to(device)
            label = label.to(device)
            y = model(sequences)
            optimizer.zero_grad()
            loss = loss_func(y, label)
            total_loss += loss.item() * sequences.size(0)
            loss.backward()
            optimizer.step()
        print(f'loss: {total_loss / len(train_ds)}')

        print('Start validation')
        total_loss = 0.0
        correct = 0
        model.eval()
        with torch.no_grad():
            for j, (sequences, label) in tqdm(enumerate(val_dl)):
                sequences = sequences.to(device)
                label = label.to(device)
                y = model(sequences)            # shape (batch, 4)
                loss = loss_func(y, label)
                prediction = torch.argmax(y, dim=1)     # shape (batch, 1)
                total_loss += loss.item() * sequences.size(0)
                correct += torch.sum(prediction == label).item()
        print(f'loss: {total_loss / len(train_ds)}')
        print(f'score: {correct / len(val_ds)} ({correct} / {len(val_ds)})')

    save_path = f'./data/saved_models/mlp_{args.epoch}e_{args.batch_size}b.pt'
    torch.save(model.to('cpu').state_dict(), save_path)
    print(f'model saved in "{save_path}"')


if __name__ == '__main__':
    main()
