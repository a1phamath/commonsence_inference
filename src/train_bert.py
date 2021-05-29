import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm

from model import BertForSequenceClassification
from dataset import KUCIDatasetForBert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='path to BERT pretrained model')
    parser.add_argument('--data-train', type=str, required=True, help='path to training data')
    parser.add_argument('--data-val', type=str, required=True, help='path to validation data')
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--gpu-id', type=str, default='0', help='cuda id')
    args = parser.parse_args()

    # prepare dataset
    train_ds = KUCIDatasetForBert(args.data_train, args.model_path)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ds = KUCIDatasetForBert(args.data_val, args.model_path)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    gpu_list = list(map(int, args.gpu_id.split(',')))
    print(gpu_list)
    if torch.cuda.is_available():
        device = torch.device('cuda', gpu_list[0])
    else:
        print('cuda is unavailable')
        device = torch.device('cpu')
    print(f'device: {device}')

    config = {'num_labels': 4}
    model = BertForSequenceClassification.from_pretrained(args.model_path, my_config=config)
    model = torch.nn.DataParallel(model, gpu_list).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_func = nn.CrossEntropyLoss()

    print('Start training')
    for i in range(args.epoch):
        print(f'epoch: {i + 1} / {args.epoch}')
        total_loss = 0.0
        model.train()
        for j, data in tqdm(enumerate(train_dl), postfix='hoge'):
            inputs = {k: v.to(device) for k, v in data.items()}
            labels = inputs['labels'].to(device)
            y = model(**inputs)
            optimizer.zero_grad()
            loss = loss_func(y, labels)
            total_loss += loss.item() * len(data)
            loss.backward()
            optimizer.step()
        print(f'loss: {total_loss / len(train_ds)}')

        print('Start validation')
        total_loss = 0.0
        correct = 0
        model.eval()
        with torch.no_grad():
            for j, data in tqdm(enumerate(val_dl)):
                inputs = {k: v.to(device) for k, v in data.items()}
                labels = inputs['labels'].to(device)
                y = model(**inputs)            # shape (batch, 4)
                loss = loss_func(y, labels)
                prediction = torch.argmax(y, dim=1)     # shape (batch, 1)
                total_loss += loss.item() * len(data)
                correct += torch.sum(prediction == labels).item()
        print(f'loss: {total_loss / len(train_ds)}')
        print(f'score: {correct / len(val_ds)} ({correct} / {len(val_ds)})')

    save_path = f'./data/saved_models/bert_{args.epoch}e_{args.batch_size}b.pt'
    torch.save(model.to('cpu').state_dict(), save_path)
    print(f'model saved in "{save_path}"')


if __name__ == '__main__':
    main()
