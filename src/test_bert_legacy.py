import argparse
import csv

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm

from model import MLP
from dataset import KUCIDataset, KUCIDatasetForBert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-test', type=str, required=True, help='path to test data')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='cuda id')
    args = parser.parse_args()

    # prepare dataset
    model_path = 'data/bert/kurohashi_bert_wwm'
    test_ds = KUCIDatasetForBert(args.data_test, model_path)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        print('cuda is unavailable')
        device = torch.device('cpu')
    print(f'device: {device}')

    model_path = 'data/saved_models/bert.pt'
    model = torch.load(model_path).to(device)
    # model = torch.nn.DataParallel(model).to(device)

    print('Start prediction')
    model.eval()
    predictions = torch.LongTensor([])
    with torch.no_grad():
        for j, data in tqdm(enumerate(test_dl)):
            inputs = {k: v.to(device) for k, v in data.items()}
            y = model(**inputs)
            prediction = torch.argmax(y, dim=1)     # shape (batch, 1)
            predictions = torch.cat((predictions, prediction), dim=0)

    output_path = 'data/prediction/bert_pred.csv'
    choices = ['a', 'b', 'c', 'd']
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow(choices[pred])


if __name__ == '__main__':
    main()
