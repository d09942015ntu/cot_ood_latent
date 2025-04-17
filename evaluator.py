import argparse
import glob
import os

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.nn import MSELoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mydataset import TestDataset


def evaluate(model, dataloader, limit=np.inf, verbose=False):
    model.eval()
    average_loss = []

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            if verbose:
                print(f'eval_step:{j}')
            device = model.device
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output2 = model.forward(
                    input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
                logits = output2['logits']

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:, :].contiguous()
                loss_mse = MSELoss()
                loss = loss_mse(shift_logits[shift_labels != -100], shift_labels[shift_labels != -100])
                loss = float(loss.detach().cpu().numpy())
                average_loss.append(loss)
                if j > limit:
                    break
    return np.average(average_loss)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="./results/checkpoints",  # "./results/checkpoint-245"
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_path', type=str, default='./data/all_64_7_100',
                        help='Path to the dataset file.')

    args = parser.parse_args()

    checkpoint_path = sorted(glob.glob(os.path.join(args.ckpt_path, "*")))[-1]
    dataset_path = args.dataset_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device)
    dataset_train_com = TestDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'),
                                    ftype='train_com')
    dataset_train_ide = TestDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'),
                                    ftype='train_ide')
    dataset_train = torch.utils.data.ConcatDataset([dataset_train_com, dataset_train_ide])
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)
    dataloader_eval_com = DataLoader(TestDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype='test_com'),
                                     batch_size=32, shuffle=False)
    dataloader_eval_ide = DataLoader(TestDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype='test_ide'),
                                     batch_size=32, shuffle=False)
    print(f"{args.dataset_path},Train accuracy: {evaluate(model, dataloader_train)}")
    print(f"{args.dataset_path}, Test_com accuracy: {evaluate(model, dataloader_eval_com)}")
    print(f"{args.dataset_path}, Test_ide accuracy: {evaluate(model, dataloader_eval_ide)}")


if __name__ == '__main__':
    main()
