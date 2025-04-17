import argparse
import glob
import json
import logging
import os
import shutil
import sys

import numpy as np
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoModelForCausalLM, TrainerState, TrainerControl
import torch
from torch.utils.data import DataLoader

from evaluator import evaluate
from mydataset import TrainDataset, TestDataset
from utils import reinitialize_weights


def setup_logger(name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class StringOutputEvaluator(TrainerCallback):
    def __init__(self, model, model_path, ckpt_path, dataset_dir, logger, epsilon_loss, epsilon_diff):
        self.model = model
        self.model_path = model_path
        self.ckpt_path = ckpt_path
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.wait = 0
        self.batch_size = 32
        self.epsilon_loss = epsilon_loss
        self.epsilon_diff = epsilon_diff

        self.h = int(os.path.basename(dataset_dir) .split('_')[1])

        self.train_dataloaders = dict([(f"train_{hp}", DataLoader(
            TestDataset(self.dataset_dir, h_position=[hp], ftype=f'train', length=self.batch_size*20),
            batch_size=self.batch_size, shuffle=False)) for hp in range(1, self.h) ])

        self.test_dataloaders = dict([(f"{ttype}_{hp}", DataLoader(
            TestDataset(self.dataset_dir, h_position=[hp], ftype=ttype, length=self.batch_size*20),
            batch_size=self.batch_size, shuffle=False))
                                      for ttype in [ os.path.basename(tname).replace(".json","")
                                                     for tname in sorted(glob.glob(os.path.join(self.dataset_dir,"test*.json")))]
                                      for hp in range(1, self.h)])



    def on_log(self, args, state, control, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()

        train_accuracy = dict([(f"{ikey}", evaluate(self.model, self.train_dataloaders[ikey])) for ikey in
                               self.train_dataloaders.keys()])
        test_accuracy = dict([(f"{ikey}", evaluate(self.model, self.test_dataloaders[ikey])) for ikey in
                              self.test_dataloaders.keys()])

        epoch = state.log_history[-1]['epoch']
        loss = state.log_history[-1].get('loss', np.inf)
        step = state.log_history[-1]['step']

        history_len = len(state.log_history)
        history_avg_1 = 9999
        history_avg_2 = 0
        if history_len > 20:
            history_len_2= 20
            history_len_4= 10
            history_avg_1 = np.average([x.get('loss',9999) for x in state.log_history[-history_len_2:-history_len_4]] )
            history_avg_2 = np.average([x.get('loss',9999) for x in state.log_history[-history_len_4:]] )


        diff=abs(history_avg_1 - history_avg_2)


        log_str = json.dumps({'step': step,
                              'epoch': epoch,
                              'loss': loss,
                              'train_loss': train_accuracy,
                              'test_loss': test_accuracy,
                              "history":{
                                  "history_avg_1":history_avg_1,
                                  "history_avg_2":history_avg_2,
                                  "diff":diff}
                              })

        self.logger.info(log_str)
        print(log_str)

        if loss <= self.epsilon_loss or diff < self.epsilon_diff:
            self.wait += 1
            if self.wait > 2:
                sys.exit()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_save(args, state, control, **kwargs)  # Call the original on_save method to ensure checkpoints are saved
        ckpt_paths = sorted(glob.glob(os.path.join(args.output_dir,"checkpoint-*")), reverse=True)
        ckpt_path=ckpt_paths[0]
        modeling_path = glob.glob(os.path.join(self.model_path,'modeling_*.py'))[0]
        shutil.copy(modeling_path, ckpt_path)




def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='./mymodels/gpt2_numeric', help='Pre-trained model name or path')
    parser.add_argument('--dataset_dir', type=str, default='./data/discrete12_3_lrelu_l2_t3_n500', help='Path to the training dataset')
    parser.add_argument('--output_name', type=str, default='./temp_3', help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size')
    parser.add_argument('--logging_step', type=int, default=2000, help='Logging step')
    parser.add_argument('--save_step', type=int, default=2000, help='Logging step')
    parser.add_argument('--num_train_epochs', type=int, default=200, help='Total number of training epoch')
    parser.add_argument('--epsilon_loss', type=int, default=0.0005, help='Total number of training epoch')
    parser.add_argument('--epsilon_diff', type=float, default=0.00001,  help='Path to output directory')

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
    reinitialize_weights(model)

    dataset_train = TrainDataset(args.dataset_dir, length=args.batch_size*args.logging_step)

    output_dir = os.path.join("./results", f"{args.output_name}")
    ckpt_path = os.path.join(output_dir, "checkpoints")
    model_file_path = os.path.join(output_dir, "model")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    model_files = glob.glob(os.path.join(args.model_name, "modeling_*"))
    for model_file in model_files:
        shutil.copy(os.path.join(model_file), model_file_path)
    logger = setup_logger("my_logger", os.path.join(output_dir, "trainer.log"))

    logger.info(str(args))

    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_step,
        save_total_limit=1,
        logging_steps=args.logging_step,
        learning_rate=5e-5,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_train,
        callbacks=[StringOutputEvaluator(model, args.model_name, ckpt_path, args.dataset_dir, logger, args.epsilon_loss, args.epsilon_diff)]
    )

    trainer.train()


if __name__ == '__main__':
    main()
