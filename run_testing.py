import glob
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from evaluator import evaluate
from mydataset import TestDataset
from utils import reinitialize_weights

dnames = [
    "discrete14_3_05",
    "discrete14_3_10",
    "discrete14_3_20",
    "discrete14_3_30",
    "discrete14_3_40",
    "discrete14_3_50",
]

for dname in dnames:
    for s in range(5):
        s_str = str(s).zfill(2)
        data_folder = f'data/{dname}_lrelu_l2_s{s_str}'
        model_name = './mymodels/gpt2_numeric'
        ckpt_basepath = f"./results/{dname}_lrelu_l2_s{s_str}/"

        h = 3
        batch_size = 64

        train_filenames = [os.path.join(data_folder, "train.json")]
        test_filenames = sorted(glob.glob(os.path.join(data_folder, "test*.json")))
        dataset_filenames = train_filenames + test_filenames
        dataset_filenames = [os.path.basename(d).replace(".json", "") for d in dataset_filenames]

        train_dataloaders = dict([(f"train_{hp}", DataLoader(
            TestDataset(data_folder, h_position=[hp], ftype=f'train', length=batch_size * 20),
            batch_size=batch_size, shuffle=False)) for hp in range(1, h)])

        test_dataloaders = dict([(f"{ttype}_{hp}", DataLoader(
            TestDataset(data_folder, h_position=[hp], ftype=ttype, length=batch_size * 20),
            batch_size=batch_size, shuffle=False))
                                 for ttype in [os.path.basename(tname).replace(".json", "")
                                               for tname in
                                               sorted(glob.glob(os.path.join(data_folder, "test*.json")))]
                                 for hp in range(1, h)])

        print(ckpt_basepath)
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_basepath, "checkpoints/checkpoint-*")))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ckpt_path = ckpt_paths[-1]
        if "checkpoints" not in ckpt_path:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True,
                                                         ignore_mismatched_sizes=True)
            reinitialize_weights(model)
            step = 1
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True,
                                                         ignore_mismatched_sizes=True)
            step = int(ckpt_path.split("-")[-1])

        model.to(device)

        print(f"ckpt_path:{ckpt_path}")

        train_loss = dict([(f"{ikey}", evaluate(model, train_dataloaders[ikey])) for ikey in train_dataloaders.keys()])
        test_loss = dict([(f"{ikey}", evaluate(model, test_dataloaders[ikey])) for ikey in test_dataloaders.keys()])

        log_item = {}
        log_item["step"] = step
        log_item["train_loss"] = train_loss
        log_item["test_loss"] = test_loss

        log_dir = os.path.join("outputs", os.path.basename(data_folder))
        os.makedirs(log_dir, exist_ok=True)

        json.dump(log_item, open(os.path.join(log_dir, "trainer.log"), "w"))
