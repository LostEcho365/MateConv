"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-30 00:19:00
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
import train
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar10"
model = "resnet20"
exp_name = "train"
root = f"log/{dataset}/{model}/pre{exp_name}"
script = "train_dynamic_pruning.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/pretrain.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python", script, config_file]

    lr, id = args

    with open(os.path.join(root, f"ResNet20_lr-{lr:.4f}_id-{id}_with_pruning.log"), "w") as wfid:
        exp = [
            f"--dataset.train_valid_split_ratio=[0.95, 0.05]",
            f"--dataset.test_ratio=0.1",
            f"--dataset.transform=augment",
            f"--checkpoint.model_comment={model}_{exp_name}_{dataset}_lr-{lr:.4f}_with_pruning",
            f"--run.random_state={41+id}",
            f"--model.dpe=None",
            f"--optimizer.lr={lr}",
            f"--optimizer.weight_decay=0.0001",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        [0.002, 1],
    ]



    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")


# def task_launcher(args):
#     lr, id = args
#
#     log_path = os.path.join(root, f"ResNet20_lr-{lr:.4f}_id-{id}.log")
#
#     # Set up the arguments as if they were command line arguments
#     args = [
#         f"--dataset.train_valid_split_ratio=[0.95, 0.05]",
#         f"--dataset.test_ratio=0.1",
#         f"--dataset.transform=augment",
#         f"--checkpoint.model_comment={model}_{exp_name}_{dataset}_lr-{lr:.4f}",
#         f"--run.random_state={41 + id}",
#         f"--model.dpe=None",
#         f"--optimizer.lr={lr}",
#         f"--optimizer.weight_decay=0.0001",
#     ]
#
#     # Call the main function of train.py with the necessary arguments
#     train.main(args)
#
#     logger.info(f"Task with lr {lr} and id {id} completed. Logs are saved in {log_path}")
#
# if __name__ == "__main__":
#     ensure_dir(root)
#     mlflow.set_experiment(configs.run.experiment)  # set experiments first
#
#     tasks = [
#         [0.002, 1],
#     ]
#
#
#
#     with Pool(1) as p:
#         p.map(task_launcher, tasks)
#     logger.info(f"Exp: {configs.run.experiment} Done.")