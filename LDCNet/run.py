# MODIFIED: This script is adapted for the LGHF model.

import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model.LGHF import LGHF

import sys

from datetime import datetime
now = datetime.now()
format = "%Y/%m/%d %H:%M:%S"
formatted_now = now.strftime(format)
formatted_now = str(formatted_now)+" - "

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)
    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger

# MODIFIED: Function name updated for clarity
def LGHF_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=1, verbose_level=1, mode = '', is_training = False
):
    # Initialization
    # MODIFIED: Ensure model name is LGHF
    model_name = 'LGHF'
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config_regression(model_name, dataset_name, config_file)
    
    # NEW: Add parameters required by the new LGHF model to the args.
    # It's best practice to add these to your config.json file.
    args.shared_dim = getattr(args, 'shared_dim', 128)
    args.private_dim = getattr(args, 'private_dim', 64)
    args.acmi_cycles = getattr(args, 'acmi_cycles', 2)
    args.num_heads = getattr(args, 'num_heads', 4)
    args.dropout = getattr(args, 'dropout', 0.3)
    # NEW: Weights for the auxiliary losses
    args.w_recon = getattr(args, 'w_recon', 0.8)
    args.w_diff = getattr(args, 'w_diff', 0.1)
    args.w_consistency = getattr(args, 'w_consistency', 0.5)

    args.is_training = is_training
    args.mode = mode # train or test
    # MODIFIED: Model save path updated to LGHF
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_training:
        criterions = list(model_results[0].keys())
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time"]+["Model"] + criterions)
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        res = [formatted_now]+res
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)

    if args.is_training:
        print("Training for LGHF")

        # MODIFIED: Instantiate the new LGHF model directly.
        model = LGHF(args)
        model = model.to(args.device)

        # REMOVED: All old distillation kernels are removed as their logic
        # is now integrated into the LGHF model.

    else:
        print("Testing phase for LGHF")
        # MODIFIED: Instantiate the new LGHF model for testing.
        model = LGHF(args)
        model = model.to(args.device)

    trainer = ATIO().getTrain(args)

    # Test mode
    if args.mode == 'test':
        # MODIFIED: Load the correct saved model file for LGHF.
        model_path = f'./pt/LGHF-{args.dataset_name}.pth'
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    # Train mode
    else:
        # MODIFIED: The trainer now only needs the single LGHF model.
        # We pass it directly, not in a list. Your trainer ATIO needs to be adapted.
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        
        # MODIFIED: Load the best model saved by the trainer.
        # The save path should be consistent with what's in your trainer.
        model_path = args.model_save_path # Using the path from args
        print(f"Loading best model from {model_path} for final test.")
        model.load_state_dict(torch.load(model_path, map_location=args.device))

        results = trainer.do_test(model, dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results

