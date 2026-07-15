"""
Training script for CACR
"""
from run import CACR_run

CACR_run(model_name='CACR', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
