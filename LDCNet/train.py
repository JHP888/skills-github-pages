"""
Training script for LGHF (Language-Guided Hyper-modality Framework)
"""
# MODIFIED: Import and call the new LGHF_run function
from run import LGHF_run

# MODIFIED: The model_name must be 'LGHF' to match our new configuration in config.json
LGHF_run(model_name='LGHF', 
         dataset_name='mosi', 
         is_tune=False, 
         seeds=[1111], 
         model_save_dir="./pt",
         res_save_dir="./result", 
         log_dir="./log", 
         mode='train', 
         is_training=True)

