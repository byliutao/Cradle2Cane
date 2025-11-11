import os
import yaml
from typing import Dict, Any
from datetime import datetime
import argparse
import re


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_training_args(parser, args, config_path: str):
    """
    Update existing args namespace with values from config file,
    prioritizing command line arguments over config values.
    
    Args:
        parser: The ArgumentParser instance
        args: The existing args namespace (result of parse_args)
        config_path: Path to the YAML configuration file
    
    Returns:
        args: Updated namespace with parameters from config
    """
    # Load config from YAML
    config = load_training_config(config_path)
    
    # Get the default values for each argument
    defaults = {action.dest: action.default for action in parser._actions if hasattr(action, 'dest')}
    
    # Process each config parameter and update args
    for param_name, param_value in config.items():

        # Special handling for scientific notation that might be read as strings
        if isinstance(param_value, str) and is_scientific_notation(param_value):
            try:
                param_value = float(param_value)
            except ValueError:
                pass
            
        # Skip if the arg was explicitly provided in command line
        # (current value is different from default)
        if hasattr(args, param_name) and getattr(args, param_name) != defaults.get(param_name):
            continue
        
        # Special handling for boolean flags
        if isinstance(getattr(args, param_name, None), bool) and not isinstance(param_value, bool):
            # Convert string 'true'/'false' to boolean
            if isinstance(param_value, str):
                setattr(args, param_name, param_value.lower() == 'true')
            else:
                setattr(args, param_name, bool(param_value))
            continue
        
        # Automatic type conversion based on existing arg type
        current_value = getattr(args, param_name, None)
        if current_value is not None:
            try:
                # Convert to same type as current value
                value_type = type(current_value)
                typed_value = value_type(param_value)
                setattr(args, param_name, typed_value)
            except Exception as e:
                print(f"Warning: Could not convert '{param_name}' value '{param_value}' to {type(current_value).__name__}: {str(e)}")
        else:
            # No existing value, use as is
            setattr(args, param_name, param_value)
    
    # Add metadata about loading
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    setattr(args, '_config_loaded_at', timestamp)
    setattr(args, '_config_loaded_from', os.path.basename(config_path))
    setattr(args, '_config_loaded_by', os.getenv('USER', 'unknown'))

    # args.output_dir = f"{args.output_dir}-{timestamp}"
    
    return args


def is_scientific_notation(value):
    """Check if a string represents a number in scientific notation."""
    if not isinstance(value, str):
        return False
    # Match patterns like "1e-2", "1.5e+3", etc.
    return bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)$', value))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to the training configuration file (YAML)."
    )
    
    # some of the arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/model/sdxl-turbo",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="/data/model/sdxl-vae-fp16-fix",
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset/agedb_source/dataset_config.yaml",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_train_epochs", type=int, default=None,)
    parser.add_argument("--train_batch_size", type=int, default=None,)
    parser.add_argument("--validation_steps", type=int, default=None,)
    parser.add_argument("--max_load_num", type=int, default=None,)
    parser.add_argument("--max_train_steps", type=int, default=None,)
    parser.add_argument("--checkpointing_steps", type=int, default=None,)
    parser.add_argument("--t1", type=int, default=None,)
    parser.add_argument("--t2", type=int, default=None,)
    parser.add_argument("--id_cos_loss_weight", type=float,default=None,)
    parser.add_argument("--age_loss_weight", type=float,default=None,)
    parser.add_argument("--age_loss_2_weight", type=float,default=None,)
    parser.add_argument("--pixel_mse_loss_weight", type=float,default=None,)
    parser.add_argument("--ssim_loss_weight", type=float,default=None,)
    parser.add_argument("--g_loss_weight", type=float,default=None,)
    parser.add_argument("--lpips_loss_weight", type=float,default=None,)
    parser.add_argument("--one_threshold", action="store_true",)


    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    # Read from config file
    args = get_training_args(parser, args, args.config)
    # Get the number of parameters
    num_params = len(vars(args))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args