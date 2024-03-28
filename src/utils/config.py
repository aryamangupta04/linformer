import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Model Configuration")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=350, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--lang_src', type=str, default='en', help='Source language')
    parser.add_argument('--lang_tgt', type=str, default='it', help='Target language')
    parser.add_argument('--model_folder', type=str, default='weights', help='Model folder path')
    parser.add_argument('--model_basename', type=str, default='tmodel_', help='Model base name')
    parser.add_argument('--preload', type=str, default=None, help='Preload model path')
    parser.add_argument('--tokenizer_file', type=str, default='tokenizer_{0}.json', help='Tokenizer file pattern')
    parser.add_argument('--experiment_name', type=str, default='runs', help='Experiment name')
    
    args = parser.parse_args()
    return args

def get_config():
    # Use the parsed command-line arguments
    args = parse_args()
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'seq_len': args.seq_len,
        'd_model': args.d_model,
        'lang_src': args.lang_src,
        'lang_tgt': args.lang_tgt,
        'model_folder': args.model_folder,
        'model_basename': args.model_basename,
        'preload': args.preload,
        'tokenizer_file': args.tokenizer_file,
        'experiment_name': args.experiment_name,
    }
    return config

def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path(model_folder) / model_filename)

# Example of how to use it in a script
if __name__ == "__main__":
    config = get_config()
    epoch = '01'  # You might also want to parse this from the command-line arguments
    weights_file_path = get_weights_file_path(config, epoch)
    print(f'Weights file path: {weights_file_path}')
