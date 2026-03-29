import argparse
import logging
import warnings 
import torch 
import open_clip
import utils
from datasets import load_dataset
from model import create_model_and_optimizer
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='multimodal')
    parser.add_argument('--dataset_path', default='./my_dataset/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--clip_lr', type=float, default=1e-6)
    parser.add_argument('--backbone', type=str, default='ViT-B-32')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    model_paths = {
        "ViT-B-32": "CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
    }

    utils.set_seed(args.seed)
    log_folder_path = utils.get_log(args.experiment_name)
    logging.info(f"Log path: {log_folder_path}")
    logging.info('Arguments:')
    for key, value in vars(args).items():
        logging.info(f'    {key}: {value}')

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=model_paths[args.backbone]
    )
    clip_model = clip_model.float()
    tokenizer = open_clip.get_tokenizer(args.backbone)
    trainset = load_dataset(args.dataset_path, [preprocess_train, preprocess_val])

    model, optimizer = create_model_and_optimizer(
        clip_model,
        tokenizer,
        args.clip_lr,
        args.weight_decay,
        args.tau,
        device
    )
    utils.train_and_evaluate(model, optimizer, trainset, args, device)


if __name__ == '__main__':
    main()