import argparse
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="模型微调脚本")
parser.add_argument("--mode", choices=["ptuning_v1", "ptuning_v2", "lora"], required=True, help="微调方式")
parser.add_argument("--model_path", required=True, help="模型路径")
parser.add_argument("--data_path", required=True, help="数据路径")
parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
parser.add_argument("--output_dir", default=None, help="模型输出目录")

args = parser.parse_args()

logging.debug(f"训练参数: mode={args.mode}, epochs={args.epochs}, data_path={args.data_path}, model_path={args.model_path}")

allowed_models = ["Qwen", "DeepSeek"]
if not any(model in args.model_path for model in allowed_models):
    raise ValueError(f"仅支持 {allowed_models} 系列模型")

def run(args):
    # 根据模式选择不同的训练脚本
    if args.mode == "ptuning_v1":
        from scripts.train_ptuning_v1 import run as run_ptuning_v1
        run_ptuning_v1(args)
    elif args.mode == "ptuning_v2":
        from scripts.train_ptuning_v2 import run as run_ptuning_v2
        run_ptuning_v2(args)
    elif args.mode == "lora":
        from scripts.train_lora import run as run_lora
        run_lora(args)
    else:
        raise ValueError(f"不支持的训练模式: {args.mode}")

if __name__ == "__main__":
    run(args)