import argparse
import torch
from model import LightMedVLM
from huggingface_hub import snapshot_download
from lightning.fabric.utilities.data import AttributeDict
torch.serialization.add_safe_globals([AttributeDict])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LightMedVLM inference on medical images"
    )

    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="One or more image paths (space-separated)"
    )

    parser.add_argument(
        "--vision_model",
        type=str,
        default="microsoft/swin-base-patch4-window7-224",
        help="Vision encoder model name"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_ckpt.pth",
        help="Absolute path to the .pth checkpoint file"
    )


    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="LLM model name"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default="huyhoangt2201/lightmedvlm-base",
        help="HuggingFace repo ID for model weights"
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        default="lightmedvlm",
        help="Local directory to store downloaded model files"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Download model files if not already present
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )

    model_args = {
        "vision_model": args.vision_model,
        "llm_model": args.llm_model,
    }

    model = LightMedVLM.load_from_checkpoint(
        args.ckpt,
        strict=False,
        **model_args
    )

    print(f"Generating report for images:")
    for img in args.images:
        print(f" - {img}")
    print()

    report = model.inference_report(args.images)

    print("=" * 60)
    print("GENERATED REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
