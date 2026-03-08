import os
import sys
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models import get_decom, load_spam_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default="output.png", help="output image path")
    parser.add_argument("--decom_path", type=str, default="checkpoints/decom.pth", help="decomposition model path")
    parser.add_argument("--spam_path", type=str, default="checkpoints/spam.pth", help="SPAM model path")
    args = parser.parse_args()

    print(f"loading models on {device}...")

    # load models
    try:
        model_decom = get_decom(args.decom_path)
        model_decom = model_decom.to(device)
        model_decom.eval()
        print("decomposition model loaded")
    except Exception as e:
        print(f"failed to load decomposition model: {e}")
        return

    try:
        model_spam = load_spam_model(args.spam_path, device)
        print("SPAM model loaded")
    except Exception as e:
        print(f"failed to load SPAM model: {e}")
        return

    # load input image
    print(f"processing {args.input}...")
    img = Image.open(args.input).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # process
    with torch.no_grad():
        # decomposition
        R, L = model_decom(img_tensor)

        # simple enhancement (can be replaced with proper models)
        R_enhanced = torch.clamp(R * 1.2, 0, 1)
        L_enhanced = torch.clamp(L * 1.5, 0, 1)

        # combine
        combined = R_enhanced * L_enhanced

        # SPAM post-processing
        output = model_spam(combined)

    # save output
    output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(output_np).save(args.output)

    print(f"saved result to {args.output}")

if __name__ == "__main__":
    main()
