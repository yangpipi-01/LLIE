import os
import sys
import glob
import time
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.decom import get_decom, np_save_TensorImg
from models.spam import load_spam_model

# try to import flow model for L component processing
try:
    # import flow processing from LLFlow
    llflow_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'LLFlow-main', 'code')
    sys.path.insert(0, llflow_path)
    from test import load_model, predict
    from test import t, rgb
    FLOW_AVAILABLE = True
except:
    print("LLFlow not available, using simple L component processing")
    FLOW_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_all_models(decom_path, r_enhance_path, l_enhance_path, spam_path):
    # load decomposition model
    print("loading decomposition model...")
    model_decom = get_decom(decom_path)
    model_decom = model_decom.to(device)
    model_decom.eval()

    # load R component enhancement model (diffusion)
    print("loading R enhancement model...")
    # load R enhancement model based on original code
    try:
        # import diffusion model
        diffusion_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'DF')
        sys.path.insert(0, diffusion_path)
        from models.ddm import load_diffusion_model
        model_r_enhance = load_diffusion_model(r_enhance_path)
        model_r_enhance = model_r_enhance.to(device)
        model_r_enhance.eval()
    except:
        print("R enhancement model loading failed, using identity")
        model_r_enhance = None

    # load L component enhancement model (flow)
    print("loading L enhancement model...")
    if FLOW_AVAILABLE:
        try:
            model_l_enhance, opt_l = load_model(l_enhance_path)
            model_l_enhance.netG = model_l_enhance.netG.cuda()
        except:
            print("L flow enhancement model loading failed, using identity")
            model_l_enhance = None
    else:
        model_l_enhance = None

    # load SPAM model for post-processing
    print("loading SPAM model...")
    model_spam = load_spam_model(spam_path, device)

    return model_decom, model_r_enhance, model_l_enhance, model_spam

def enhance_image(model_decom, model_r_enhance, model_l_enhance, model_spam, img_tensor):
    with torch.no_grad():
        # step 1: Retinex decomposition
        R, L = model_decom(img_tensor)

        # step 2: enhance R component
        if model_r_enhance is not None:
            R_enhanced = model_r_enhance(R)
        else:
            R_enhanced = R

        # step 3: enhance L component
        if model_l_enhance is not None:
            # convert to numpy for flow model
            L_np = L.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            L_enhanced_np = predict(model_l_enhance, L_np)
            L_enhanced = torch.from_numpy(L_enhanced_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        else:
            L_enhanced = L

        # step 4: combine enhanced R and L
        combined = R_enhanced * L_enhanced

        # step 5: SPAM post-processing
        final_output = model_spam(combined)

        return final_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path or directory")
    parser.add_argument("--output", type=str, default="results", help="output directory")
    parser.add_argument("--decom_path", type=str, default="checkpoints/decom.pth", help="decomposition model path")
    parser.add_argument("--r_enhance_path", type=str, default="checkpoints/r_enhance.pth", help="R enhancement model path")
    parser.add_argument("--l_enhance_path", type=str, default="../LLFlow-main/code/confs/LOL_smallNet.yml", help="L enhancement config path")
    parser.add_argument("--spam_path", type=str, default="checkpoints/spam.pth", help="SPAM model path")
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output, exist_ok=True)

    # load all models
    model_decom, model_r_enhance, model_l_enhance, model_spam = load_all_models(
        args.decom_path, args.r_enhance_path, args.l_enhance_path, args.spam_path
    )

    # get input images
    if os.path.isfile(args.input):
        img_paths = [args.input]
    else:
        img_paths = glob.glob(os.path.join(args.input, "*.png"))
        img_paths += glob.glob(os.path.join(args.input, "*.jpg"))

    print(f"found {len(img_paths)} images")

    # process images
    for img_path in img_paths:
        print(f"processing {img_path}")

        # load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # enhance image
        start_time = time.time()
        output = enhance_image(model_decom, model_r_enhance, model_l_enhance, model_spam, img_tensor)
        elapsed = time.time() - start_time

        # save result
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output, f"{img_name}_enhanced.png")

        output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        Image.fromarray(output_np).save(output_path)

        print(f"saved to {output_path}, time: {elapsed:.2f}s")

    print("processing complete!")

if __name__ == "__main__":
    main()
