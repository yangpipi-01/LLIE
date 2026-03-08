import torch.nn as nn
import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def np_save_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img*255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def np_save_illum_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().detach().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img*255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def load_decom(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path,map_location=device)
        model.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # freeze the params of Decomposition Model
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L

def get_decom(decom_path):
    model_Decom_low = Decom()
    model_Decom_low = load_decom(model_Decom_low, decom_path)
    return model_Decom_low

if __name__ == "__main__":
    low_img_path ="../data/test/179.png"
    decom_path = "../ckpt/decom/decom.pth"
    model_Decom_low = get_decom(decom_path)

    transform = transforms.Compose([transforms.ToTensor()])
    low_img = transform(Image.open(low_img_path)).unsqueeze(0)
    R, L = model_Decom_low(low_img)
    print(R.shape)   # torch.Size([1, 3, 400, 600])
    print(L.shape)   # torch.Size([1, 1, 400, 600])
    np_save_TensorImg(R, "../result/decom/R_high.png")
    np_save_TensorImg(L, "../result/decom/L_high.png")
    np_save_TensorImg(R*L, "../result/decom/all_high.png")
