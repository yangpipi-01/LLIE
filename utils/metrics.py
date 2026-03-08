import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import niqe

def calculate_psnr(img1, img2):
    """calculate PSNR between two images"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """calculate SSIM between two images"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)

def calculate_niqe(img):
    """calculate NIQE for an image"""
    return niqe(img.astype(np.float32))

def evaluate_image(pred, gt):
    """evaluate single image prediction against ground truth"""
    metrics = {
        'PSNR': calculate_psnr(pred, gt),
        'SSIM': calculate_ssim(pred, gt),
        'NIQE': calculate_niqe(pred)
    }
    return metrics
