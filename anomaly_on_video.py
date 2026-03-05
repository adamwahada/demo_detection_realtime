import cv2
import os
import sys
import torch
import numpy as np
import argparse
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from torch import tensor
from torchvision import transforms
from collections import defaultdict
from ultralytics import YOLO

from efficientad import predict as effpredict
from helpers import letterbox_image

def get_args():
    parser = argparse.ArgumentParser(description="YOLO & EfficientAD Anomaly Detection")

    parser.add_argument('--input', type=str, default="/data/datasets/Farine_Khomsa/testAnomalie-Trim2.mp4", help="Path to input video")

    parser.add_argument('--output_dir', type=str, default="./output/videos/", help="Output directory for video")
    parser.add_argument('--yolo_model', type=str, default='/data/checkpoints/yolo26m_seg_farine_FV_v2.pt')
    parser.add_argument('--ad_teacher', type=str, default='/data/checkpoints/EfficientAD/khomsa-ad-v0/teacher_best.pth')
    parser.add_argument('--ad_student', type=str, default='/data/checkpoints/EfficientAD/khomsa-ad-v0/student_best.pth')
    parser.add_argument('--ad_ae', type=str, default='/data/checkpoints/EfficientAD/khomsa-ad-v0/autoencoder_best.pth')

    parser.add_argument('--yolo_imgsz', type=int, default=1280, help="YOLO image size")
    parser.add_argument('--yolo_conf', type=float, default=0.5, help="YOLO confidence")
    parser.add_argument('--yolo_tracker', type=str, default="bytetrack.yaml", help="YOLO tracker")

    parser.add_argument('--ad_thresh', type=float, default=5000.0, help="Anomaly score threshold")
    parser.add_argument('--ad_imgsz', type=int, default=256, help="Anomaly image size")
    parser.add_argument('--ad_strategy', type=str, default="MAJORITY", choices=["OR", "MAJORITY"])
    
    parser.add_argument('--refine_mask', action='store_true', help="Enable mask smoothing/erosion")
    parser.add_argument('--margin_pct', type=float, default=0.1, help="Crop margin percentage")
    parser.add_argument('--erosion_size', type=int, default=3, help="Erosion kernel size")

    parser.add_argument('--zone_start_pct', type=float, default=0.20, help="Start detection at this % of image width (0.0 - 1.0)")
    parser.add_argument('--zone_end_pct', type=float, default=0.60, help="Lock decision at this % of image width (0.0 - 1.0)")
    
    return parser.parse_args()

def get_ad_constants(device):
    # NOTE: EfficientAD precomputed teacher_mean, teacher_std and quantiles from training run 1; to later save somewhere else
    teacher_mean = tensor([[[[-5.1363e-02]], [[ 1.7760e+00]], [[-2.4545e-01]], [[-4.9816e-01]], [[ 1.5313e+00]], [[ 3.1152e-01]], [[ 6.0188e-01]], [[ 3.6512e-01]], [[ 5.1304e-02]], [[-4.2684e-01]], [[ 1.1894e-01]], [[-4.1604e-01]], [[-5.1443e-01]], [[-1.2203e+00]], [[-2.9167e-01]], [[ 5.0720e-02]], [[ 2.5826e-01]], [[-3.7486e-01]], [[ 2.3602e-02]], [[ 6.5434e-02]], [[-5.0326e-02]], [[-2.5824e-01]], [[-2.5790e-01]], [[-4.0658e-01]], [[ 4.7416e-01]], [[-7.9013e-01]], [[ 1.8196e-01]], [[ 7.6295e-01]], [[ 2.0669e-01]], [[-3.9837e-01]], [[ 6.0947e-01]], [[-2.5171e-01]], [[ 6.5435e-01]], [[-5.3981e-01]], [[ 1.2066e-01]], [[-1.7396e-01]], [[ 5.6246e-01]], [[ 5.5591e-01]], [[-5.1715e-01]], [[ 1.0987e+00]], [[ 9.3702e-01]], [[ 4.4015e-01]], [[-8.3180e-01]], [[ 6.2347e-01]], [[-1.3720e-01]], [[ 8.5777e-01]], [[-1.0072e-01]], [[-9.0407e-01]], [[-3.3021e-02]], [[-2.3091e-01]], [[-3.2188e-01]], [[-5.9730e-01]], [[-4.5015e-01]], [[-6.0188e-01]], [[-7.1397e-01]], [[-9.0748e-01]], [[-5.4010e-01]], [[ 4.2432e-02]], [[-6.3098e-01]], [[-9.9702e-01]], [[-1.4206e-01]], [[ 4.9262e-01]], [[-4.4754e-01]], [[ 3.6046e-01]], [[ 4.5837e-01]], [[-3.2684e-01]], [[-8.7544e-01]], [[ 9.0098e-01]], [[ 2.0489e-02]], [[ 2.6504e-01]], [[-4.0550e-01]], [[-2.8311e-01]], [[-4.2792e-01]], [[-2.3451e-01]], [[-5.8289e-01]], [[ 7.0677e-01]], [[ 4.3928e-02]], [[ 1.0353e+00]], [[ 1.1956e+00]], [[ 6.3407e-01]], [[-2.5607e-01]], [[ 6.4021e-01]], [[ 2.4215e-01]], [[ 2.1571e-01]], [[-5.8828e-01]], [[-1.4923e-01]], [[-6.0643e-01]], [[ 2.6132e-02]], [[-7.1472e-01]], [[-1.0984e+00]], [[-1.9217e-02]], [[-2.7370e-01]], [[-8.7377e-01]], [[ 1.3038e+00]], [[-2.3501e-01]], [[ 1.3838e+00]], [[ 1.1510e+00]], [[ 1.2212e-01]], [[-2.5279e-01]], [[ 6.3292e-01]], [[ 3.4613e-01]], [[-2.9578e-01]], [[ 2.8015e-01]], [[ 1.2522e+00]], [[ 3.6096e-01]], [[-2.7012e-01]], [[ 7.0600e-02]], [[-6.1828e-02]], [[ 1.4512e-01]], [[ 1.5657e+00]], [[ 4.9503e-01]], [[-4.6318e-01]], [[-5.8731e-01]], [[ 1.1112e+00]], [[ 1.1093e-01]], [[ 1.4196e+00]], [[-4.3614e-01]], [[-1.0489e+00]], [[ 5.7934e-01]], [[-2.7431e-01]], [[-3.5358e-01]], [[ 7.1722e-01]], [[ 4.2237e-01]], [[ 1.1514e+00]], [[-1.5490e-01]], [[ 1.4055e+00]], [[-4.7353e-01]], [[ 1.1065e-01]], [[-2.2599e-01]], [[-7.8084e-01]], [[ 1.2447e-01]], [[ 1.7425e-02]], [[-1.0356e-01]], [[ 1.0497e+00]], [[-2.6242e-01]], [[ 1.6211e+00]], [[ 1.3975e+00]], [[-5.3413e-01]], [[-1.1583e-01]], [[-1.8033e-01]], [[ 1.9506e-01]], [[ 1.2611e+00]], [[-1.6965e-01]], [[-1.7216e-01]], [[ 6.4230e-02]], [[ 1.2701e-01]], [[-6.8662e-01]], [[ 1.2005e+00]], [[ 3.0258e-01]], [[-4.9366e-01]], [[-1.6557e-01]], [[-3.3752e-01]], [[ 2.4166e-01]], [[-3.0904e-01]], [[-2.3715e-01]], [[-6.9151e-02]], [[-5.3396e-01]], [[-1.4155e-01]], [[ 2.4736e-01]], [[-4.5878e-01]], [[-5.5535e-01]], [[-5.5484e-02]], [[-1.8567e-02]], [[-8.5309e-01]], [[ 6.5638e-01]], [[-8.7676e-01]], [[-2.3930e-01]], [[ 4.1847e-01]], [[ 5.4393e-01]], [[-5.2045e-01]], [[ 2.5669e-02]], [[-4.5409e-01]], [[-3.8863e-01]], [[ 1.3316e+00]], [[ 1.6209e-02]], [[-4.4136e-01]], [[-6.7502e-01]], [[ 2.0200e-01]], [[-8.2756e-01]], [[ 5.7160e-01]], [[-4.7096e-01]], [[ 1.0038e+00]], [[-5.6045e-01]], [[-5.8645e-01]], [[-3.9122e-01]], [[-6.3427e-01]], [[ 1.3840e-01]], [[-8.8798e-01]], [[-5.3357e-01]], [[-2.6915e-02]], [[ 6.3786e-01]], [[-7.9083e-01]], [[ 2.0343e-01]], [[ 1.0612e-01]], [[-2.1644e-01]], [[ 2.1468e-01]], [[-1.9903e-01]], [[-2.6494e-02]], [[ 1.5634e-01]], [[ 9.0139e-02]], [[ 1.7848e-01]], [[-1.9109e-01]], [[-1.7605e-01]], [[-2.4558e-01]], [[ 6.5429e-02]], [[ 2.0649e-01]], [[ 6.1663e-01]], [[-3.5338e-01]], [[-1.5367e-01]], [[ 3.6014e-02]], [[-2.0263e-01]], [[-5.1552e-02]], [[-3.9506e-02]], [[ 4.5415e-01]], [[ 1.7180e-01]], [[ 4.6381e-01]], [[ 4.0441e-01]], [[-5.7452e-02]], [[-3.2698e-01]], [[ 1.3647e-01]], [[ 6.5655e-02]], [[-1.6518e-01]], [[-9.1803e-02]], [[-2.3841e-01]], [[ 2.0825e-01]], [[-4.1887e-01]], [[ 1.8100e-01]], [[-1.4283e-01]], [[ 1.1517e-01]], [[-1.5492e-02]], [[-1.8953e-01]], [[ 1.5802e-01]], [[-3.7760e-01]], [[ 1.5275e-01]], [[ 9.1592e-02]], [[-2.6449e-01]], [[-8.5420e-02]], [[ 2.1913e-02]], [[-2.8792e-01]], [[-4.4601e-01]], [[ 8.7716e-02]], [[-7.4726e-02]], [[ 5.0872e-01]], [[-2.0931e-01]], [[-1.6279e-01]], [[-2.1468e-01]], [[ 3.8307e-01]], [[ 3.0814e-04]], [[ 1.0893e-01]], [[-3.2548e-01]], [[ 1.6866e-01]], [[-8.5367e-02]], [[ 1.8354e-01]], [[-4.8861e-02]], [[ 1.0512e-01]], [[-4.7072e-01]], [[-1.6218e-01]], [[-5.7795e-01]], [[ 2.5325e-01]], [[-1.3043e-01]], [[-3.1296e-02]], [[-2.3530e-02]], [[-2.2087e-01]], [[-9.3874e-02]], [[ 3.0182e-03]], [[-1.9577e-01]], [[-5.7833e-02]], [[-4.1544e-01]], [[-5.1366e-01]], [[-2.0854e-01]], [[-9.1510e-02]], [[ 2.3724e-01]], [[ 1.3237e-01]], [[-2.8528e-01]], [[ 3.7264e-01]], [[ 2.0809e-01]], [[-4.6088e-02]], [[-2.7879e-01]], [[ 4.9456e-02]], [[-1.0511e-01]], [[-2.5724e-01]], [[-2.4669e-01]], [[-2.3485e-01]], [[-3.7485e-01]], [[-1.9134e-01]], [[-1.1587e-01]], [[-3.8640e-01]], [[ 6.2040e-02]], [[-3.0930e-01]], [[-1.5093e-01]], [[-7.8048e-02]], [[-7.5478e-02]], [[-1.5802e-01]], [[-2.9285e-01]], [[ 6.8601e-01]], [[ 1.8382e-01]], [[ 2.9089e-01]], [[ 4.4897e-01]], [[ 1.6217e-01]], [[ 2.4949e-01]], [[-2.6229e-01]], [[-1.8600e-01]], [[-1.1557e-01]], [[ 7.8778e-01]], [[-6.3277e-02]], [[-3.8674e-01]], [[-3.4525e-01]], [[-4.1785e-01]], [[-7.9131e-02]], [[-5.9789e-01]], [[-2.9240e-01]], [[-2.0488e-01]], [[-2.4908e-01]], [[ 5.2083e-03]], [[ 2.2822e-01]], [[ 3.7187e-01]], [[-8.0793e-02]], [[-6.7274e-02]], [[-6.4008e-01]], [[-1.9308e-01]], [[-5.9406e-01]], [[-1.8197e-01]], [[-1.7009e-01]], [[-4.1618e-01]], [[-2.9832e-01]], [[-9.7684e-03]], [[ 2.2376e-01]], [[-4.6695e-01]], [[-1.8582e-02]], [[-1.3330e-01]], [[-1.8362e-01]], [[ 9.0170e-02]], [[ 4.9270e-01]], [[-2.6746e-01]], [[-5.1084e-01]], [[-1.9661e-01]], [[ 1.4029e-01]], [[ 4.8134e-02]], [[-3.7430e-02]], [[-3.0449e-01]], [[-1.0045e-01]], [[-2.0338e-01]], [[-5.5641e-01]], [[-4.2754e-01]], [[ 9.4283e-02]], [[-1.7388e-01]], [[ 2.1757e-01]], [[-1.8555e-01]], [[-2.8787e-01]], [[-2.4350e-01]], [[ 1.4595e-01]], [[-5.3691e-02]], [[-4.0674e-01]], [[-1.0118e-01]], [[-3.9949e-01]], [[-4.9209e-02]], [[-2.3909e-01]], [[-1.7490e-02]], [[-4.3783e-01]], [[-2.7591e-01]], [[-3.2150e-01]], [[-2.1542e-01]], [[ 8.0212e-02]], [[-1.4593e-01]], [[-1.6407e-01]], [[ 2.5814e-01]], [[-4.6419e-02]], [[ 1.4318e-01]], [[ 1.3753e-02]], [[-2.4695e-01]], [[ 1.7561e-01]], [[-2.5793e-01]], [[ 9.4256e-02]], [[-1.0280e-01]], [[-1.1936e-01]], [[-1.9038e-01]], [[-1.2021e-01]], [[ 1.3856e-01]], [[-1.6114e-01]], [[ 4.3487e-01]], [[ 4.9672e-01]], [[ 7.2060e-01]], [[-1.0900e-01]], [[-3.6739e-01]]]], device=device)
    teacher_std = tensor([[[[0.5239]], [[1.7099]], [[1.1781]], [[0.5201]], [[1.3881]], [[0.7909]], [[0.7830]], [[0.7774]], [[0.7463]], [[0.6269]], [[0.5206]], [[0.8670]], [[1.0329]], [[0.8651]], [[0.5772]], [[0.7147]], [[0.5641]], [[0.6617]], [[0.7652]], [[0.5744]], [[0.5347]], [[0.6492]], [[0.7748]], [[0.5672]], [[0.8560]], [[0.8794]], [[0.7888]], [[1.0368]], [[0.5493]], [[0.6766]], [[0.5687]], [[0.5205]], [[1.0236]], [[0.4699]], [[0.8026]], [[0.6527]], [[0.7836]], [[0.5780]], [[0.9568]], [[0.6846]], [[1.0674]], [[0.4754]], [[0.6732]], [[0.8590]], [[0.6051]], [[0.9164]], [[0.5593]], [[0.8501]], [[0.5229]], [[0.5472]], [[0.9782]], [[0.4325]], [[0.6951]], [[0.3345]], [[0.7716]], [[1.0568]], [[1.0392]], [[0.5794]], [[0.6717]], [[0.7485]], [[0.8134]], [[0.9115]], [[0.6237]], [[0.4603]], [[0.7627]], [[0.7226]], [[0.4053]], [[1.0667]], [[0.7937]], [[0.5888]], [[0.5236]], [[0.3564]], [[0.6023]], [[1.1231]], [[0.6317]], [[1.1716]], [[0.6431]], [[0.8499]], [[0.8374]], [[0.5577]], [[1.0245]], [[0.4850]], [[0.5876]], [[0.3597]], [[0.5522]], [[0.7427]], [[0.8091]], [[0.6497]], [[0.6687]], [[0.3412]], [[0.6591]], [[0.4553]], [[0.8642]], [[1.0125]], [[0.7663]], [[0.9187]], [[1.2466]], [[0.6146]], [[0.5898]], [[1.0309]], [[0.7090]], [[0.7905]], [[0.9073]], [[1.6798]], [[0.7963]], [[0.4698]], [[0.4962]], [[0.3831]], [[0.6631]], [[1.4975]], [[0.8146]], [[0.7185]], [[0.8982]], [[1.1086]], [[1.1288]], [[1.6333]], [[0.5831]], [[0.8202]], [[1.4793]], [[0.7431]], [[0.6172]], [[0.9800]], [[0.8063]], [[0.7428]], [[0.5318]], [[1.1341]], [[0.5433]], [[0.6969]], [[0.4878]], [[0.9681]], [[1.7680]], [[0.5253]], [[0.5268]], [[1.0664]], [[0.7137]], [[1.0936]], [[1.1628]], [[0.3559]], [[0.6900]], [[0.4400]], [[0.6801]], [[1.6017]], [[0.5572]], [[0.5120]], [[0.4872]], [[0.7920]], [[0.8694]], [[1.3107]], [[0.7773]], [[0.5354]], [[0.6581]], [[0.7697]], [[0.7410]], [[1.0121]], [[0.7566]], [[0.6325]], [[0.8130]], [[0.8734]], [[0.8519]], [[0.5506]], [[0.6268]], [[0.7554]], [[0.4658]], [[1.0158]], [[0.6850]], [[0.3668]], [[1.0021]], [[0.9711]], [[1.2045]], [[0.7808]], [[0.8128]], [[0.7182]], [[0.5152]], [[1.0998]], [[0.6270]], [[0.7888]], [[0.8740]], [[0.6980]], [[1.1965]], [[0.8548]], [[0.8449]], [[0.9829]], [[0.9319]], [[0.4343]], [[0.6292]], [[0.5753]], [[0.6605]], [[0.6896]], [[0.4461]], [[0.8729]], [[0.8763]], [[0.7213]], [[0.3009]], [[0.2828]], [[0.3933]], [[0.6549]], [[0.5356]], [[0.3494]], [[0.4413]], [[0.2644]], [[0.2628]], [[0.3306]], [[0.5734]], [[0.2240]], [[0.5362]], [[0.2586]], [[0.9917]], [[0.7133]], [[0.6453]], [[0.3180]], [[0.4017]], [[0.5878]], [[0.3601]], [[0.4815]], [[0.3112]], [[0.1719]], [[0.4312]], [[0.3760]], [[0.4344]], [[1.1007]], [[0.7569]], [[0.5289]], [[0.5694]], [[0.4710]], [[0.2453]], [[0.4443]], [[0.3240]], [[0.3954]], [[0.3854]], [[0.3561]], [[0.7183]], [[0.5552]], [[0.5400]], [[0.6592]], [[0.5554]], [[0.5251]], [[0.3531]], [[0.4248]], [[0.5976]], [[0.4228]], [[0.7262]], [[0.4144]], [[0.3434]], [[0.5469]], [[0.6303]], [[0.3852]], [[0.6718]], [[0.3195]], [[0.5141]], [[0.6718]], [[0.4618]], [[0.5661]], [[0.4426]], [[0.3891]], [[0.6596]], [[0.3094]], [[0.3311]], [[0.4337]], [[0.5812]], [[0.3699]], [[0.4671]], [[0.6759]], [[0.9921]], [[0.6204]], [[0.3958]], [[0.3468]], [[0.5505]], [[0.5954]], [[0.4540]], [[0.5419]], [[0.4154]], [[0.2791]], [[0.3647]], [[0.6823]], [[0.6050]], [[0.4158]], [[0.5041]], [[0.6709]], [[0.3662]], [[0.3929]], [[0.4809]], [[1.0397]], [[0.4083]], [[0.6303]], [[0.5705]], [[0.3377]], [[0.2751]], [[0.5305]], [[0.5875]], [[0.8373]], [[0.2631]], [[0.6016]], [[0.4963]], [[0.3352]], [[0.4260]], [[0.4538]], [[0.7235]], [[1.4305]], [[0.5420]], [[0.4355]], [[0.2538]], [[0.3862]], [[0.5087]], [[0.9754]], [[1.0311]], [[0.3855]], [[0.3532]], [[0.6664]], [[0.5376]], [[0.5028]], [[0.6151]], [[0.3041]], [[0.3937]], [[0.5465]], [[0.2593]], [[0.3919]], [[0.5044]], [[0.3048]], [[0.6256]], [[0.5799]], [[0.4901]], [[0.3634]], [[0.3817]], [[0.3364]], [[0.4106]], [[0.5272]], [[0.5996]], [[0.8464]], [[0.4032]], [[0.3356]], [[0.6789]], [[0.4265]], [[1.0305]], [[0.5309]], [[0.6397]], [[0.4897]], [[0.2707]], [[0.3724]], [[0.4369]], [[0.4454]], [[0.3789]], [[0.5349]], [[0.5651]], [[0.6743]], [[0.2720]], [[0.4948]], [[0.5130]], [[0.5182]], [[0.4310]], [[0.2850]], [[0.3351]], [[0.3272]], [[0.5462]], [[0.7595]], [[0.4656]], [[0.3075]], [[0.6429]], [[0.5781]], [[0.3836]], [[0.6243]], [[0.5970]], [[0.5477]], [[0.4419]], [[0.2318]], [[0.4326]], [[0.3210]], [[0.3647]], [[0.4317]], [[0.3083]], [[0.5583]], [[0.4220]], [[0.3467]], [[0.6323]], [[0.4236]], [[0.3297]], [[0.3068]], [[0.3168]], [[0.4089]], [[0.3270]], [[0.4408]], [[0.5791]], [[0.4958]], [[0.3135]], [[0.4144]]]], device=device)
    
    quantiles = {
        "q_st_start": 0.2566621005535126,
        "q_st_end": 0.2566621005535126,
        "q_ae_start": 0.04747752845287323,
        "q_ae_end": 0.12276165932416916
    }
    return teacher_mean, teacher_std, quantiles

class AnomalyPipeline:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Loading Models...")
        self.yolo = YOLO(args.yolo_model)
        self.teacher = torch.load(args.ad_teacher, map_location=self.device, weights_only=False).eval()
        self.student = torch.load(args.ad_student, map_location=self.device, weights_only=False).eval()
        self.autoencoder = torch.load(args.ad_ae, map_location=self.device, weights_only=False).eval()
        
        self.t_mean, self.t_std, self.qs = get_ad_constants(self.device)
        
        self.line_start_px = 0
        self.line_end_px = 0

        self.transform = transforms.Compose([
            transforms.Resize((self.args.ad_imgsz, self.args.ad_imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.track_states = defaultdict(lambda: {'results': [], 'decision': None})
        os.makedirs(args.output_dir, exist_ok=True)
        plt.switch_backend('Agg')

    def refine_mask(self, mask_crop):
        """Smooths edges and erodes background"""
        mask_blurred = cv2.GaussianBlur(mask_crop, (5, 5), 0)
        _, mask_final = cv2.threshold(mask_blurred, 0.5, 1, cv2.THRESH_BINARY)
        mask_final = mask_final.astype(np.uint8)
        
        if self.args.erosion_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.erosion_size, self.args.erosion_size))
            mask_final = cv2.erode(mask_final, kernel, iterations=1)
        return mask_final

    def crop_and_mask_object(self, frame, mask_raw):
        """Preprocess instances for EfficientAD: crop, apply margin, blackout background and letterbox image"""
        h_f, w_f = frame.shape[:2]
        mask_full = cv2.resize(mask_raw, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
        _, mask_full = cv2.threshold(mask_full, 0.5, 1, cv2.THRESH_BINARY)
        
        rows, cols = np.where(mask_full > 0)
        if len(rows) == 0: return None
        x1, y1, x2, y2 = np.min(cols), np.min(rows), np.max(cols), np.max(rows)
        
        w_box, h_box = x2 - x1, y2 - y1
        m_x, m_y = int(w_box * self.args.margin_pct), int(h_box * self.args.margin_pct)
        
        cx1, cy1 = max(0, x1 - m_x), max(0, y1 - m_y)
        cx2, cy2 = min(w_f, x2 + m_x), min(h_f, y2 + m_y)
        
        img_crop = frame[cy1:cy2, cx1:cx2].copy()
        mask_crop = mask_full[cy1:cy2, cx1:cx2].copy()
        
        if self.args.refine_mask:
            mask_final = self.refine_mask(mask_crop)
        else:
            mask_final = mask_crop.astype(np.uint8)
            
        img_crop[mask_final == 0] = 0
        
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        return letterbox_image(img_rgb)

    def detect_anomaly(self, img_crop, track_id):
        orig_w, orig_h = img_crop.size
        img_tensor = self.transform(img_crop).unsqueeze(0).to(self.device)

        map_combined, _, _ = effpredict(
            image=img_tensor, teacher=self.teacher, student=self.student,
            autoencoder=self.autoencoder, teacher_mean=self.t_mean,
            teacher_std=self.t_std, q_st_start=self.qs['q_st_start'],
            q_st_end=self.qs['q_st_end'], q_ae_start=self.qs['q_ae_start'], q_ae_end=self.qs['q_ae_end']
        )

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_h, orig_w), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        score = np.max(map_combined)
        return score > self.args.ad_thresh, score

    def process_video(self):
        cap = cv2.VideoCapture(self.args.input)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out_path = os.path.join(self.args.output_dir, "temp_output.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        self.line_start_px = int(w * self.args.zone_start_pct)
        self.line_end_px = int(w * self.args.zone_end_pct)

        print(f"Resolution: {w}x{h} | Zone: {self.line_start_px}px to {self.line_end_px}px")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.yolo.track(frame, persist=True, conf=self.args.yolo_conf, tracker=self.args.yolo_tracker, 
                                    retina_masks=True, imgsz=self.args.yolo_imgsz, verbose=False)
            
            cv2.line(frame, (self.line_start_px, 0), (self.line_start_px, h), (255, 0, 255), 2)
            cv2.line(frame, (self.line_end_px, 0), (self.line_end_px, h), (255, 0, 255), 2)

            if results[0].masks is not None and results[0].boxes.id is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    state = self.track_states[track_id]
                    x1, y1, x2, y2 = map(int, boxes[i])
                    center_x = (x1 + x2) // 2
                    
                    label = "WAITING"
                    color = (200, 200, 200)

                    # Before Zone: skip
                    if center_x > self.line_end_px:
                        label = f"ID:{track_id} ENTERING"
                    
                    # Inside Zone: segment and detect anomalies
                    elif (self.line_start_px <= center_x <= self.line_end_px) and (state['decision'] is None):
                        img_crop = self.crop_and_mask_object(frame, masks[i])
                        if img_crop is not None:
                            is_def, score = self.detect_anomaly(Image.fromarray(img_crop), track_id)
                            state['results'].append(is_def)
                            label = f"ID:{track_id} SCANNING..."
                            color = (0, 255, 255)

                    # Exit Zone: lock decision following a strategy (OR, MAJORITY, etc.)
                    else:
                        if state['decision'] is None:
                            state['decision'] = self.get_final_decision(state['results'], strategy=self.args.ad_strategy)
                        
                        is_def = state['decision']
                        label = f"ID:{track_id} {'DEFECTIVE' if is_def else 'GOOD'}"
                        color = (0, 0, 255) if is_def else (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(frame)
        
        cap.release()
        writer.release()
        self.compress_video(out_path)

    def crop_object(self, frame, mask_raw):
        h, w = frame.shape[:2]
        mask = cv2.resize(mask_raw, (w, h))
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        rows, cols = np.where(mask > 0)
        if len(rows) == 0: return None
        
        y1, x1, y2, x2 = np.min(rows), np.min(cols), np.max(rows), np.max(cols)
        crop = frame[y1:y2, x1:x2].copy()

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def get_final_decision(self, results, strategy="MAJORITY"):
        if not results:
            return False
        
        match strategy:
            case "OR":
                return any(results)
            case "MAJORITY":
                return sum(results) > (len(results) / 2)
            case _: # NOTE: add average score strategy
                raise(ValueError("Unsupported decision strategy"))
    
        return False

    def compress_video(self, path):
        final_path = path.replace("temp_", "")
        print(f"Compressing to H.265...")
        cmd = ['ffmpeg', '-y', '-i', path, '-vcodec', 'libx265', '-crf', '28', '-preset', 'fast', final_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.remove(path)
        print(f"Done: {final_path}")

    def process_live(self):
        source = int(self.args.input) if self.args.input.isdigit() else self.args.input
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        #target resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.line_start_px = int(w * self.args.zone_start_pct)
        self.line_end_px = int(w * self.args.zone_end_pct)

        print(f"Live Stream Started. Resolution: {w}x{h}")
        print(f"Right-to-Left Flow: Start Scan @ {line_end_px}px | Finish Scan @ {line_start_px}px")

        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo.track(
                frame, 
                persist=True, 
                conf=self.args.yolo_conf, 
                imgsz=self.args.yolo_imgsz, 
                tracker="bytetrack.yaml",
                retina_masks=True, 
                verbose=False
            )

            cv2.line(frame, (self.line_start_px, 0), (self.line_start_px, h), (255, 0, 255), 2)
            cv2.line(frame, (self.line_end_px, 0), (self.line_end_px, h), (255, 0, 255), 2)

            if results[0].masks is not None and results[0].boxes.id is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    state = self.track_states[track_id]
                    x1, y1, x2, y2 = map(int, boxes[i])
                    center_x = (x1 + x2) // 2
                    
                    label = "WAITING"
                    color = (200, 200, 200)
                    
                    # Before Zone: skip
                    if center_x > self.line_end_px:
                        label = f"ID:{track_id} ENTERING"
                    
                    # Inside Zone: segment and detect anomalies
                    elif (self.line_start_px <= center_x <= self.line_end_px) and (state['decision'] is None):
                        img_crop = self.crop_and_mask_object(frame, masks[i])
                        if img_crop is not None:
                            is_def, score = self.detect_anomaly(Image.fromarray(img_crop), track_id)
                            state['results'].append(is_def)
                            label = f"ID:{track_id} SCANNING..."
                            color = (0, 255, 255)

                    # Exit Zone: lock decision following a strategy (OR, MAJORITY, etc.)
                    else:
                        if state['decision'] is None:
                            state['decision'] = self.get_final_decision(state['results'], strategy=self.args.ad_strategy)
                        
                        is_def = state['decision']
                        label = f"ID:{track_id} {'DEFECTIVE' if is_def else 'GOOD'}"
                        color = (0, 0, 255) if is_def else (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            curr_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Khomsa Live AD", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    pipe = AnomalyPipeline(args)
    pipe.process_video()