import numpy as np
from src.config import cfg
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import json
import warnings
import ipdb

warnings.filterwarnings("ignore", category=UserWarning)


class Evaluator:
    def __init__(
        self,
    ):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.imgs = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, id, num_imgs):
        result_dir = os.path.join(cfg.result_dir, "images")
        os.system("mkdir -p {}".format(result_dir))
        cv2.imwrite(
            "{}/view{:03d}_pred.png".format(result_dir, id),
            (img_pred[..., [2, 1, 0]] * 255),
        )
        cv2.imwrite(
            "{}/view{:03d}_gt.png".format(result_dir, id),
            (img_gt[..., [2, 1, 0]] * 255),
        )
        img_pred = (img_pred * 255).astype(np.uint8)

        #ipdb.set_trace()
        #ssim = compare_ssim(img_pred, img_gt, win_size=101, full=True, )
        ssim = compare_ssim(img_pred, img_gt, full=True, channel_axis=-1, data_range=img_pred.max() - img_pred.min())
        #ipdb.set_trace()
        return ssim[0]

    def evaluate(self, output, batch):
        """
        Write your codes here.
        """
        img_pred_flat = output['rgb_map_f'].cpu().numpy()
        img_gt_batched = batch['rgbs'].cpu().numpy()
        i = batch['i'].item() 

        #result_dir = os.path.join(cfg.result_dir, "images")
        #os.makedirs(result_dir, exist_ok=True)
        img_gt_flat = img_gt_batched.squeeze(0)
        #ipdb.set_trace()
        H, W = batch['H'].item(), batch['W'].item()
        #reshape成图片的形状
        img_pred = img_pred_flat.reshape(H, W, 3)
        img_gt = img_gt_flat.reshape(H, W, 3)
        psnr = self.psnr_metric(img_pred, img_gt)
        #把真实图片也转成0-255的形式，与预测图片对应
        img_gt = (img_gt * 255).astype(np.uint8)
        ssim = self.ssim_metric(img_pred, img_gt, batch, i, 100)  #硬编码，后续要修改

        self.psnr.append(psnr)
        self.ssim.append(ssim)

    def summarize(self):
        """
        Write your codes here.
        """
        mean_psnr = np.mean(self.psnr)
        mean_ssim = np.mean(self.ssim)

        print(f"Final Evaluation Results:")
        print(f"  Average PSNR: {mean_psnr:.4f}")
        print(f"  Average SSIM: {mean_ssim:.4f}")

        summary_path = os.path.join(cfg.result_dir, "summary.json")
        summary_data = {
            'mean_psnr': mean_psnr,
            'mean_ssim': mean_ssim
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        print(f"\nSummary saved to {summary_path}")
        
        return {'psnr': mean_psnr, 'ssim': mean_ssim}
