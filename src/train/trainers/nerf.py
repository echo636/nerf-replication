import torch
import torch.nn as nn
from src.models.nerf.renderer.volume_renderer import Renderer
import ipdb

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = Renderer(self.net)

        # add metrics here
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        """
        Write your codes here.
        """
        ret = self.renderer.render(batch)
        gt_rgb = batch['rgbs']
        loss_c = self.loss_fn(ret['rgb_map_c'], gt_rgb)

        loss_stats = {
            'loss_c': loss_c
        }

        if 'rgb_map_f' in ret:
            loss_f = self.loss_fn(ret['rgb_map_f'], gt_rgb)
            total_loss = loss_c + loss_f
            loss_stats['loss_f'] = loss_f
            loss_stats['total_loss'] = total_loss
        else:
            total_loss = loss_c
            loss_stats['total_loss'] = total_loss
        
        output = ret

        loss = total_loss

        #先不在这里计算psnr了，在evaluator里算
        # with torch.no_grad():
        #     pred_rgb = ret.get('rgb_map_f', ret['rgb_map_c'])
        #     #ipdb.set_trace()
        #     mse = torch.mean((pred_rgb - gt_rgb) ** 2)
        #     psnr = -10. * torch.log10(mse + 1e-6)

        # image_stats = {
        #     'psnr': psnr
        # }
        
        return output, loss, loss_stats#, image_stats