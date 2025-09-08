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

    def forward(self, batch, writer=None, global_step=None): #增加参数，用于TensorBoard可视化
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

        with torch.no_grad():
            pred_rgb = ret.get('rgb_map_f', ret['rgb_map_c'])
            #ipdb.set_trace()
            mse = torch.mean((pred_rgb - gt_rgb) ** 2)
            psnr = -10. * torch.log10(mse)

        image_stats = {
            'psnr': psnr
        }
        
        if writer is not None:
            writer.add_scalar('Loss/total', loss_stats['total_loss'].item(), global_step)
            writer.add_scalar('Loss/coarse', loss_stats['loss_c'].item(), global_step)
            if 'loss_f' in loss_stats:
                writer.add_scalar('Loss/fine', loss_stats['loss_f'].item(), global_step)
            
            #记录训练时的PSNR
            writer.add_scalar('PSNR/train', image_stats['psnr'].item(), global_step)
            
        return output, loss, loss_stats, image_stats