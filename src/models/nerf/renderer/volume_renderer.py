import numpy as np
import torch
from src.config import cfg
import ipdb


class Renderer:
    def __init__(self, net):
        """
        Write your codes here.
        """
        self.net = net
        
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        
        sigma = torch.nn.functional.relu(raw[..., 3] + noise)
        #ipdb.set_trace()
        alpha = 1. - torch.exp(-sigma * dists)
        #ipdb.set_trace()
        transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        weights = alpha * transmittance

        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        depth_map = torch.sum(weights * z_vals, -1)

        acc_map = torch.sum(weights, -1)

        #ipdb.set_trace()
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights

    def sample_pdf(self, bins, weights, N_samples, det=False):
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

        u = u.contiguous()
        #ipdb.set_trace()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples
    

    def render(self, batch):
        """
        Write your codes here.
        """
        #ipdb.set_trace()
        rays = batch['rays']

        if rays.ndim == 3:
            B, N, C = rays.shape
            rays_flat = rays.reshape(B * N, C)
        else:
            rays_flat = rays

        #ray_o, ray_d = rays_flat[..., 0:3], rays_flat[..., 3:6]

        near, far = batch['near'], batch['far']
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        N_rays_total = rays_flat.shape[0]
        for i in range(0, N_rays_total, chunk):
            rays_chunk = rays_flat[i:i+chunk]
            ray_o, ray_d = rays_chunk[..., 0:3], rays_chunk[..., 3:6]
            N_rays = ray_o.shape[0]
            N_samples = cfg.task_arg.N_samples
            
            t_vals = torch.linspace(0., 1., steps=N_samples, device=ray_o.device, dtype=torch.float32) #要在GPU上
            #ipdb.set_trace()
            z_vals = near * (1. - t_vals) + far * t_vals
            z_vals = z_vals.expand([N_rays, N_samples])
            
            if cfg.task_arg.perturb > 0.:
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                t_rand = torch.rand(z_vals.shape, device=ray_o.device)
                #ipdb.set_trace()
                z_vals = lower + (upper - lower) * t_rand

            pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]

            viewdirs = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            #ipdb.set_trace()
            #viewdirs_expanded = viewdirs.unsqueeze(1).expand(-1, N_samples, -1)
            raw_c = self.net(pts, viewdirs, 'coarse')
            
            #ipdb.set_trace()

            rgb_map_c, depth_map_c, acc_map_c, weights_c = self.raw2outputs(raw_c, z_vals, ray_d,
                                                                        cfg.task_arg.raw_noise_std,
                                                                        cfg.task_arg.white_bkgd)
            
            #ipdb.set_trace()

            chunk_ret = {'rgb_map_c': rgb_map_c, 'depth_map_c': depth_map_c, 'acc_map_c': acc_map_c}

            N_importance = cfg.task_arg.N_importance
            if N_importance > 0:
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                #ipdb.set_trace()
                z_samples = self.sample_pdf(z_vals_mid, weights_c[..., 1:-1], N_importance,
                                            det=(cfg.task_arg.perturb == 0.))
                z_samples = z_samples.detach()

                z_vals_f, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                
                pts_f = ray_o[..., None, :] + ray_d[..., None, :] * z_vals_f[..., :, None]

                #viewdirs_expanded_f = viewdirs.unsqueeze(1).expand(-1, N_samples + N_importance, -1)
                raw_f = self.net(pts_f, viewdirs, 'fine')
                
                #ipdb.set_trace()
                rgb_map_f, depth_map_f, acc_map_f, _ = self.raw2outputs(raw_f, z_vals_f, ray_d,
                                                                    cfg.task_arg.raw_noise_std,
                                                                    cfg.task_arg.white_bkgd)
                
                #ipdb.set_trace()
                chunk_ret['rgb_map_f'] = rgb_map_f
                chunk_ret['depth_map_f'] = depth_map_f
                chunk_ret['acc_map_f'] = acc_map_f
            
            for k, v in chunk_ret.items():
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(v)

        for k, v in all_ret.items():
            all_ret[k] = torch.cat(v, 0)

        return all_ret

