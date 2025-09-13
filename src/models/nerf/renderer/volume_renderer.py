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
        
    #体渲染计算函数
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
        """
        Description:
            raw2outputs 函数负责将神经网络的输出转换为颜色图、深度图和不透明度图

        Input:
            @raw: 神经网络的原始输出，形状为 [num_rays, num_samples, 4]，最后一个维度包含 RGB 颜色和密度值
            @z_vals: 采样点的深度值，形状为 [num_rays, num_samples]
            @rays_d: 光线的方向，形状为 [num_rays, 3]
            @raw_noise_std: 添加到密度值的噪声标准差，用于正则化
            @white_bkgd: 是否使用白色背景

        Output:
            @rgb_map: 渲染得到的颜色图，形状为 [num_rays, 3]
            @depth_map: 渲染得到的深度图，形状为 [num_rays]
            @acc_map: 渲染得到的不透明度图，形状为 [num_rays]
            @weights: 每个采样点的权重，形状为 [num_rays, num_samples]
        """

        #计算每个采样点之间的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        #拼接一个无穷大的距离到最后一个采样点
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)

        #rays_d[..., None, :]用于增加一个维度方便广播
        #将距离乘以该条光线的方向向量的模长，以获得真实的距离
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        
        #对第4个通道（原始密度）应用ReLU激活函数，确保密度sigma为非负数
        sigma = torch.nn.functional.relu(raw[..., 3] + noise)
        #ipdb.set_trace()
        alpha = 1. - torch.exp(-sigma * dists)
        #ipdb.set_trace()
        #计算透射率 T_i = Π(1 - alpha_j) for j < i
        #torch.cumprod是一个累积乘法函数，适合用来计算这个连乘
        #1e-10是为了数值稳定性
        transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        weights = alpha * transmittance

        #将每个采样点的颜色rgb与它的权重weights相乘，然后求和，得到最终的像素颜色
        #None用于增加一个维度以便广播
        #在倒数第二个维度上求和，即对所有采样点进行加权求和
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        #计算深度图：每个采样点的深度z_vals与其权重weights的加权平均
        depth_map = torch.sum(weights * z_vals, -1)

        #计算累积透明度图
        acc_map = torch.sum(weights, -1)

        #ipdb.set_trace()
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights

    def sample_pdf(self, bins, weights, N_samples, det=False):
        """
        Description:
            sample_pdf 函数负责根据给定的权重分布进行重要性采样
        
        Input:
            @bins: 采样点的边界，形状为 [num_rays, num_bins]
            @weights: 每个采样点的权重，形状为 [num_rays, num_bins-1]
            @N_samples: 需要采样的点数
            @det: 是否使用确定性采样（均匀间隔）还是随机采样
        
        Output:
            @samples: 采样得到的新点，形状为 [num_rays, N_samples]
        """
        weights = weights + 1e-5

        #归一化权重，得到概率密度函数(PDF)
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        #计算PDF的累积分布函数(CDF)，并在前面拼接一个0
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

        #确保张量 u 在内存中是连续存储的，可以提高后续操作的效率
        u = u.contiguous()
        #ipdb.set_trace()

        #searchsorted会在已经排好序的 cdf 张量中，为 u 中的每一个随机数，找到一个合适的插入位置，使得 cdf 仍然保持有序。
        #返回值inds 是一个和 u 形状相同的张量 [N_rays, N_samples_fine]。inds[i, j] 的值，就是 u[i, j] 这个随机数应该被插入到 cdf[i, :] 这个CDF数组中的索引。
        #right=True 表示如果 u 中的值与 cdf 中的某个值相等，则插入到该值的右边。
        inds = torch.searchsorted(cdf, u, right=True)

        #确定采样点的上下界索引，有安全检查
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        #inds_g 的形状现在是 [N_rays, N_samples_fine, 2]。inds_g[i, j, 0] 是第j个新采样点的下界索引，inds_g[i, j, 1] 是上界索引。
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
    
        #确保光线数据是 [N, 6] 的二维形式
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

        #光线分块处理，防止显存爆炸
        for i in range(0, N_rays_total, chunk):
            rays_chunk = rays_flat[i:i+chunk]
            ray_o, ray_d = rays_chunk[..., 0:3], rays_chunk[..., 3:6]
            N_rays = ray_o.shape[0]
            #
            N_samples = cfg.task_arg.N_samples
            
            #生成线性采样点
            t_vals = torch.linspace(0., 1., steps=N_samples, device=ray_o.device, dtype=torch.float32) #要在GPU上
            #ipdb.set_trace()
            #把线性采样点映射到实际的深度值
            z_vals = near * (1. - t_vals) + far * t_vals
            z_vals = z_vals.expand([N_rays, N_samples])
            
            #在训练时对采样点进行扰动
            if cfg.task_arg.perturb > 0.:
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                t_rand = torch.rand(z_vals.shape, device=ray_o.device)
                #ipdb.set_trace()
                z_vals = lower + (upper - lower) * t_rand

            #计算每个采样点的世界坐标
            pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]

            #准备视角方向，是单位向量
            viewdirs = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            #ipdb.set_trace()
            #viewdirs_expanded = viewdirs.unsqueeze(1).expand(-1, N_samples, -1)

            #送入粗糙网络
            raw_c = self.net(pts, viewdirs, 'coarse')
            
            #ipdb.set_trace()

            #体渲染
            rgb_map_c, depth_map_c, acc_map_c, weights_c = self.raw2outputs(raw_c, z_vals, ray_d,
                                                                        cfg.task_arg.raw_noise_std,
                                                                        cfg.task_arg.white_bkgd)
            
            #ipdb.set_trace()

            chunk_ret = {'rgb_map_c': rgb_map_c, 'depth_map_c': depth_map_c, 'acc_map_c': acc_map_c}

            N_importance = cfg.task_arg.N_importance

            #进行精细采样
            if N_importance > 0:
                #根据粗糙网络的输出权重进行重要性采样
                #z_vals_mid 采样点的中间值
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                #ipdb.set_trace()
                #生成N_importance个新的采样点
                z_samples = self.sample_pdf(z_vals_mid, weights_c[..., 1:-1], N_importance,
                                            det=(cfg.task_arg.perturb == 0.))
                z_samples = z_samples.detach()

                #将新的采样点和原始采样点合并，并排序
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
            
            #
            for k, v in chunk_ret.items():
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(v)

        for k, v in all_ret.items():
            all_ret[k] = torch.cat(v, 0)

        return all_ret

