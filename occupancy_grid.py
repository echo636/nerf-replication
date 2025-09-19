#该脚本用于获取occupancy grid
from src.config import cfg, args
from src.models import make_network
from src.utils.net_utils import (
    load_model,
    save_model,
    load_network,
    save_trained_config,
    load_pretrain,
)
import torch
import os
import numpy as np

def main():
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    resolution = cfg.task_arg.occupancy_grid_res
    density_threshold = cfg.task_arg.occupancy_grid_threshold
    scene_bbox = torch.tensor(cfg.train_dataset.scene_bbox, device='cuda', dtype=torch.float32)

    global_min = scene_bbox[0]
    grid_list = [resolution, resolution, resolution]
    voxel_size = (scene_bbox[1] - scene_bbox[0]) / torch.tensor(grid_list, device='cuda', dtype=torch.float32)
    num_voxels = grid_list[0] * grid_list[1] * grid_list[2]
    subsample_list = [2, 2, 2]
    num_samples = subsample_list[0] * subsample_list[1] * subsample_list[2]

    voxel_samples = []
    for dim in range(3):
        voxel_samples.append(torch.linspace(0.0, 1.0, subsample_list[dim]) * voxel_size.cpu()[dim])

    voxel_samples = torch.stack(torch.meshgrid(*voxel_samples, indexing='ij'), -1).view(-1, 3)

    ranges = [torch.arange(0, res) for res in grid_list]
    grid_indices = torch.stack(torch.meshgrid(*ranges, indexing='ij'), -1).float()
    base_addr = global_min.cpu() + grid_indices * voxel_size.cpu()
    points = base_addr.unsqueeze(3) + voxel_samples.view(1, 1, 1, num_samples, 3)
    points = points.view(num_voxels, num_samples, 3)

    batch_size = cfg.task_arg.occupancy_grid_batch_size
    all_density = torch.empty((num_voxels, num_samples), device='cpu', dtype=torch.float32)
    viewdirs_template = torch.zeros((num_samples, 3), device='cuda', dtype=torch.float32)

    from tqdm import tqdm
    with torch.no_grad():
        for i in tqdm(range(0, num_voxels, batch_size), desc="Baking Grid"):
            start = i
            end = min(i + batch_size, num_voxels)

            points_batch = points[start:end].cuda()
            #pts = points_batch.view(-1, 3)

            num_voxels_in_batch = end - start
            dummy_viewdirs = torch.zeros(num_voxels_in_batch, 3, device='cuda')
            #viewdirs = viewdirs_template.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, 3)
            raw = network(points_batch, dummy_viewdirs, 'coarse')
            sigma = torch.nn.functional.relu(raw[..., 3])
            all_density[start:end] = sigma.view(end - start, num_samples).cpu()

    del points

    all_density_gpu = all_density.cuda()
    occupancy_grid = all_density_gpu > density_threshold
    del all_density_gpu

    occupancy_grid = occupancy_grid.any(dim=-1)
    occupancy_grid = occupancy_grid.view(resolution, resolution, resolution)

    config_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
    log_dir = os.path.join('logs', config_name)
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, 'occupancy_grid.pt')

    print(f"Saving occupancy grid to: {output_path}")
    torch.save(occupancy_grid.cpu(), output_path)

    print("Done.")
if __name__ == "__main__":
    main()