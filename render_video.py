#渲染360度旋转视频

from src.config import cfg, args
import torch
import numpy as np
import os
import imageio
from tqdm import tqdm

trans_t = lambda t : torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]]).float()
rot_phi = lambda phi : torch.Tensor([[1,0,0,0],[0,np.cos(phi),-np.sin(phi),0],[0,np.sin(phi), np.cos(phi),0],[0,0,0,1]]).float()
rot_theta = lambda th : torch.Tensor([[np.cos(th),0,-np.sin(th),0],[0,1,0,0],[np.sin(th),0, np.cos(th),0],[0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def render_360_video():
    from src.models import make_network
    from src.models.nerf.renderer.make_renderer import make_renderer
    from src.datasets.nerf.blender import Dataset as BlenderDataset
    from src.utils.net_utils import load_network
    from src.utils.data_utils import to_cuda

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Rendering experiment: {cfg.exp_name} on scene: {cfg.scene}")

    network = make_network(cfg)
    load_network(network, cfg.trained_model_dir, resume=True)
    network.to(device)
    network.eval()
    
    renderer = make_renderer(cfg, network)

    num_frames = 240
    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, num_frames + 1)[:-1]], 
        0
    ).to(device)

    video_frames_dir = os.path.join(cfg.result_dir, "video_frames")
    os.makedirs(video_frames_dir, exist_ok=True)
    
    temp_dataset = BlenderDataset(**cfg.test_dataset)
    H, W, focal = temp_dataset.H, temp_dataset.W, temp_dataset.focal

    frames = []

    for i, pose in enumerate(tqdm(render_poses, desc="Rendering video frames")):
        with torch.no_grad():
            rays_o, rays_d = temp_dataset.get_rays(H, W, focal, pose)
            rays = torch.cat([rays_o.view(-1, 3), rays_d.view(-1, 3)], 1)
            
            batch = {'rays': rays, 'near': torch.tensor(cfg.task_arg.near), 'far': torch.tensor(cfg.task_arg.far)}
            batch = to_cuda(batch, device)

            output = renderer.render(batch)
            
            img = output.get('rgb_map_f', output['rgb_map_c']).cpu().numpy()
            img = (np.clip(img.reshape(H, W, 3), 0, 1) * 255).astype(np.uint8)
            
            frames.append(img)
    
    video_path = os.path.join(cfg.result_dir, f"{cfg.exp_name}_360_video_60fps.mp4")
    imageio.mimsave(video_path, frames, fps=30, quality=8)
    print(f"\n🎉 360度旋转视频已成功生成并保存至: {video_path}")


if __name__ == '__main__':
    render_360_video()