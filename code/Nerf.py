import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== Data Loading ====================
class NeRFDataset:
    def __init__(self, data_dir, split='train', img_wh=(800, 800), downsample=4):
        self.data_dir = data_dir
        self.split = split
        self.img_wh = (img_wh[0] // downsample, img_wh[1] // downsample)
        
        # Load transforms
        with open(os.path.join(data_dir, f'transforms_{split}.json'), 'r') as f:
            meta = json.load(f)
        
        self.focal = 0.5 * self.img_wh[0] / np.tan(0.5 * meta['camera_angle_x'])
        
        # Load images and poses
        self.images = []
        self.poses = []
        
        for frame in meta['frames']:
            img_path = os.path.join(data_dir, split, f"{frame['file_path'].split('/')[-1]}.png")
            img = Image.open(img_path).convert('RGBA')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = np.array(img) / 255.0
            
            # White background
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            self.images.append(img)
            self.poses.append(np.array(frame['transform_matrix']))
        
        self.images = np.stack(self.images, 0)
        self.poses = np.stack(self.poses, 0)
        
        print(f"Loaded {len(self.images)} {split} images at {self.img_wh}")
    
    def get_rays(self, H, W, focal, c2w):
        """Get ray origins and directions from camera pose"""
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
        i, j = i.to(c2w.device), j.to(c2w.device)
        dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d

# ==================== NeRF Model ====================
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        # Positional encoding
        self.L_pos = 10  # Levels for position
        self.L_dir = 4   # Levels for direction
        self.input_ch = 3 + 3 * 2 * self.L_pos
        self.input_ch_views = 3 + 3 * 2 * self.L_dir
        
        # Spatial network
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) 
             for i in range(D-1)])
        
        # Direction network
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W//2)])
        
        # Output layers
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
    
    def positional_encoding(self, x, L):
        """Positional encoding"""
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * np.pi * x))
            out.append(torch.cos(2**j * np.pi * x))
        return torch.cat(out, -1)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, 3], dim=-1)
        input_pts = self.positional_encoding(input_pts, self.L_pos)
        input_views = self.positional_encoding(input_views, self.L_dir)
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

# ==================== Rendering ====================
def render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=64, N_importance=128):
    """Volume rendering"""
    # Coarse sampling
    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([rays_o.shape[0], N_samples])
    
    # Add noise to sampling
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=rays_o.device)
    z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    dirs = rays_d[:,None,:].expand(pts.shape)
    
    # Evaluate model
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    input_flat = torch.cat([pts_flat, dirs_flat], -1)
    
    # Process in chunks to avoid OOM
    chunk = 1024*32
    outputs = []
    for i in range(0, input_flat.shape[0], chunk):
        outputs.append(model(input_flat[i:i+chunk]))
    raw = torch.cat(outputs, 0).reshape(*pts.shape[:-1], 4)
    
    rgb, sigma = raw[..., :3], raw[..., 3]
    rgb = torch.sigmoid(rgb)
    
    # Volume rendering
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)
    
    alpha = 1.-torch.exp(-F.relu(sigma) * dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    # Fine sampling
    if N_importance > 0:
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance)
        z_samples = z_samples.detach()
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        dirs = rays_d[:,None,:].expand(pts.shape)
        
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs.reshape(-1, 3)
        input_flat = torch.cat([pts_flat, dirs_flat], -1)
        
        outputs = []
        for i in range(0, input_flat.shape[0], chunk):
            outputs.append(model(input_flat[i:i+chunk]))
        raw = torch.cat(outputs, 0).reshape(*pts.shape[:-1], 4)
        
        rgb, sigma = raw[..., :3], raw[..., 3]
        rgb = torch.sigmoid(rgb)
        
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)
        
        alpha = 1.-torch.exp(-F.relu(sigma) * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[...,None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
    
    return rgb_map, depth_map, acc_map

def sample_pdf(bins, weights, N_samples):
    """Hierarchical sampling"""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    
    return samples

# ==================== Training ====================
def train():
    # Load dataset
    dataset = NeRFDataset('drive/MyDrive/Fundamentals_DL_and_AI/Latest/lego', split='train', downsample=4)
    val_dataset = NeRFDataset('drive/MyDrive/Fundamentals_DL_and_AI/Latest/lego', split='val', downsample=4)
    
    # Initialize model
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Training parameters
    n_iters = 200000
    batch_size = 1024
    i_print = 100
    i_img = 2000
    i_weights = 10000
    
    H, W = dataset.img_wh[1], dataset.img_wh[0]
    focal = dataset.focal
    
    print("Starting training...")
    for i in tqdm(range(n_iters)):
        # Random image
        img_i = np.random.choice(len(dataset.images))
        target = torch.Tensor(dataset.images[img_i]).to(device)
        pose = torch.Tensor(dataset.poses[img_i]).to(device)
        
        # Get rays
        rays_o, rays_d = dataset.get_rays(H, W, focal, pose)
        
        # Random ray batch
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), 
                                           torch.linspace(0, W-1, W), indexing='ij'), -1)
        coords = coords.reshape(-1, 2)
        select_inds = np.random.choice(coords.shape[0], size=[batch_size], replace=False)
        select_coords = coords[select_inds].long()
        
        rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]].to(device)
        rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]].to(device)
        target_batch = target[select_coords[:, 0], select_coords[:, 1]]
        
        # Render
        rgb, depth, acc = render_rays(model, rays_o_batch, rays_d_batch)
        
        # Loss
        loss = F.mse_loss(rgb, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Decay learning rate
        decay_rate = 0.1
        decay_steps = n_iters // 2
        new_lrate = 5e-4 * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        if i % i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():.4f}")
        
        # Validation
        if i % i_img == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                val_img_i = 0
                val_pose = torch.Tensor(val_dataset.poses[val_img_i]).to(device)
                rays_o, rays_d = val_dataset.get_rays(H, W, focal, val_pose)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                
                # Render in chunks
                chunk = 1024
                rgbs = []
                for j in range(0, rays_o.shape[0], chunk):
                    rgb, _, _ = render_rays(model, rays_o[j:j+chunk], rays_d[j:j+chunk])
                    rgbs.append(rgb.cpu())
                rgb = torch.cat(rgbs, 0).reshape(H, W, 3).numpy()
                
                # Save image
                os.makedirs('results', exist_ok=True)
                plt.imsave(f'results/iter_{i:06d}.png', np.clip(rgb, 0, 1))
                print(f"Saved validation image at iteration {i}")
            model.train()
        
        # Save weights
        if i % i_weights == 0 and i > 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'iter': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/nerf_{i:06d}.pth')
            print(f"Saved checkpoint at iteration {i}")
    
    print("Training complete!")
    torch.save(model.state_dict(), 'nerf_final.pth')

# ==================== 3D Reconstruction Export ====================
def extract_geometry(model, resolution=256, threshold=10.0, bbox_min=-1.5, bbox_max=1.5):
    """Extract 3D mesh from trained NeRF using marching cubes"""
    print("Extracting 3D geometry...")
    model.eval()
    
    # Create 3D grid
    x = np.linspace(bbox_min, bbox_max, resolution)
    y = np.linspace(bbox_min, bbox_max, resolution)
    z = np.linspace(bbox_min, bbox_max, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Query density at all points
    print("Querying density field...")
    chunk = 1024 * 64
    densities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(points), chunk)):
            pts = torch.FloatTensor(points[i:i+chunk]).to(device)
            # Use dummy view direction
            dirs = torch.zeros_like(pts)
            dirs[:, 2] = -1
            
            input_data = torch.cat([pts, dirs], -1)
            output = model(input_data)
            density = output[:, 3].cpu().numpy()
            densities.append(density)
    
    densities = np.concatenate(densities, axis=0)
    densities = densities.reshape(resolution, resolution, resolution)
    
    # Apply threshold
    densities = np.maximum(densities, 0)
    
    # Save as numpy array
    print("Saving density grid...")
    os.makedirs('3d_output', exist_ok=True)
    np.save('3d_output/density_grid.npy', densities)
    
    # Export to PLY point cloud
    print("Creating point cloud...")
    mask = densities > threshold
    pts = np.stack([xx[mask], yy[mask], zz[mask]], axis=-1)
    
    # Get colors for points
    colors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(pts), chunk)):
            pts_batch = torch.FloatTensor(pts[i:i+chunk]).to(device)
            dirs_batch = torch.zeros_like(pts_batch)
            dirs_batch[:, 2] = -1
            
            input_data = torch.cat([pts_batch, dirs_batch], -1)
            output = model(input_data)
            rgb = torch.sigmoid(output[:, :3]).cpu().numpy()
            colors.append(rgb)
    
    colors = np.concatenate(colors, axis=0)
    colors = (colors * 255).astype(np.uint8)
    
    # Save PLY file
    print(f"Saving point cloud with {len(pts)} points...")
    save_ply('3d_output/reconstruction.ply', pts, colors)
    
    # Try marching cubes if available
    try:
        from skimage import measure
        print("Running marching cubes...")
        vertices, faces, normals, values = measure.marching_cubes(densities, threshold)
        
        # Scale vertices to world coordinates
        vertices = vertices / resolution * (bbox_max - bbox_min) + bbox_min
        
        # Save mesh
        save_obj('3d_output/reconstruction.obj', vertices, faces)
        print("Saved mesh to 3d_output/reconstruction.obj")
    except ImportError:
        print("scikit-image not available, skipping mesh extraction")
        print("Install with: pip install scikit-image")
    
    print("3D reconstruction complete!")
    print("Files saved in 3d_output/:")
    print("  - density_grid.npy (raw density values)")
    print("  - reconstruction.ply (colored point cloud)")
    if os.path.exists('3d_output/reconstruction.obj'):
        print("  - reconstruction.obj (triangle mesh)")

def save_ply(filename, points, colors):
    """Save point cloud as PLY file"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")

def save_obj(filename, vertices, faces):
    """Save mesh as OBJ file"""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def evaluate_test_set(model, test_dataset, save_dir='test_results'):
    """Evaluate on test set and compute metrics"""
    print("Evaluating on test set...")
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    H, W = test_dataset.img_wh[1], test_dataset.img_wh[0]
    focal = test_dataset.focal
    
    psnrs = []
    ssims = []
    lpips_scores = []
    
    # Try to import LPIPS for perceptual loss
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        use_lpips = True
    except ImportError:
        print("LPIPS not available (install with: pip install lpips)")
        use_lpips = False
    
    for img_idx in tqdm(range(len(test_dataset.images))):
        target = torch.Tensor(test_dataset.images[img_idx]).to(device)
        pose = torch.Tensor(test_dataset.poses[img_idx]).to(device)
        
        # Render image
        with torch.no_grad():
            rays_o, rays_d = test_dataset.get_rays(H, W, focal, pose)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            chunk = 1024
            rgbs = []
            depths = []
            for j in range(0, rays_o.shape[0], chunk):
                rgb, depth, _ = render_rays(model, rays_o[j:j+chunk], rays_d[j:j+chunk])
                rgbs.append(rgb.cpu())
                depths.append(depth.cpu())
            
            rgb = torch.cat(rgbs, 0).reshape(H, W, 3)
            depth = torch.cat(depths, 0).reshape(H, W)
        
        # Compute PSNR
        mse = F.mse_loss(rgb, target.cpu())
        psnr = -10. * torch.log10(mse)
        psnrs.append(psnr.item())
        
        # Compute SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            rgb_np = rgb.numpy()
            target_np = target.cpu().numpy()
            ssim_val = ssim(rgb_np, target_np, multichannel=True, channel_axis=2, data_range=1.0)
            ssims.append(ssim_val)
        except ImportError:
            pass
        
        # Compute LPIPS
        if use_lpips:
            rgb_lpips = rgb.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            target_lpips = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lpips_val = lpips_fn(rgb_lpips, target_lpips).item()
            lpips_scores.append(lpips_val)
        
        # Save images
        rgb_np = np.clip(rgb.numpy(), 0, 1)
        target_np = np.clip(target.cpu().numpy(), 0, 1)
        depth_np = depth.numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        
        # Create comparison image
        comparison = np.concatenate([target_np, rgb_np, np.stack([depth_np]*3, axis=-1)], axis=1)
        plt.imsave(f'{save_dir}/test_{img_idx:03d}.png', comparison)
    
    # Compute average metrics
    avg_psnr = np.mean(psnrs)
    print(f"\n{'='*50}")
    print(f"Test Set Evaluation Results:")
    print(f"{'='*50}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    if ssims:
        avg_ssim = np.mean(ssims)
        print(f"Average SSIM: {avg_ssim:.4f}")
    
    if lpips_scores:
        avg_lpips = np.mean(lpips_scores)
        print(f"Average LPIPS: {avg_lpips:.4f}")
    
    print(f"{'='*50}")
    print(f"\nResults for each test image saved in {save_dir}/")
    print("Each image shows: [Ground Truth | Rendered | Depth Map]")
    
    # Save metrics to file
    with open(f'{save_dir}/metrics.txt', 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        if ssims:
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        if lpips_scores:
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write("\nPer-image PSNR:\n")
        for i, psnr in enumerate(psnrs):
            f.write(f"Image {i}: {psnr:.2f} dB\n")
    
    print(f"Metrics saved to {save_dir}/metrics.txt")
    
    return avg_psnr, ssims, lpips_scores

def render_360_video(model, dataset, n_frames=120, output_path='3d_output/rotation.mp4'):
    """Render 360 degree rotation video"""
    print("Rendering 360° video...")
    model.eval()
    
    H, W = dataset.img_wh[1], dataset.img_wh[0]
    focal = dataset.focal
    
    frames = []
    
    # Create circular camera path
    for i in tqdm(range(n_frames)):
        angle = 2 * np.pi * i / n_frames
        
        # Camera position in circle
        radius = 4.0
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 0.0
        
        # Look at origin
        cam_pos = np.array([cam_x, cam_y, cam_z])
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Create camera-to-world matrix
        z_axis = cam_pos - look_at
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        c2w = np.eye(4)
        c2w[:3, 0] = x_axis
        c2w[:3, 1] = y_axis
        c2w[:3, 2] = z_axis
        c2w[:3, 3] = cam_pos
        
        pose = torch.FloatTensor(c2w).to(device)
        
        # Render frame
        with torch.no_grad():
            rays_o, rays_d = dataset.get_rays(H, W, focal, pose)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            chunk = 1024
            rgbs = []
            for j in range(0, rays_o.shape[0], chunk):
                rgb, _, _ = render_rays(model, rays_o[j:j+chunk], rays_d[j:j+chunk])
                rgbs.append(rgb.cpu())
            
            rgb = torch.cat(rgbs, 0).reshape(H, W, 3).numpy()
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            frames.append(rgb)
    
    # Save video
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Saved 360° video to {output_path}")
    except ImportError:
        print("OpenCV not available for video export")
        print("Saving frames as images instead...")
        os.makedirs('3d_output/frames', exist_ok=True)
        for i, frame in enumerate(frames):
            plt.imsave(f'3d_output/frames/frame_{i:04d}.png', frame)
        print(f"Saved {len(frames)} frames to 3d_output/frames/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'extract', 'video', 'test'])
    parser.add_argument('--checkpoint', type=str, default='nerf_final.pth')
    parser.add_argument('--resolution', type=int, default=256, help='Grid resolution for 3D extraction')
    parser.add_argument('--threshold', type=float, default=10.0, help='Density threshold')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        # Evaluate on test set
        test_dataset = NeRFDataset('drive/MyDrive/Fundamentals_DL_and_AI/Latest/lego', split='test', downsample=4)
        model = NeRF().to(device)
        
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            exit(1)
        
        evaluate_test_set(model, test_dataset)
    elif args.mode == 'extract':
        # Load model
        dataset = NeRFDataset('drive/MyDrive/Fundamentals_DL_and_AI/Latest/lego', split='val', downsample=4)
        model = NeRF().to(device)
        
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            exit(1)
        
        extract_geometry(model, resolution=args.resolution, threshold=args.threshold)
    elif args.mode == 'video':
        # Load model and render video
        dataset = NeRFDataset('drive/MyDrive/Fundamentals_DL_and_AI/Latest/lego', split='val', downsample=4)
        model = NeRF().to(device)
        
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")
            exit(1)
        
        render_360_video(model, dataset)