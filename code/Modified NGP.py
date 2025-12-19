import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from skimage import measure

# ==========================================
# 1. HELPER FUNCTIONS & RENDERING
# ==========================================

def save_obj(vertices, faces, colors, filename):
    print(f"Saving mesh to {filename}...")
    with open(filename, 'w') as f:
        f.write(f"# Mesh extracted from NGP\n")
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print("Done.")

@torch.no_grad()
def extract_mesh(model, resolution=256, threshold=10.0, bound=1.5, device='cuda'):
    print(f"Extracting mesh with resolution {resolution}^3 and density threshold {threshold}...")
    
    # Grid generation on GPU
    X = torch.linspace(-bound, bound, resolution, device=device)
    Y = torch.linspace(-bound, bound, resolution, device=device)
    Z = torch.linspace(-bound, bound, resolution, device=device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    # Query density in chunks (keep on GPU until result needed)
    chunk_size = 64**3 
    sigma_grid = []
    dummy_dirs = torch.zeros_like(points, device=device)
    
    print("Querying density field...")
    for i in tqdm(range(0, points.shape[0], chunk_size)):
        chunk_pts = points[i:i+chunk_size]
        chunk_dirs = dummy_dirs[i:i+chunk_size]
        _, sigma = model(chunk_pts, chunk_dirs)
        sigma_grid.append(sigma.cpu()) # Move to CPU only for storage to save VRAM for marching cubes
    
    sigma_grid = torch.cat(sigma_grid).numpy().reshape(resolution, resolution, resolution)
    
    # Marching Cubes (CPU only - skimage limitation)
    try:
        vertices, faces, normals, values = measure.marching_cubes(sigma_grid, level=threshold)
    except (RuntimeError, ValueError) as e:
        print(f"Mesh extraction failed: {e}")
        return

    # Normalize vertices back to world space
    vertices = vertices / (resolution - 1) * (2 * bound) - bound
    
    # Color Query on GPU
    print("Querying vertex colors...")
    vertex_colors = []
    vertices_tensor = torch.from_numpy(vertices).float().to(device)
    vertex_dirs = torch.zeros_like(vertices_tensor)
    vertex_dirs[..., 2] = -1.0 
    
    chunk_size = 100000
    for i in tqdm(range(0, vertices_tensor.shape[0], chunk_size)):
        chunk_pts = vertices_tensor[i:i+chunk_size]
        chunk_dirs = vertex_dirs[i:i+chunk_size]
        colors, _ = model(chunk_pts, chunk_dirs)
        vertex_colors.append(colors.cpu())
        
    vertex_colors = torch.cat(vertex_colors).numpy()
    
    os.makedirs('meshes', exist_ok=True)
    save_obj(vertices, faces, vertex_colors, f'meshes/mesh_res{resolution}_thresh{threshold}.obj')

@torch.no_grad()
def test_and_compare(model, hn, hf, dataset, img_index, chunk_size=10, nb_bins=256, H=800, W=800, device='cuda'):
    """
    Renders view and compares with GT. Everything stays on GPU until final image save.
    """
    # 1. Render Generated View
    # Slicing tensor on GPU
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    px_values = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size]
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size]
        px_values.append(render_rays(model, ray_origins_, ray_directions_,
                                     hn=hn, hf=hf, nb_bins=nb_bins))
    
    # Move to CPU only for image saving
    gen_img = torch.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
    gen_img = (gen_img.clip(0, 1) * 255.).astype(np.uint8)
    
    # 2. Extract Ground Truth (already on GPU)
    gt_flat = dataset[img_index * H * W: (img_index + 1) * H * W, 6:9]
    gt_img = gt_flat.cpu().numpy().reshape(H, W, 3)
    gt_img = (gt_img.clip(0, 1) * 255.).astype(np.uint8)
    
    # 3. Stitch Side-by-Side
    border = 5
    combined_w = W * 2 + border
    combined_img = np.zeros((H, combined_w, 3), dtype=np.uint8) + 255 # White border
    
    combined_img[:, :W, :] = gen_img
    combined_img[:, W+border:, :] = gt_img
    
    final_img = Image.fromarray(combined_img)
    save_path = f'novel_views/comparison_img_{img_index}.png'
    final_img.save(save_path)
    print(f"Saved comparison to {save_path}")

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=256):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    
    # Perturb sampling (Stratified)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor(
        [1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions_expanded = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_expanded.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors.reshape(x.shape)).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    
    return c + 1 - weight_sum.unsqueeze(-1) # White background

# ==========================================
# 2. DATA LOADER (GPU OPTIMIZED)
# ==========================================

def load_nerfstudio_data(data_dir, split='train', downscale=4, test_holdout=8, device='cuda'):
    """
    Loads images and computes rays entirely on the GPU.
    """
    json_path = os.path.join(data_dir, 'transforms.json')
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    W = int(meta['w']) // downscale
    H = int(meta['h']) // downscale
    
    # Camera Intrinsics
    fl_x = meta['fl_x'] / downscale
    fl_y = meta['fl_y'] / downscale
    cx = meta['cx'] / downscale
    cy = meta['cy'] / downscale
    
    all_frames = meta['frames']
    if split == 'train':
        selected_frames = [f for i, f in enumerate(all_frames) if i % test_holdout != 0]
    else:
        selected_frames = [f for i, f in enumerate(all_frames) if i % test_holdout == 0]

    print(f"Loading {split} data directly to {device}...")
    
    # Pre-compute ray directions grid ON GPU
    # Create grid
    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32), 
        torch.arange(H, device=device, dtype=torch.float32), 
        indexing='xy'
    )
    
    # Ray directions in Camera Coordinates
    dirs = torch.stack([(i - cx) / fl_x, -(j - cy) / fl_y, -torch.ones_like(i)], -1)
    
    all_rays_list = []
    
    for frame in tqdm(selected_frames):
        fname = frame['file_path']
        if fname.startswith('./'): fname = fname[2:]
        img_path = os.path.join(data_dir, fname)
        
        if not os.path.exists(img_path): continue
            
        # 1. Load Image (CPU -> GPU)
        # PIL is CPU based, unavoidable, but we move to GPU immediately
        img = Image.open(img_path)
        img = img.resize((W, H), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float().to(device) / 255.0
        
        # Handle Alpha
        if img_tensor.shape[-1] == 4:
            img_tensor = img_tensor[..., :3] * img_tensor[..., -1:] + (1 - img_tensor[..., -1:])
        else:
            img_tensor = img_tensor[..., :3]
            
        # 2. Compute Rays (GPU)
        c2w = torch.tensor(frame['transform_matrix'], device=device, dtype=torch.float32)
        
        # Rotate directions (Map vector to world)
        # dirs: [H, W, 3], c2w[:3, :3]: [3, 3]
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1) 
        
        # Translate origins (Camera position)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        
        # 3. Flatten and Store
        # [Origins, Directions, Colors]
        rays = torch.cat([rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img_tensor.reshape(-1, 3)], dim=-1)
        all_rays_list.append(rays)
        
    if not all_rays_list: raise ValueError("No rays loaded!")
    
    # Concatenate all into one massive GPU tensor
    return torch.cat(all_rays_list, dim=0), H, W, meta.get('aabb_scale', 16)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

class NGP(torch.nn.Module):
    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.L = L
        self.aabb_scale = aabb_scale
        # Hash tables stored on GPU
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T, 2), device=device) * 2 - 1) * 1e-4) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861
        
        self.density_MLP = nn.Sequential(
            nn.Linear(self.F * len(Nl), 64), nn.ReLU(), nn.Linear(64, 16)
        ).to(device)
        
        self.color_MLP = nn.Sequential(
            nn.Linear(27 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid()
        ).to(device)

    def positional_encoding(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        x /= self.aabb_scale
        mask = (x[:, 0].abs() < .5) & (x[:, 1].abs() < .5) & (x[:, 2].abs() < .5)
        x += 0.5
        
        color = torch.zeros((x.shape[0], 3), device=x.device)
        log_sigma = torch.zeros((x.shape[0]), device=x.device) - 100000
        if mask.sum() == 0: return color, torch.exp(log_sigma)
            
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
        for i, N in enumerate(self.Nl):
            # Grid hash lookups (Pure GPU)
            floor = torch.floor(x[mask] * N)
            ceil = torch.ceil(x[mask] * N)
            vertices = torch.zeros((x[mask].shape[0], 8, 3), dtype=torch.int64, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat((ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 7] = ceil

            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)
            looked_up = self.lookup_tables[str(i)][h_x].transpose(-1, -2)
            volume = looked_up.reshape((looked_up.shape[0], 2, 2, 2, 2))
            features[:, i*2:(i+1)*2] = torch.nn.functional.grid_sample(
                volume, ((x[mask] * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1),
                align_corners=False).squeeze(-1).squeeze(-1).squeeze(-1)

        xi = self.positional_encoding(d[mask])
        h = self.density_MLP(features)
        log_sigma[mask] = h[:, 0]
        color[mask] = self.color_MLP(torch.cat((h, xi), dim=1))
        return color, torch.exp(log_sigma)

# ==========================================
# 4. TRAINING LOOP (GPU SHUFFLING)
# ==========================================

def train(nerf_model, optimizer, scheduler, data, device='cuda', hn=0, hf=1, nb_epochs=5,
          nb_bins=256, H=800, W=800, batch_size=2**14, save_every=2000):
    loss_history = []
    iteration = 0
    data_size = data.shape[0]
    
    print(f"Training on {data_size} rays. Batch size: {batch_size}")
    
    for epoch in range(nb_epochs):
        # Generate random indices directly on GPU
        # This is incredibly fast compared to CPU shuffling
        indices = torch.randperm(data_size, device=device)
        
        pbar = tqdm(range(0, data_size, batch_size), desc=f"Epoch {epoch+1}/{nb_epochs}")
        
        for i in pbar:
            # Direct GPU slicing
            batch_indices = indices[i : i + batch_size]
            batch = data[batch_indices]
            
            ray_origins = batch[:, :3]
            ray_directions = batch[:, 3:6]
            gt_px_values = batch[:, 6:]
            
            pred_px_values = render_rays(nerf_model, ray_origins, ray_directions, 
                                         hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((gt_px_values - pred_px_values) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % 10 == 0:
                loss_history.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if iteration % save_every == 0 and iteration > 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(nerf_model.state_dict(), f'checkpoints/ngp_iter_{iteration}.pth')
            
            iteration += 1
        scheduler.step()
    
    return loss_history

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: CUDA not found. This code is optimized for GPU only.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    os.makedirs('novel_views', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    DATA_DIR = '/content/drive/MyDrive/Fundamentals_DL_and_AI/Customdata'
    DOWNSCALE = 4 
    
    # 1. Load Data (Everything happens on GPU)
    training_dataset, H, W, aabb_scale = load_nerfstudio_data(
        DATA_DIR, split='train', downscale=DOWNSCALE, test_holdout=8, device=device
    )
    testing_dataset, _, _, _ = load_nerfstudio_data(
        DATA_DIR, split='test', downscale=DOWNSCALE, test_holdout=8, device=device
    )
    
    # 2. Setup Model
    L, F, T = 16, 2, 2**19
    N_min, N_max = 16, 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    
    model = NGP(T, Nl, 4, device, aabb_scale).to(device)
    
    optimizer = torch.optim.Adam([
        {"params": model.lookup_tables.parameters(), "lr": 1e-2},
        {"params": model.density_MLP.parameters(), "lr": 1e-2},
        {"params": model.color_MLP.parameters(), "lr": 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    
    # 3. Train
    # Larger batch size since you have "enough memory"
    # 2^15 (32k rays) is usually safe on 16GB+ VRAM
    BATCH_SIZE = 2**15 
    print("Starting training...")
    loss_history = train(model, optimizer, scheduler, training_dataset, nb_epochs=5, device=device,
          hn=0.1, hf=15, nb_bins=196, H=H, W=W, batch_size=BATCH_SIZE, save_every=2000)
    
    torch.save(model.state_dict(), 'checkpoints/ngp_final.pth')
    
    # 4. Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved.")
    
    # 5. Test & Compare
    print("Rendering test views...")
    num_test_images = len(testing_dataset) // (H * W)
    for img_index in range(min(5, num_test_images)): 
        test_and_compare(model, 0.1, 15, testing_dataset, img_index, nb_bins=196, H=H, W=W, device=device)
        
    # 6. Extract Mesh
    extract_mesh(model, resolution=256, threshold=10.0, bound=1.5, device=device)
    print("All tasks completed.")