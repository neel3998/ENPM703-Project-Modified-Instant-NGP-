import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm
from skimage import measure

def save_obj(vertices, faces, colors, filename):
    """
    Saves vertices and faces to a Wavefront .obj file with vertex colors.
    """
    print(f"Saving mesh to {filename}...")
    with open(filename, 'w') as f:
        f.write(f"# Mesh extracted from NGP\n")
        
        # Write vertices with colors (v x y z r g b)
        # OBJ standard supports vertex colors (though not all viewers display them)
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            
        # Write faces (f v1 v2 v3)
        # OBJ indices are 1-based
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print("Done.")

@torch.no_grad()
def extract_mesh(model, resolution=256, threshold=10.0, bound=1.5, device='cuda'):
    """
    Extracts a mesh using Marching Cubes.
    
    Args:
        resolution: Grid resolution (e.g., 256 or 512). Higher = more detail but more RAM.
        threshold: Density threshold. Higher = cuts out more 'fog'. Lower = fuller mesh.
        bound: The physical size of the cube to scan (e.g., -1.5 to 1.5).
    """
    print(f"Extracting mesh with resolution {resolution}^3 and density threshold {threshold}...")
    
    # 1. Create a grid of points
    X = torch.linspace(-bound, bound, resolution).to(device)
    Y = torch.linspace(-bound, bound, resolution).to(device)
    Z = torch.linspace(-bound, bound, resolution).to(device)
    
    # Create 3D grid
    grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    # 2. Query the model for density (sigma) in chunks
    chunk_size = 64 * 64 * 64
    sigma_grid = []
    
    # We use a dummy direction because density is view-independent, 
    # but the model signature requires directions.
    dummy_dirs = torch.zeros_like(points) 
    
    print("Querying density field...")
    for i in tqdm(range(0, points.shape[0], chunk_size)):
        chunk_pts = points[i:i+chunk_size]
        chunk_dirs = dummy_dirs[i:i+chunk_size]
        
        # We only need sigma here
        _, sigma = model(chunk_pts, chunk_dirs)
        sigma_grid.append(sigma.cpu())
    
    sigma_grid = torch.cat(sigma_grid).numpy().reshape(resolution, resolution, resolution)
    
    # 3. Run Marching Cubes
    # sigma_grid contains density. We find the isosurface where density == threshold.
    try:
        vertices, faces, normals, values = measure.marching_cubes(sigma_grid, level=threshold)
    except RuntimeError as e:
        print(f"Failed to extract mesh: {e}")
        print("Try lowering the 'threshold' value.")
        return

    # 4. Convert grid coordinates back to world coordinates
    # Vertices returned by marching_cubes are in index coordinates [0, resolution-1]
    # We transform them back to [-bound, bound]
    vertices = vertices / (resolution - 1) * (2 * bound) - bound
    
    # 5. Query colors for the vertices
    # To color the mesh, we query the NeRF at the vertex locations.
    print("Querying vertex colors...")
    vertex_colors = []
    vertices_tensor = torch.from_numpy(vertices).float().to(device)
    
    # We view the color from a standard direction (e.g., facing negative Z)
    # or simply use zeros to get the "diffuse" color if the model relies heavily on position.
    vertex_dirs = torch.zeros_like(vertices_tensor)
    vertex_dirs[..., 2] = -1.0 # arbitrary view direction
    
    # Query in chunks again to avoid OOM
    chunk_size = 100000
    for i in tqdm(range(0, vertices_tensor.shape[0], chunk_size)):
        chunk_pts = vertices_tensor[i:i+chunk_size]
        chunk_dirs = vertex_dirs[i:i+chunk_size]
        colors, _ = model(chunk_pts, chunk_dirs)
        vertex_colors.append(colors.cpu())
        
    vertex_colors = torch.cat(vertex_colors).numpy()
    
    # 6. Save
    os.makedirs('meshes', exist_ok=True)
    save_obj(vertices, faces, vertex_colors, f'meshes/mesh_res{resolution}_thresh{threshold}.obj')



# ==========================================
# 1. HELPER FUNCTIONS & RENDERING
# ==========================================

@torch.no_grad()
def test(model, hn, hf, dataset, img_index, chunk_size=10, nb_bins=256, H=800, W=800, device='cuda'):
    # Extract rays for the specific image index
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    px_values = []
    # Render in chunks to avoid OOM
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        px_values.append(render_rays(model, ray_origins_, ray_directions_,
                                     hn=hn, hf=hf, nb_bins=nb_bins))
    
    img = torch.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
    img = (img.clip(0, 1)*255.).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f'novel_views/img_{img_index}.png')
    return img

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=256):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    
    # Stratified sampling
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
    
    # Background composition (assume white background for dataset consistency, though typically black for real scenes)
    # Using black background for real world data usually works better unless masked
    return c + 1 - weight_sum.unsqueeze(-1) # White background
    # return c # Black background (Uncomment if results look washed out)

# ==========================================
# 2. DATA LOADER (MODIFIED FOR NERFSTUDIO)
# ==========================================

def load_nerfstudio_data(data_dir, split='train', downscale=4, test_holdout=8):
    """
    Loads Nerfstudio/Colmap style data from a single transforms.json
    """
    json_path = os.path.join(data_dir, 'transforms.json')
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    # 1. Get image dimensions and intrinsics
    W = int(meta['w'])
    H = int(meta['h'])
    
    fl_x = meta['fl_x']
    fl_y = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']
    
    # 2. Adjust for downscaling
    W = W // downscale
    H = H // downscale
    fl_x = fl_x / downscale
    fl_y = fl_y / downscale
    cx = cx / downscale
    cy = cy / downscale
    
    # 3. Filter frames based on split
    all_frames = meta['frames']
    selected_frames = []
    
    if split == 'train':
        selected_frames = [f for i, f in enumerate(all_frames) if i % test_holdout != 0]
    else: # test/val
        selected_frames = [f for i, f in enumerate(all_frames) if i % test_holdout == 0]

    print(f"Loading {split} data: {len(selected_frames)} images (Downscale factor: {downscale})")
    print(f"Resolution: {W}x{H}")

    all_rays = []
    
    for frame in tqdm(selected_frames):
        # Handle file paths (remove leading ./ if present and join with data_dir)
        fname = frame['file_path']
        if fname.startswith('./'):
            fname = fname[2:]
        img_path = os.path.join(data_dir, fname)
        
        # Open and resize image
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}")
            continue
            
        img = Image.open(img_path)
        img = img.resize((W, H), Image.LANCZOS)
        img = np.array(img) / 255.0
        
        # Handle Alpha Channel (JPGs usually don't have it, but just in case)
        if img.shape[-1] == 4:
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:]) # Blend with white
        else:
            img = img[..., :3] # Keep RGB
            
        # Parse Pose
        c2w = np.array(frame['transform_matrix'])
        
        # Generate Rays
        # Create grid of coordinates
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        
        # Pinhole Camera Model with explicit Principal Point
        # OpenGL coordinate system: Y up, -Z forward. 
        # Pixel (0,0) is top-left.
        dirs = np.stack([(i - cx) / fl_x, -(j - cy) / fl_y, -np.ones_like(i)], -1)
        
        # Rotate ray directions by camera pose
        rays_d = np.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        # Ray origins are camera position
        rays_o = np.broadcast_to(c2w[:3,-1], rays_d.shape)
        
        # Flatten and concat [Origins, Directions, RGB]
        rays = np.concatenate([rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), img.reshape(-1, 3)], axis=-1)
        all_rays.append(rays)
        
    if len(all_rays) == 0:
        raise ValueError("No rays loaded! Check file paths.")
        
    return torch.from_numpy(np.concatenate(all_rays, axis=0)).float(), H, W, meta.get('aabb_scale', 16)

# ==========================================
# 3. MODEL ARCHITECTURE (NGP)
# ==========================================

class NGP(torch.nn.Module):
    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(NGP, self).__init__()
        self.T = T
        self.Nl = Nl
        self.F = F
        self.L = L
        self.aabb_scale = aabb_scale
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T, 2), device=device) * 2 - 1) * 1e-4) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861
        
        self.density_MLP = nn.Sequential(
            nn.Linear(self.F * len(Nl), 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        ).to(device)
        
        self.color_MLP = nn.Sequential(
            nn.Linear(27 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
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
        
        if mask.sum() == 0:
            return color, torch.exp(log_sigma)
            
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
        for i, N in enumerate(self.Nl):
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
                volume,
                ((x[mask] * N - floor) - 0.5).unsqueeze(1).unsqueeze(1).unsqueeze(1),
                align_corners=False
                ).squeeze(-1).squeeze(-1).squeeze(-1)

        xi = self.positional_encoding(d[mask])
        h = self.density_MLP(features)
        log_sigma[mask] = h[:, 0]
        color[mask] = self.color_MLP(torch.cat((h, xi), dim=1))
        return color, torch.exp(log_sigma)

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=5,
          nb_bins=256, H=800, W=800, save_every=1000):
    iteration = 0
    for epoch in range(nb_epochs):
        print(f"\nEpoch {epoch+1}/{nb_epochs}")
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            gt_px_values = batch[:, 6:].to(device)
            
            pred_px_values = render_rays(nerf_model, ray_origins, ray_directions, 
                                         hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((gt_px_values - pred_px_values) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % save_every == 0 and iteration > 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(nerf_model.state_dict(), f'checkpoints/ngp_iter_{iteration}.pth')
            
            iteration += 1
        scheduler.step()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('novel_views', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Path to your dataset
    DATA_DIR = '/content/drive/MyDrive/Fundamentals_DL_and_AI/Nerfstudio_gen_bb_dataset'
    
    # DOWNSCALE is critical here. 1920x1080 is too big for training efficiently.
    # Downscale 4 means approx 480x270 resolution.
    DOWNSCALE = 4 
    
    # Load Training Data
    training_dataset, H, W, aabb_scale = load_nerfstudio_data(
        DATA_DIR, split='train', downscale=DOWNSCALE, test_holdout=8
    )
    
    # Load Test Data (Every 8th image)
    testing_dataset, _, _, _ = load_nerfstudio_data(
        DATA_DIR, split='test', downscale=DOWNSCALE, test_holdout=8
    )
    
    print(f"\nData Loaded.")
    print(f"AABB Scale from JSON: {aabb_scale}")
    
    # Model configuration
    L = 16
    F = 2
    T = 2**19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    
    # Use the AABB scale from the JSON, or default to 16
    model = NGP(T, Nl, 4, device, aabb_scale).to(device)
    
    # Optimizer
    model_optimizer = torch.optim.Adam([
        {"params": model.lookup_tables.parameters(), "lr": 1e-2},
        {"params": model.density_MLP.parameters(), "lr": 1e-2},
        {"params": model.color_MLP.parameters(), "lr": 1e-2}
    ])
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[5, 10], gamma=0.5)
    
    # Batch size (reduced slightly just in case)
    BATCH_SIZE = 2**14 
    data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training
    # Note: real world data usually requires a larger 'hf' (far plane) compared to synthetic Lego.
    # hn=0.1 (near), hf=15 (far) is a safe starting point for real scenes.
    print("Starting training...")
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=15, device=device,
          hn=0.1, hf=15, nb_bins=196, H=H, W=W, save_every=2000)
    
    # Save Final Model
    torch.save(model.state_dict(), 'checkpoints/ngp_final.pth')
    
    # Testing
    print("Rendering test views...")
    num_test_images = len(testing_dataset) // (H * W)
    
    for img_index in range(min(5, num_test_images)): # Render first 5 test images
        print(f"Rendering image {img_index}...")
        test(model, 0.1, 15, testing_dataset, img_index, nb_bins=196, H=H, W=W, device=device)
        
    print("Done! Check 'novel_views' folder.")


# ==========================================
# RUN EXTRACTION
# ==========================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Re-initialize the model structure EXACTLY as it was trained
    # (You must match these params to your training script)
    L = 16
    F = 2
    T = 2**19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    
    # IMPORTANT: Ensure this matches the aabb_scale from your transforms.json 
    # (It was 16 in the json you provided previously)
    aabb_scale = 16 
    
    model = NGP(T, Nl, 4, device, aabb_scale).to(device)
    
    # 2. Load the checkpoint
    checkpoint_path = 'checkpoints/ngp_final.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval() # Set to eval mode
        
        # 3. Extract Mesh
        # Parameters to tweak:
        # resolution: 256 is fast, 512 is high detail (requires more RAM)
        # threshold: If the mesh is empty, LOWER this (e.g. to 5.0 or 0.1). 
        #            If the mesh has too much noise/floating blobs, RAISE this (e.g. to 20.0).
        extract_mesh(model, resolution=256, threshold=10.0, bound=1.5, device=device)
    else:
        print(f"Checkpoint {checkpoint_path} not found. Train the model first.")