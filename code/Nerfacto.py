import torch
import numpy as np
import os
import imageio
from tqdm import tqdm
from PIL import Image
import torch.nn as nn

# ==========================================
# 1. MODEL DEFINITION (Must match training)
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
# 2. RENDERING LOGIC
# ==========================================

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=256):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    
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
    
    return c + 1 - weight_sum.unsqueeze(-1)

def get_camera_rays(c2w, H, W, fov_x=60, device='cuda'):
    focal = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_x))
    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def generate_360_path(frames=120, elevation=30, radius=4.0, device='cuda'):
    c2ws = []
    for i in range(frames):
        theta = 2 * np.pi * i / frames
        phi = np.deg2rad(90 - elevation)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        cam_pos = torch.tensor([x, y, z], device=device).float()
        
        forward = -cam_pos / torch.norm(cam_pos)
        up = torch.tensor([0, 0, 1], device=device).float()
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        c2w = torch.eye(4, device=device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        c2ws.append(c2w)
    return c2ws

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ---------------- CONFIG ----------------
    CHECKPOINTS = [
        '/content/ngp_final (2).pth'
    ]
    OUTPUT_FOLDER = 'videos'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    W, H = 800, 800
    FRAMES = 120
    RADIUS = 2.5
    ELEVATION = 20
    FOV = 60
    
    # Model Setup
    L, F, T = 16, 2, 2**19
    N_min, N_max = 16, 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    aabb_scale = 16 
    
    model = NGP(T, Nl, 4, device, aabb_scale).to(device)

    poses = generate_360_path(frames=FRAMES, elevation=ELEVATION, radius=RADIUS, device=device)
    
    # SAFE CHUNK SIZE (Drastically reduced)
    # 8192 rays is very safe for 8GB+ VRAM cards
    SAFE_CHUNK_SIZE = 8192 

    for ckpt_path in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path}, not found.")
            continue
            
        print(f"\nProcessing {ckpt_path}...")
        
        # Load weights
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            continue
            
        model.eval()
        
        video_frames = []
        
        # tqdm wrapper
        pbar = tqdm(poses, desc="Rendering Frames")
        
        for c2w in pbar:
            rays_o, rays_d = get_camera_rays(c2w, H, W, fov_x=FOV, device=device)
            
            px_values = []
            
            # Inner loop over rays in chunks
            with torch.no_grad():
                for k in range(0, rays_o.shape[0], SAFE_CHUNK_SIZE):
                    ro = rays_o[k:k+SAFE_CHUNK_SIZE]
                    rd = rays_d[k:k+SAFE_CHUNK_SIZE]
                    
                    # Render chunk
                    val = render_rays(model, ro, rd, hn=0.1, hf=15, nb_bins=196)
                    px_values.append(val)
            
            img = torch.cat(px_values).data.cpu().numpy().reshape(H, W, 3)
            img = (img.clip(0, 1) * 255.).astype(np.uint8)
            video_frames.append(img)
            
            # Clear cache occasionally to prevent fragmentation
            torch.cuda.empty_cache()
            
        # Save Video
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        save_path = f"{OUTPUT_FOLDER}/360_{ckpt_name}.mp4"
        imageio.mimwrite(save_path, video_frames, fps=30, quality=8)
        print(f"Video saved to {save_path}")

    print("\nDone.")