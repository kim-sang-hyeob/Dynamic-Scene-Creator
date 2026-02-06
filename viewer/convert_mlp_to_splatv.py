"""
Convert SC4D MLP-based 4DGS model to splaTV format.

SC4D uses an MLP (TimeNet) to predict position/rotation deltas based on
position and time. This script samples the MLP at multiple time points
and fits polynomial motion parameters for the splatv format.

splaTV format stores motion parameters per Gaussian:
- position: float32[3]
- rotation: float16[4] (quaternion)
- scale: float16[3]
- color: uint8[4] (RGBA from SH DC)
- motion_0~8: 9 coefficients for position delta (dt, dt², dt³ terms)
- omega_0~3: 4 coefficients for rotation delta
- trbf_center: temporal center
- trbf_scale: temporal scale
"""

import os
import sys
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData
from tqdm import tqdm
from argparse import ArgumentParser
from argparse import ArgumentParser
import pytorch3d.ops

# ==================== Math Helpers (Numpy) ====================
def rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def euler_to_quat_wxyz(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)

def q_mult_wxyz(q1, q2):
    # q1, q2 are [N, 4] or [4]
    # WXYZ convention
    if q1.ndim == 1: q1 = q1[None, :]
    if q2.ndim == 1: q2 = q2[None, :]
    
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([w, x, y, z], axis=1)


# ==================== Positional Encoding ====================
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs.cpu()).to(inputs.device) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    embed_kwargs = {
        'include_input': False,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# ==================== TimeNet (MLP) ====================
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.xavier_uniform_(m.weight, gain=1)


def initialize_weights_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_weights_one(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            m.bias.data = torch.tensor([1., 0., 0., 0.])


class TimeNet(nn.Module):
    def __init__(self, D=8, W=256, skips=[4], device="cuda"):
        super(TimeNet, self).__init__()
        self.pts_ch = 10
        self.times_ch = 6
        self.pts_emb_fn, pts_out_dims = get_embedder(self.pts_ch, 3)
        self.times_emb_fn, times_out_dims = get_embedder(self.times_ch, 1)
        self.input_ch = pts_out_dims + times_out_dims
        self.skips = skips
        self.deformnet = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        self.pts_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.rot_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
        self.device = device
        self.deformnet.apply(initialize_weights)
        self.pts_layers.apply(initialize_weights)
        self.rot_layers.apply(initialize_weights)
        self.pts_layers[-1].apply(initialize_weights_zero)
        self.rot_layers[-1].apply(initialize_weights_one)

    def forward(self, pts, t, nobatch=False, t_apply=False):
        if len(pts.shape) == 2:
            nobatch = True
            pts = pts.unsqueeze(0)
        if t_apply:
            times = t
            pts = pts.repeat(times.shape[0], 1, 1)
        else:
            times = torch.tensor([t])[:, None, None].repeat(1, pts.shape[1], 1).to(self.device)
        pts_emb = self.pts_emb_fn(pts)
        times_emb = self.times_emb_fn(times)
        pts_emb = torch.cat([pts_emb, times_emb], dim=-1)
        h = pts_emb
        for i, l in enumerate(self.deformnet):
            h = self.deformnet[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts_emb, h], dim=-1)
        pts_t, rot_t = self.pts_layers(h), self.rot_layers(h)
        if nobatch:
            pts_t, rot_t = pts_t[0], rot_t[0]
        return pts_t, rot_t


# ==================== Utility Functions ====================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def SH2RGB(sh):
    """Convert SH DC coefficient to RGB (0-255)."""
    C0 = 0.28209479177387814
    return np.clip((0.5 + C0 * sh) * 255, 0, 255).astype(np.uint8)


def pack_half2x16(x, y):
    """Pack two float16 values into a uint32."""
    x_half = np.float16(x)
    y_half = np.float16(y)
    x_bits = x_half.view(np.uint16)
    y_bits = y_half.view(np.uint16)
    return np.uint32(x_bits) | (np.uint32(y_bits) << 16)


def build_rotation(r):
    """Build rotation matrix from quaternion."""
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_rotation_3d(r):
    """Build rotation matrix from quaternion for batched input."""
    norm = torch.sqrt(r[:,:,0]*r[:,:,0] + r[:,:,1]*r[:,:,1] + r[:,:,2]*r[:,:,2] + r[:,:,3]*r[:,:,3])
    q = r / norm[:, :, None]

    R = torch.zeros((q.size(0), q.size(1), 3, 3), device=r.device)

    r = q[:, :, 0]
    x = q[:, :, 1]
    y = q[:, :, 2]
    z = q[:, :, 3]

    R[:, :, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, :, 0, 1] = 2 * (x*y - r*z)
    R[:, :, 0, 2] = 2 * (x*z + r*y)
    R[:, :, 1, 0] = 2 * (x*y + r*z)
    R[:, :, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, :, 1, 2] = 2 * (y*z - r*x)
    R[:, :, 2, 0] = 2 * (x*z - r*y)
    R[:, :, 2, 1] = 2 * (y*z + r*x)
    R[:, :, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def quat_mul(q1, q2):
    """Multiply two quaternions."""
    q = torch.zeros((q1.size(0), 4), device=q1.device)

    r1, r2 = q1[:, 0], q2[:, 0]
    x1, x2 = q1[:, 1], q2[:, 1]
    y1, y2 = q1[:, 2], q2[:, 2]
    z1, z2 = q1[:, 3], q2[:, 3]

    q[:, 0] = r1*r2 - x1*x2 - y1*y2 - z1*z2
    q[:, 1] = r1*x2 + x1*r2 + y1*z2 - z1*y2
    q[:, 2] = r1*y2 - x1*z2 + y1*r2 + z1*x2
    q[:, 3] = r1*z2 + x1*y2 - y1*x2 + z1*r2
    return q


# ==================== Data Loading ====================
def load_gaussian_ply(path):
    """Load Gaussian data from PLY file."""
    plydata = PlyData.read(path)

    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # SH DC coefficients
    f_dc_0 = np.asarray(plydata.elements[0]["f_dc_0"])
    f_dc_1 = np.asarray(plydata.elements[0]["f_dc_1"])
    f_dc_2 = np.asarray(plydata.elements[0]["f_dc_2"])
    features_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

    # Scale
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Rotation (quaternion)
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return {
        'xyz': xyz,
        'opacity': opacity,
        'features_dc': features_dc,
        'scales': scales,
        'rotations': rots
    }


def load_control_points_ply(path):
    """Load control points from PLY file."""
    plydata = PlyData.read(path)

    c_xyz = np.stack((
        np.asarray(plydata.elements[0]["c_x"]),
        np.asarray(plydata.elements[0]["c_y"]),
        np.asarray(plydata.elements[0]["c_z"])
    ), axis=1)

    c_radius = np.asarray(plydata.elements[0]["c_radius"])[..., np.newaxis]

    return {
        'c_xyz': c_xyz,
        'c_radius': c_radius
    }


# ==================== Deformation Computation ====================
def compute_neighbor_info(xyz, c_xyz, K=4):
    """Compute neighbor indices and distances for Gaussians to control points."""
    xyz_tensor = torch.from_numpy(xyz).float().cuda()[None]
    c_xyz_tensor = torch.from_numpy(c_xyz).float().cuda()[None]

    # K nearest control points for each Gaussian
    knn_res = pytorch3d.ops.knn_points(xyz_tensor, c_xyz_tensor, K=K)
    neighbor_dists = knn_res.dists[0].sqrt()  # [N, K]
    neighbor_indices = knn_res.idx[0]  # [N, K]

    return neighbor_dists, neighbor_indices


def get_deformed_state(timenet, xyz, c_xyz, c_radius, rotations, neighbor_dists, neighbor_indices, time_value, device='cuda'):
    """Get deformed Gaussian state at a specific time."""
    xyz_t = torch.from_numpy(xyz).float().to(device)
    c_xyz_t = torch.from_numpy(c_xyz).float().to(device)
    c_radius_t = torch.from_numpy(c_radius).float().to(device)
    rotations_t = torch.from_numpy(rotations).float().to(device)

    with torch.no_grad():
        # Get deformation for control points
        means3D_deform, rots_deform = timenet(c_xyz_t, time_value)

        # Interpolate deformation to Gaussians
        eps = 1e-7
        c_radius_exp = torch.exp(c_radius_t)
        c_radius_n = c_radius_exp[neighbor_indices]

        w = torch.exp(-1.0 * neighbor_dists**2 / (2. * (c_radius_n[:, :, 0]**2)))
        w = w + eps
        w = F.normalize(w, p=1, dim=1)

        # Neighbor deformations
        means3D_n = c_xyz_t[neighbor_indices]  # [N, K, 3]
        means3D_n_deform = means3D_deform[neighbor_indices]  # [N, K, 3]
        rots3D_n_deform = rots_deform[neighbor_indices]  # [N, K, 4]

        # Apply local frame deformation
        rot_matrices = build_rotation_3d(rots3D_n_deform)  # [N, K, 3, 3]
        local_offset = xyz_t[:, None] - means3D_n  # [N, K, 3]
        rotated_offset = torch.einsum('nkij,nkj->nki', rot_matrices, local_offset)

        pts3D = (w[..., None] * (rotated_offset + means3D_n + means3D_n_deform)).sum(dim=1)
        rots3D = (w[..., None] * rots3D_n_deform).sum(dim=1)

        # Apply rotation to base rotation
        final_rotation = quat_mul(rots3D, rotations_t)
        final_rotation = F.normalize(final_rotation, dim=1)

    return pts3D.cpu().numpy(), final_rotation.cpu().numpy()


# ==================== Motion Fitting ====================
def fit_motion_params(positions_over_time, times):
    """
    Fit polynomial coefficients for position change over time.
    position(t) = base_pos + motion_0*dt + motion_1*dt + motion_2*dt +
                  motion_3*dt² + motion_4*dt² + motion_5*dt² +
                  motion_6*dt³ + motion_7*dt³ + motion_8*dt³
    """
    num_gaussians = positions_over_time[0].shape[0]
    num_times = len(times)

    center_time = 0.5
    base_idx = num_times // 2
    base_positions = positions_over_time[base_idx].copy()

    motion_coeffs = np.zeros((num_gaussians, 9), dtype=np.float32)

    if num_times < 3:
        return base_positions, motion_coeffs, center_time

    dts = np.array(times) - center_time
    A = np.column_stack([dts, dts**2, dts**3])

    for dim in range(3):
        deltas = np.zeros((num_times, num_gaussians), dtype=np.float32)
        for t_idx in range(num_times):
            deltas[t_idx] = positions_over_time[t_idx][:, dim] - base_positions[:, dim]

        coeffs, _, _, _ = np.linalg.lstsq(A, deltas, rcond=None)

        motion_coeffs[:, dim] = coeffs[0]      # dt coefficient
        motion_coeffs[:, dim + 3] = coeffs[1]  # dt² coefficient
        motion_coeffs[:, dim + 6] = coeffs[2]  # dt³ coefficient

    return base_positions, motion_coeffs, center_time


def fit_rotation_params(rotations_over_time, times):
    """Fit linear rotation delta over time."""
    num_gaussians = rotations_over_time[0].shape[0]
    num_times = len(times)

    center_time = 0.5
    base_idx = num_times // 2
    base_rotations = rotations_over_time[base_idx].copy()

    omega_coeffs = np.zeros((num_gaussians, 4), dtype=np.float32)

    if num_times < 2:
        return base_rotations, omega_coeffs

    dts = np.array(times) - center_time

    for dim in range(4):
        deltas = np.zeros((num_times, num_gaussians), dtype=np.float32)
        for t_idx in range(num_times):
            deltas[t_idx] = rotations_over_time[t_idx][:, dim] - base_rotations[:, dim]

        if np.abs(dts).sum() > 0:
            omega_coeffs[:, dim] = np.sum(deltas * dts[:, None], axis=0) / np.sum(dts**2)

    return base_rotations, omega_coeffs


# ==================== SplatV Writer ====================
def write_splatv(output_path, base_positions, base_rotations, scales, rgb, opacity_uint8,
                 motion_coeffs, omega_coeffs, trbf_center, trbf_scale):
    """Write splatv file."""
    num_gaussians = base_positions.shape[0]

    # Sort by importance
    importance = np.prod(scales, axis=1) * (opacity_uint8 / 255.0)
    sort_indices = np.argsort(-importance)

    base_positions = base_positions[sort_indices]
    base_rotations = base_rotations[sort_indices]
    scales = scales[sort_indices]
    rgb = rgb[sort_indices]
    opacity_uint8 = opacity_uint8[sort_indices]
    motion_coeffs = motion_coeffs[sort_indices]
    omega_coeffs = omega_coeffs[sort_indices]

    print("Building texture data...")

    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * num_gaussians) / texwidth))
    texdata = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_c = texdata.view(np.uint8)

    for j in tqdm(range(num_gaussians), desc="Building texture"):
        # Row 0: position (xyz) + rotation0/1
        texdata_f[16 * j + 0] = base_positions[j, 0]
        texdata_f[16 * j + 1] = base_positions[j, 1]
        texdata_f[16 * j + 2] = base_positions[j, 2]
        texdata[16 * j + 3] = pack_half2x16(base_rotations[j, 0], base_rotations[j, 1])

        # Row 1: rotation2/3 + scale + rgba
        texdata[16 * j + 4] = pack_half2x16(base_rotations[j, 2], base_rotations[j, 3])
        texdata[16 * j + 5] = pack_half2x16(scales[j, 0], scales[j, 1])
        texdata[16 * j + 6] = pack_half2x16(scales[j, 2], 0)

        # RGBA
        texdata_c[4 * (16 * j + 7) + 0] = rgb[j, 0]
        texdata_c[4 * (16 * j + 7) + 1] = rgb[j, 1]
        texdata_c[4 * (16 * j + 7) + 2] = rgb[j, 2]
        texdata_c[4 * (16 * j + 7) + 3] = opacity_uint8[j]

        # Row 2: motion coefficients (9 values)
        texdata[16 * j + 8 + 0] = pack_half2x16(motion_coeffs[j, 0], motion_coeffs[j, 1])
        texdata[16 * j + 8 + 1] = pack_half2x16(motion_coeffs[j, 2], motion_coeffs[j, 3])
        texdata[16 * j + 8 + 2] = pack_half2x16(motion_coeffs[j, 4], motion_coeffs[j, 5])
        texdata[16 * j + 8 + 3] = pack_half2x16(motion_coeffs[j, 6], motion_coeffs[j, 7])
        texdata[16 * j + 8 + 4] = pack_half2x16(motion_coeffs[j, 8], 0)

        # Row 3: omega (rotation over time) + trbf
        texdata[16 * j + 8 + 5] = pack_half2x16(omega_coeffs[j, 0], omega_coeffs[j, 1])
        texdata[16 * j + 8 + 6] = pack_half2x16(omega_coeffs[j, 2], omega_coeffs[j, 3])
        texdata[16 * j + 8 + 7] = pack_half2x16(trbf_center, trbf_scale)

    print(f"Texture size: {texwidth}x{texheight}")

    metadata = [{
        "type": "splat",
        "size": texdata.nbytes,
        "texwidth": texwidth,
        "texheight": texheight,
        "cameras": [{
            "id": 0,
            "img_name": "00001",
            "width": 800,
            "height": 800,
            "position": [0, 0, -3],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "fy": 800,
            "fx": 800,
        }]
    }]

    json_bytes = json.dumps(metadata).encode('utf-8')

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', 0x674b))
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(json_bytes)
        f.write(texdata.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Created {output_path} ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Gaussians: {num_gaussians}")


# ==================== Main Conversion ====================
def convert_sc4d_to_splatv(model_dir, output_path, num_samples=30, iteration=None, rotate_args=None):
    """Convert SC4D MLP-based 4DGS to splatv format."""

    # Determine file paths
    if iteration:
        ply_path = os.path.join(model_dir, f"point_cloud_{iteration}.ply")
        c_ply_path = os.path.join(model_dir, f"point_cloud_c_{iteration}.ply")
        timenet_path = os.path.join(model_dir, f"timenet_{iteration}.pth")
    else:
        ply_path = os.path.join(model_dir, "point_cloud.ply")
        c_ply_path = os.path.join(model_dir, "point_cloud_c.ply")
        timenet_path = os.path.join(model_dir, "timenet.pth")

    print(f"Loading Gaussian PLY: {ply_path}")
    print(f"Loading Control Points PLY: {c_ply_path}")
    print(f"Loading TimeNet: {timenet_path}")

    # Load data
    gaussian_data = load_gaussian_ply(ply_path)
    control_data = load_control_points_ply(c_ply_path)

    xyz = gaussian_data['xyz']
    opacity = gaussian_data['opacity']
    features_dc = gaussian_data['features_dc']
    scales_log = gaussian_data['scales']
    rotations = gaussian_data['rotations']

    c_xyz = control_data['c_xyz']
    c_radius = control_data['c_radius']

    print(f"Loaded {xyz.shape[0]} Gaussians and {c_xyz.shape[0]} control points")

    # Load TimeNet
    timenet = TimeNet().cuda()
    timenet.load_state_dict(torch.load(timenet_path, map_location='cuda'))
    timenet.eval()

    # Compute neighbor info
    print("Computing neighbor information...")
    neighbor_dists, neighbor_indices = compute_neighbor_info(xyz, c_xyz, K=4)

    # Sample at multiple time points
    print(f"Sampling {num_samples} time points...")
    times = np.linspace(0, 1, num_samples)

    positions_over_time = []
    rotations_over_time = []

    for t in tqdm(times, desc="Sampling"):
        pos, rot = get_deformed_state(
            timenet, xyz, c_xyz, c_radius, rotations,
            neighbor_dists, neighbor_indices, float(t)
        )
        positions_over_time.append(pos)
        rotations_over_time.append(rot)

        rotations_over_time.append(rot)

    # Apply Global Rotation if requested
    if rotate_args is not None:
        print(f"Applying global rotation: {rotate_args}")
        rx, ry, rz = rotate_args
        R_mat = rotation_matrix(rx, ry, rz).astype(np.float32)
        q_global = euler_to_quat_wxyz(rx, ry, rz)
        
        # Rotate positions: P_new = P_old @ R_T
        # Rotate quaternions: Q_new = Q_global * Q_old
        for i in range(len(positions_over_time)):
            positions_over_time[i] = positions_over_time[i] @ R_mat.T
            
            # Rotations are [N, 4] wxyz
            rotations_over_time[i] = q_mult_wxyz(q_global, rotations_over_time[i])

    # Fit motion parameters
    print("Fitting motion parameters...")
    base_positions, motion_coeffs, trbf_center = fit_motion_params(positions_over_time, times)
    base_rotations, omega_coeffs = fit_rotation_params(rotations_over_time, times)

    # Prepare other properties
    scales = np.exp(scales_log)
    rgb = SH2RGB(features_dc)
    opacity_values = sigmoid(opacity.flatten())
    opacity_uint8 = (opacity_values * 255).astype(np.uint8)

    trbf_scale = 0.5

    # Write output
    write_splatv(
        output_path, base_positions, base_rotations, scales, rgb, opacity_uint8,
        motion_coeffs, omega_coeffs, trbf_center, trbf_scale
    )

    print("Conversion complete!")


def main():
    parser = ArgumentParser(description="Convert SC4D MLP-based 4DGS to splaTV format")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing point_cloud.ply, point_cloud_c.ply, timenet.pth")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .splatv file path")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of time samples for fitting (default: 30)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Specific iteration to load (e.g., 8000 for point_cloud_8000.ply)")
    parser.add_argument("--rotate", nargs=3, type=float, default=None, 
                        help="Rotation degrees (x y z) to apply to the model")

    args = parser.parse_args()

    convert_sc4d_to_splatv(
        args.model_dir,
        args.output,
        args.num_samples,
        args.iteration,
        args.rotate
    )


if __name__ == "__main__":
    main()
