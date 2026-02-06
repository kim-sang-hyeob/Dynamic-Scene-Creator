"""
Convert 4DGS model to splaTV format.

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
from tqdm import tqdm
from argparse import ArgumentParser

# Add 4DGaussians to path
# Use current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state

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

def get_state_at_time(gaussians, time_value):
    """Get Gaussian state at a specific time value (0.0 to 1.0)"""
    with torch.no_grad():
        means3D = gaussians.get_xyz
        time = torch.tensor([[time_value]], dtype=torch.float32).to(means3D.device).repeat(means3D.shape[0], 1)
        opacity = gaussians._opacity
        shs = gaussians.get_features
        scales = gaussians._scaling
        rotations = gaussians._rotation
        
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = gaussians._deformation(
            means3D, scales, rotations, opacity, shs, time
        )
        
        return {
            'positions': means3D_final.cpu().numpy(),
            'scales': scales_final.cpu().numpy(),
            'rotations': rotations_final.cpu().numpy(),
            'opacity': opacity.cpu().numpy(),
            'shs': shs_final.cpu().numpy()
        }

def fit_motion_params(positions_over_time, times):
    """
    Fit polynomial coefficients for position change over time.
    position(t) = base_pos + motion_0*dt + motion_1*dt + motion_2*dt + 
                  motion_3*dt² + motion_4*dt² + motion_5*dt² +
                  motion_6*dt³ + motion_7*dt³ + motion_8*dt³
    
    Returns: base_position, motion_coeffs[9]
    """
    num_gaussians = positions_over_time[0].shape[0]
    num_times = len(times)
    
    # Use center time as reference
    center_time = 0.5
    base_idx = num_times // 2
    base_positions = positions_over_time[base_idx].copy()
    
    # Build design matrix for polynomial fitting
    # For each Gaussian, fit: delta_pos = a*dt + b*dt² + c*dt³
    motion_coeffs = np.zeros((num_gaussians, 9), dtype=np.float32)
    
    if num_times < 3:
        return base_positions, motion_coeffs, center_time
    
    # Compute deltas from center
    dts = np.array(times) - center_time
    
    # Design matrix: [dt, dt², dt³]
    A = np.column_stack([dts, dts**2, dts**3])
    
    # For each dimension (x, y, z)
    for dim in range(3):
        # Gather position deltas for all times
        deltas = np.zeros((num_times, num_gaussians), dtype=np.float32)
        for t_idx in range(num_times):
            deltas[t_idx] = positions_over_time[t_idx][:, dim] - base_positions[:, dim]
        
        # Solve least squares for each Gaussian
        # deltas = A @ coeffs
        coeffs, _, _, _ = np.linalg.lstsq(A, deltas, rcond=None)
        
        # coeffs shape: (3, num_gaussians) - [dt, dt², dt³]
        motion_coeffs[:, dim] = coeffs[0]      # dt coefficient for x/y/z
        motion_coeffs[:, dim + 3] = coeffs[1]  # dt² coefficient
        motion_coeffs[:, dim + 6] = coeffs[2]  # dt³ coefficient
    
    return base_positions, motion_coeffs, center_time

def fit_rotation_params(rotations_over_time, times):
    """
    Fit linear rotation delta over time.
    Returns: base_rotation, omega_coeffs[4]
    """
    num_gaussians = rotations_over_time[0].shape[0]
    num_times = len(times)
    
    center_time = 0.5
    base_idx = num_times // 2
    base_rotations = rotations_over_time[base_idx].copy()
    
    omega_coeffs = np.zeros((num_gaussians, 4), dtype=np.float32)
    
    if num_times < 2:
        return base_rotations, omega_coeffs
    
    # Simple linear fit for rotation delta
    dts = np.array(times) - center_time
    
    for dim in range(4):
        deltas = np.zeros((num_times, num_gaussians), dtype=np.float32)
        for t_idx in range(num_times):
            deltas[t_idx] = rotations_over_time[t_idx][:, dim] - base_rotations[:, dim]
        
        # Linear fit: delta = omega * dt
        if np.abs(dts).sum() > 0:
            omega_coeffs[:, dim] = np.sum(deltas * dts[:, None], axis=0) / np.sum(dts**2)
    
    return base_rotations, omega_coeffs

def convert_to_splatv(gaussians, output_path, num_samples=20):
    """Convert 4DGS model to splaTV format."""
    
    print(f"Sampling {num_samples} time points...")
    times = np.linspace(0, 1, num_samples)
    
    # Sample Gaussian states at multiple times
    states = []
    for t in tqdm(times, desc="Sampling"):
        state = get_state_at_time(gaussians, float(t))
        states.append(state)
    
    # Extract data
    positions_over_time = [s['positions'] for s in states]
    rotations_over_time = [s['rotations'] for s in states]
    
    num_gaussians = positions_over_time[0].shape[0]
    print(f"Fitting motion parameters for {num_gaussians} gaussians...")
    
    # Fit motion parameters
    base_positions, motion_coeffs, trbf_center = fit_motion_params(positions_over_time, times)
    base_rotations, omega_coeffs = fit_rotation_params(rotations_over_time, times)
    
    # Use center state for static properties
    center_state = states[len(states)//2]
    scales = np.exp(center_state['scales'])  # Convert from log scale
    opacity = sigmoid(center_state['opacity'].flatten())
    shs = center_state['shs'][:, 0, :]  # DC component only
    
    # Convert SH to RGB
    rgb = SH2RGB(shs)
    opacity_uint8 = (opacity * 255).astype(np.uint8)
    
    # Sort by importance (size * opacity)
    importance = np.prod(scales, axis=1) * opacity
    sort_indices = np.argsort(-importance)
    
    # Apply sorting
    base_positions = base_positions[sort_indices]
    base_rotations = base_rotations[sort_indices]
    scales = scales[sort_indices]
    rgb = rgb[sort_indices]
    opacity_uint8 = opacity_uint8[sort_indices]
    motion_coeffs = motion_coeffs[sort_indices]
    omega_coeffs = omega_coeffs[sort_indices]
    
    # Temporal scale (how spread out the motion is)
    trbf_scale = 0.5  # Default, covers most of the time range
    
    print("Building texture data...")
    
    # Build texture data (16 uint32 per gaussian = 64 bytes)
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
    
    # Create JSON metadata
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
    
    # Write splatv file
    with open(output_path, 'wb') as f:
        # Magic header
        f.write(struct.pack('<I', 0x674b))  # Magic number
        f.write(struct.pack('<I', len(json_bytes)))  # JSON length
        f.write(json_bytes)
        f.write(texdata.tobytes())
    
    file_size = os.path.getsize(output_path)
    print(f"Created {output_path} ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Gaussians: {num_gaussians}")
    print(f"  Time samples used: {num_samples}")

def main():
    parser = ArgumentParser(description="Convert 4DGS to splaTV format")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output", type=str, required=True, help="Output .splatv file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of time samples")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    
    args = get_combined_args(parser)
    print("Loading model from", args.model_path)
    
    if args.configs:
        from mmengine.config import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(args.quiet)
    
    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args))
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    
    convert_to_splatv(gaussians, args.output, args.num_samples)
    print("Done!")

if __name__ == "__main__":
    main()
