#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import pickle
import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from omegaconf import OmegaConf, open_dict
import hydra
from tqdm import tqdm
import gym

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.wall.wall_env_wrapper import WallEnvWrapper, DEFAULT_CFG
from models.visual_world_model import VWorldModel
from preprocessor import Preprocessor
from custom_resolvers import replace_slash
from utils import seed, cfg_to_dict

log = logging.getLogger(__name__)

def get_env_state_ranges(env):
    """Extract the valid state coordinate ranges from the environment."""
    # Calculate the valid coordinate ranges based on environment configuration
    # This uses the same logic as generate_random_state method
    
    start_min_x = env.border_padding
    start_max_x = env.left_wall_x.item() - env.padding
    target_min_x = env.right_wall_x.item() + env.padding
    target_max_x = env.img_size - 1 - env.border_padding
    
    min_y = env.border_padding
    max_y = env.img_size - 1 - env.border_padding
    
    # Combine left and right sides for full x range
    x_min = start_min_x
    x_max = target_max_x
    
    return np.array([
        [x_min, x_max],  # x range
        [min_y, max_y]   # y range  
    ])

def load_ckpt(snapshot_path, device):
    """Load model checkpoint from file."""
    print("loading model checkpoint from", snapshot_path)
    print(f"device: {device}")
    
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    
    ALL_MODEL_KEYS = [
        "encoder", "predictor", "decoder", "proprio_encoder", 
        "action_encoder", "action_quantizer", "state_quantizer"
    ]
    
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        print(f"Loading key: {k}, type: {type(v)}")
        if k == "encoder":
            # Reconstructing the encoder from its saved state
            if isinstance(v, dict) and "class_name" in v and "module_name" in v:
                encoder_class = hydra.utils.get_class(v["module_name"] + "." + v["class_name"])
                print(f"Loading encoder class: {v['class_name']} from module {v['module_name']}")
                encoder = encoder_class(**v["init_args"])
                encoder.load_state_dict(v["state_dict"])
                v = encoder
            elif not isinstance(v, dict):
                # Encoder is already loaded as an object
                print(f"Encoder already loaded as object: {type(v)}")
                pass
            else:
                # Raise an error if the encoder class is not found
                raise ValueError(f"Encoder class {v.get('class_name', 'Unknown')} not found in module {v.get('module_name', 'Unknown')}")
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, device):
    """Load complete model from checkpoint."""
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "proprio_encoder" not in result:
        result["proprio_encoder"] = None

    # For visualization, we don't need predictor, but we'll handle it gracefully
    if "predictor" not in result:
        result["predictor"] = None
    
    if "action_encoder" not in result:
        result["action_encoder"] = None
    
    if "state_quantizer" not in result or "action_quantizer" not in result:
        # No quantization
        result["state_quantizer"] = None
        result["action_quantizer"] = None

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    # Get num_action_repeat from train_cfg (following plan.py pattern)
    num_action_repeat = getattr(train_cfg, 'num_action_repeat', 1)

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        state_quantizer=result["state_quantizer"],
        action_quantizer=result["action_quantizer"],
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
    )
    model.to(device)
    return model


def generate_position_grid(x_range, y_range, grid_size=20):
    """Generate a grid of positions within the given ranges."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_min, y_max, grid_size)
    
    positions = []
    for x in x_coords:
        for y in y_coords:
            positions.append([x, y])
    
    return np.array(positions), (x_coords, y_coords)


def get_observation_at_position(env, position, device):
    """Get visual observation at a specific position."""
    position_tensor = torch.tensor(position, dtype=torch.float32).to(device)
    
    # Use the custom prepare method instead of standard reset for wall environment
    if hasattr(env, 'prepare'):
        # For environments like WallEnvWrapper that have custom prepare method
        env.seed(42)  # Use a fixed seed for consistency
        env.set_init_state(position_tensor)
        obs, _ = env.reset()
        # Apply the same transform as used during training
        if hasattr(env, 'transform') and env.transform is not None:
            obs['visual'] = env.transform(obs['visual'])
            # Convert from (C, H, W) to (H, W, C) to match training format
            if obs['visual'].dim() == 3 and obs['visual'].shape[0] <= 4:  # Assume C, H, W if first dim is small
                obs['visual'] = obs['visual'].permute(1, 2, 0)
    else:
        # Standard gym environment reset
        obs, _ = env.reset()
        if isinstance(obs, dict) and 'visual' in obs:
            # Apply any necessary transforms to match training format
            if obs['visual'].dim() == 3 and obs['visual'].shape[0] <= 4:
                obs['visual'] = obs['visual'].permute(1, 2, 0)
    
    return obs


def encode_observation(model, obs, device):
    """Encode a visual observation using the model's encoder."""
    model.eval()
    with torch.no_grad():
        # Prepare visual input - ensure it's the right shape and on the right device
        visual = obs['visual'].to(device).unsqueeze(0)  # Add batch dimension
        # Encode using the encoder
        visual = rearrange(visual, 'b h w c -> b c h w')
        embedding = model.encoder(visual)
        # Handle different encoder outputs
        if embedding.dim() == 3: # patch-based encoders
            # we restore the width and height dimensions
            embedding = embedding.unsqueeze(-3)
        # flatten the embedding to 1D
        embedding_flat = rearrange(embedding, 'b c h w -> b (c h w)')
        return embedding_flat


def compute_embedding_distances(reference_embedding, grid_embeddings, distance_metric='euclidean'):
    """Compute distances between reference embedding and grid embeddings."""
    if distance_metric == 'cosine':
        # Normalize embeddings for cosine distance
        ref_norm = F.normalize(reference_embedding.unsqueeze(0), dim=1)
        grid_norm = F.normalize(grid_embeddings, dim=1)
        # Cosine similarity -> cosine distance
        similarities = torch.mm(grid_norm, ref_norm.t()).squeeze()
        distances = 1 - similarities
    elif distance_metric == 'euclidean':
        distances = torch.norm(grid_embeddings - reference_embedding, dim=1)
    elif distance_metric == 'l1':
        distances = torch.norm(grid_embeddings - reference_embedding, p=1, dim=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    return distances.cpu().numpy()


def generate_reference_positions(state_ranges, num_positions=10, strategy='grid', seed=42):
    """Generate multiple reference positions for visualization."""
    x_range, y_range = state_ranges
    
    if strategy == 'grid':
        # Create a roughly square grid of reference positions
        grid_dim = int(np.ceil(np.sqrt(num_positions)))
        x_refs = np.linspace(x_range[0], x_range[1], grid_dim)
        y_refs = np.linspace(y_range[0], y_range[1], grid_dim)
        
        positions = []
        for i, x in enumerate(x_refs):
            for j, y in enumerate(y_refs):
                if len(positions) < num_positions:
                    positions.append([x, y])
        return np.array(positions[:num_positions])
    
    elif strategy == 'random':
        # Generate random positions within valid ranges
        np.random.seed(seed)  # Use provided seed
        x_positions = np.random.uniform(x_range[0], x_range[1], num_positions)
        y_positions = np.random.uniform(y_range[0], y_range[1], num_positions)
        return np.array(list(zip(x_positions, y_positions)))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def visualize_multiple_embedding_distances(positions, all_distances, x_coords, y_coords, 
                                         reference_positions, env, device, save_path, 
                                         output_dir, title="Multi-Reference Embedding Distances"):
    """Create a visualization with multiple reference positions and their distance heatmaps."""
    
    num_refs = len(reference_positions)
    grid_size = len(x_coords)
    
    # Create figure with subplots - 2 columns, num_refs rows
    fig, axes = plt.subplots(num_refs, 2, figsize=(12, 4 * num_refs))
    
    # Handle case where num_refs = 1 (axes won't be 2D)
    if num_refs == 1:
        axes = axes.reshape(1, -1)
    
    for i, (ref_pos, distances) in enumerate(zip(reference_positions, all_distances)):
        # Reshape distances to grid
        distance_grid = distances.reshape(grid_size, grid_size).T
        
        # Plot 1: Reference environment image
        ref_obs = get_observation_at_position(env, ref_pos, device)
        ref_visual = ref_obs['visual'].cpu().numpy()
        if ref_visual.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            ref_visual = ref_visual.transpose(1, 2, 0)
        axes[i, 0].imshow(ref_visual.astype(np.uint8))
        axes[i, 0].set_title(f'Reference {i+1}\nPos ({ref_pos[0]:.2f}, {ref_pos[1]:.2f})')
        axes[i, 0].axis('off')
        
        # Plot 2: Distance heatmap with flipped y-axis extent to match image coordinates
        im = axes[i, 1].imshow(distance_grid, 
                              extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], 
                              cmap='Reds', alpha=1.0, origin='upper')
        
        # Add reference point
        axes[i, 1].plot(ref_pos[0], ref_pos[1], 'r*', markersize=15, 
                       markeredgecolor='white', markeredgewidth=1)
        
        axes[i, 1].set_title(f'Distance Heatmap {i+1}')
        axes[i, 1].set_xlabel('X Position')
        axes[i, 1].set_ylabel('Y Position')
        
        # Add colorbar for each distance heatmap
        cbar = plt.colorbar(im, ax=axes[i, 1])
        cbar.set_label('Distance')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    final_save_path = os.path.join(output_dir, save_path)
    plt.savefig(final_save_path, dpi=150, bbox_inches='tight')
    log.info(f"Multi-reference visualization saved to {final_save_path}")
    
    plt.close()  # Close the figure to free memory


def visualization_main(cfg_dict):
    """Main visualization function that can be called programmatically."""
    
    # Set random seed
    seed(cfg_dict["seed"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Construct model path from ckpt_base_path and model_name (following plan.py pattern)
    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/{cfg_dict['model_name']}/"
    model_path = Path(model_path)
    
    # Load model configuration
    config_path = model_path / "hydra.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        train_cfg = OmegaConf.load(f)
    
    # Load model checkpoint
    model_ckpt = model_path / "checkpoints" / f"model_{cfg_dict['epoch']}.pth"
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")
    
    log.info("Loading model...")
    model = load_model(model_ckpt, train_cfg, device)
    
    # Load dataset to get proper preprocessing and configuration (like plan.py)
    log.info("Loading dataset...")
    _, dset = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset = dset["valid"]
    
    # Create environment using the same pattern as plan.py
    log.info("Setting up environment...")
    
    # Create environment directly without gym wrappers to avoid API conflicts
    # This matches how plan.py creates environments in the vector env constructors
    if train_cfg.env.name == "wall":
        from env.wall.wall_env_wrapper import WallEnvWrapper
        env = WallEnvWrapper(*train_cfg.env.args, **train_cfg.env.kwargs)
    elif train_cfg.env.name == "pusht":
        from env.pusht.pusht_wrapper import PushTWrapper
        env = PushTWrapper(*train_cfg.env.args, **train_cfg.env.kwargs)
    else:
        # For other environments, create them directly if possible, otherwise use gym.make
        try:
            # Try to create directly first
            env_class = hydra.utils.get_class(f"env.{train_cfg.env.name}.{train_cfg.env.name}_wrapper")
            env = env_class(*train_cfg.env.args, **train_cfg.env.kwargs)
        except:
            # Fall back to gym.make
            env = gym.make(train_cfg.env.name, *train_cfg.env.args, **train_cfg.env.kwargs)
    log.info(f"Created environment: {train_cfg.env.name}")
    
    # Get state ranges from environment
    state_ranges = get_env_state_ranges(env)
    log.info(f"Environment state ranges: x=[{state_ranges[0,0]:.2f}, {state_ranges[0,1]:.2f}], y=[{state_ranges[1,0]:.2f}, {state_ranges[1,1]:.2f}]")
    
    # Generate position grid
    log.info(f"Generating {cfg_dict['grid_size']}x{cfg_dict['grid_size']} position grid...")
    positions, (x_coords, y_coords) = generate_position_grid(
        state_ranges[0], state_ranges[1], cfg_dict["grid_size"]
    )
    
    # Generate reference positions
    if cfg_dict.get("use_single_reference", False):
        # Use single reference position (backward compatibility)
        reference_positions = [np.array(cfg_dict["reference_pos"])]
        log.info(f"Using single reference position: [{reference_positions[0][0]:.2f}, {reference_positions[0][1]:.2f}]")
    else:
        # Generate multiple reference positions
        num_refs = cfg_dict.get("num_reference_positions", 10)
        ref_strategy = cfg_dict.get("reference_strategy", "grid")
        reference_positions = generate_reference_positions(state_ranges, num_refs, ref_strategy, cfg_dict["seed"])
        log.info(f"Generated {len(reference_positions)} reference positions using '{ref_strategy}' strategy")
    
    # Compute embeddings for all grid positions (done once)
    log.info("Computing embeddings for grid positions...")
    grid_embeddings = []
    
    for i, pos in enumerate(tqdm(positions, desc="Encoding positions", unit="pos")):
        obs = get_observation_at_position(env, pos, device)
        embedding = encode_observation(model, obs, device)
        grid_embeddings.append(embedding)
    
    grid_embeddings = torch.cat(grid_embeddings, dim=0)
    log.info(f"Grid embeddings shape: {grid_embeddings.shape}")
    
    # Compute distances for each reference position
    all_distances = []
    all_distance_stats = []
    
    log.info(f"Computing {cfg_dict['distance_metric']} distances for {len(reference_positions)} reference positions...")
    for i, ref_pos in enumerate(tqdm(reference_positions, desc="Processing references", unit="ref")):
        # Get reference embedding
        ref_obs = get_observation_at_position(env, ref_pos, device)
        reference_embedding = encode_observation(model, ref_obs, device)
        
        # Compute distances
        distances = compute_embedding_distances(reference_embedding, grid_embeddings, cfg_dict["distance_metric"])
        all_distances.append(distances)
        
        # Collect statistics
        stats = {
            "min": distances.min(),
            "max": distances.max(),
            "mean": distances.mean(),
            "std": distances.std()
        }
        all_distance_stats.append(stats)
        
        log.info(f"Reference {i+1} distance stats - Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Set default save filename if not provided
    if cfg_dict["save_path"] is None:
        if len(reference_positions) == 1:
            ref_pos = reference_positions[0]
            save_filename = f"embedding_distances_ref_{ref_pos[0]:.1f}_{ref_pos[1]:.1f}_{cfg_dict['distance_metric']}.png"
        else:
            save_filename = f"embedding_distances_multi_ref_{len(reference_positions)}_{cfg_dict['distance_metric']}.png"
    else:
        save_filename = cfg_dict["save_path"]
    
    # Use current working directory (Hydra has already set this correctly)
    output_dir = os.getcwd()
    
    # Create visualization
    log.info("Creating multi-reference visualization...")
    title = f"Multi-Reference Embedding Distances ({cfg_dict['distance_metric'].capitalize()})"
    
    # Always use multi-reference visualization (works for single reference too)
    visualize_multiple_embedding_distances(
        positions, all_distances, x_coords, y_coords, reference_positions,
        env, device, save_filename, output_dir, title
    )
    
    print(f"output_dir: {output_dir}")
    print(f"save_filename: {save_filename}")
    return {
        "output_path": os.path.join(output_dir, save_filename),
        "reference_positions": reference_positions.tolist(),
        "distance_stats": all_distance_stats
    }


@hydra.main(config_path="conf", config_name="visualize", version_base="1.1")
def main(cfg: OmegaConf):
    """Main entry point using Hydra configuration."""
    
    # Validate required parameters
    if cfg.model_name is None:
        raise ValueError("model_name must be specified in config or via command line")
    
    # Convert config to dict manually to avoid issues with cfg_to_dict
    cfg_dict = {
        "ckpt_base_path": cfg.ckpt_base_path,
        "model_name": cfg.model_name,
        "epoch": cfg.epoch,
        "reference_pos": list(cfg.reference_pos),  # Convert to list explicitly
        "grid_size": cfg.grid_size,
        "distance_metric": cfg.distance_metric,
        "save_path": cfg.save_path,
        "output_base_dir": cfg.output_base_dir,
        "seed": cfg.seed,
        # New multi-reference parameters
        "use_single_reference": cfg.get("use_single_reference", False),
        "num_reference_positions": cfg.get("num_reference_positions", 10),
        "reference_strategy": cfg.get("reference_strategy", "grid")
    }
    
    # Run visualization
    results = visualization_main(cfg_dict)
    
    log.info("Visualization is complete!")


if __name__ == "__main__":
    main()