"""
Utility functions for getting statistics about tensor arrays.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from collections import OrderedDict


def get_tensor_stats(tensor: Union[torch.Tensor, np.ndarray], 
                     name: str = "tensor",
                     detailed: bool = True,
                     percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a tensor or numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array to analyze
        name: Name for the tensor (used in output keys)
        detailed: If True, compute additional statistics
        percentiles: List of percentiles to compute (e.g., [25, 50, 75])
    
    Returns:
        Dictionary containing various statistics
    """
    
    # Convert to tensor if numpy array
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    # Basic stats that always get computed
    stats = OrderedDict()
    
    # Shape and type info
    stats[f"{name}_shape"] = list(tensor.shape)
    stats[f"{name}_dtype"] = str(tensor.dtype)
    stats[f"{name}_device"] = str(tensor.device)
    stats[f"{name}_numel"] = tensor.numel()
    
    # Check for special values
    stats[f"{name}_has_nan"] = torch.isnan(tensor).any().item()
    stats[f"{name}_has_inf"] = torch.isinf(tensor).any().item()
    
    if stats[f"{name}_has_nan"]:
        stats[f"{name}_nan_count"] = torch.isnan(tensor).sum().item()
        stats[f"{name}_nan_ratio"] = stats[f"{name}_nan_count"] / tensor.numel()
    
    if stats[f"{name}_has_inf"]:
        stats[f"{name}_inf_count"] = torch.isinf(tensor).sum().item()
        stats[f"{name}_inf_ratio"] = stats[f"{name}_inf_count"] / tensor.numel()
    
    # Basic statistics (handling NaN values)
    finite_tensor = tensor[torch.isfinite(tensor)]
    
    if finite_tensor.numel() > 0:
        stats[f"{name}_mean"] = finite_tensor.mean().item()
        stats[f"{name}_std"] = finite_tensor.std().item()
        stats[f"{name}_min"] = finite_tensor.min().item()
        stats[f"{name}_max"] = finite_tensor.max().item()
        stats[f"{name}_range"] = stats[f"{name}_max"] - stats[f"{name}_min"]
        
        if detailed:
            # Additional statistics
            stats[f"{name}_median"] = finite_tensor.median().item()
            stats[f"{name}_var"] = finite_tensor.var().item()
            stats[f"{name}_abs_mean"] = finite_tensor.abs().mean().item()
            stats[f"{name}_norm_l1"] = finite_tensor.abs().sum().item()
            stats[f"{name}_norm_l2"] = finite_tensor.norm(p=2).item()
            
            # Value distribution
            stats[f"{name}_n_zeros"] = (finite_tensor == 0).sum().item()
            stats[f"{name}_n_positive"] = (finite_tensor > 0).sum().item()
            stats[f"{name}_n_negative"] = (finite_tensor < 0).sum().item()
            stats[f"{name}_zero_ratio"] = stats[f"{name}_n_zeros"] / finite_tensor.numel()
            
            # Unique values (only for reasonably sized tensors)
            if tensor.numel() < 1e6:
                unique_vals = torch.unique(finite_tensor)
                stats[f"{name}_n_unique"] = unique_vals.numel()
                stats[f"{name}_unique_ratio"] = stats[f"{name}_n_unique"] / finite_tensor.numel()
            
            # Gradient-related statistics
            if tensor.requires_grad and tensor.grad is not None:
                grad_stats = get_tensor_stats(tensor.grad, f"{name}_grad", detailed=False)
                stats.update(grad_stats)
            
            # Percentiles
            if percentiles is None:
                percentiles = [1, 5, 25, 50, 75, 95, 99]
            
            for p in percentiles:
                stats[f"{name}_p{p}"] = torch.quantile(finite_tensor.float(), p/100.0).item()
    else:
        # All values are NaN or Inf
        for key in ["mean", "std", "min", "max", "range"]:
            stats[f"{name}_{key}"] = float('nan')
    
    return stats


def get_batch_tensor_stats(tensors: Dict[str, torch.Tensor],
                          detailed: bool = True) -> Dict[str, Any]:
    """
    Get statistics for multiple named tensors.
    
    Args:
        tensors: Dictionary of name -> tensor pairs
        detailed: If True, compute additional statistics
    
    Returns:
        Combined dictionary of all tensor statistics
    """
    all_stats = OrderedDict()
    
    for name, tensor in tensors.items():
        if tensor is not None:
            tensor_stats = get_tensor_stats(tensor, name, detailed)
            all_stats.update(tensor_stats)
    
    return all_stats


def print_tensor_stats(tensor: Union[torch.Tensor, np.ndarray],
                       name: str = "tensor",
                       detailed: bool = True):
    """
    Print formatted statistics about a tensor.
    
    Args:
        tensor: Tensor to analyze
        name: Name for the tensor
        detailed: If True, print additional statistics
    """
    stats = get_tensor_stats(tensor, name, detailed)
    
    # Print header
    print(f"\n{'='*60}")
    print(f"Statistics for: {name}")
    print(f"{'='*60}")
    
    # Group statistics for better readability
    print("\n📊 Shape & Type:")
    for key in ["shape", "dtype", "device", "numel"]:
        full_key = f"{name}_{key}"
        if full_key in stats:
            print(f"  {key:20s}: {stats[full_key]}")
    
    print("\n📈 Basic Statistics:")
    for key in ["mean", "std", "min", "max", "range", "median"]:
        full_key = f"{name}_{key}"
        if full_key in stats:
            value = stats[full_key]
            if isinstance(value, float):
                print(f"  {key:20s}: {value:12.6f}")
            else:
                print(f"  {key:20s}: {value}")
    
    print("\n⚠️  Special Values:")
    for key in ["has_nan", "has_inf", "nan_count", "inf_count", "n_zeros", "zero_ratio"]:
        full_key = f"{name}_{key}"
        if full_key in stats:
            value = stats[full_key]
            if isinstance(value, float):
                print(f"  {key:20s}: {value:12.6f}")
            else:
                print(f"  {key:20s}: {value}")
    
    if detailed and f"{name}_p1" in stats:
        print("\n📊 Percentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            key = f"{name}_p{p}"
            if key in stats:
                print(f"  {f'p{p}':20s}: {stats[key]:12.6f}")
    
    print(f"{'='*60}\n")


def compare_tensors(tensor1: torch.Tensor, 
                    tensor2: torch.Tensor,
                    name1: str = "tensor1",
                    name2: str = "tensor2",
                    tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two tensors and return comparison statistics.
    
    Args:
        tensor1, tensor2: Tensors to compare
        name1, name2: Names for the tensors
        tolerance: Tolerance for equality check
    
    Returns:
        Dictionary containing comparison statistics
    """
    comparison = OrderedDict()
    
    # Shape comparison
    comparison["shapes_match"] = tensor1.shape == tensor2.shape
    comparison[f"{name1}_shape"] = list(tensor1.shape)
    comparison[f"{name2}_shape"] = list(tensor2.shape)
    
    if comparison["shapes_match"]:
        # Element-wise comparison
        diff = tensor1 - tensor2
        abs_diff = diff.abs()
        
        comparison["max_abs_diff"] = abs_diff.max().item()
        comparison["mean_abs_diff"] = abs_diff.mean().item()
        comparison["std_abs_diff"] = abs_diff.std().item()
        
        # Relative difference (avoiding division by zero)
        denom = tensor2.abs() + 1e-10
        rel_diff = abs_diff / denom
        comparison["max_rel_diff"] = rel_diff.max().item()
        comparison["mean_rel_diff"] = rel_diff.mean().item()
        
        # Count differences
        comparison["n_equal"] = (abs_diff < tolerance).sum().item()
        comparison["n_different"] = (abs_diff >= tolerance).sum().item()
        comparison["equal_ratio"] = comparison["n_equal"] / tensor1.numel()
        
        # Correlation
        if tensor1.numel() > 1:
            tensor1_flat = tensor1.flatten()
            tensor2_flat = tensor2.flatten()
            
            # Remove NaN values for correlation
            mask = torch.isfinite(tensor1_flat) & torch.isfinite(tensor2_flat)
            if mask.sum() > 1:
                t1_clean = tensor1_flat[mask]
                t2_clean = tensor2_flat[mask]
                
                # Pearson correlation
                if t1_clean.std() > 0 and t2_clean.std() > 0:
                    corr = torch.corrcoef(torch.stack([t1_clean, t2_clean]))[0, 1]
                    comparison["correlation"] = corr.item()
    
    return comparison


def monitor_tensor_gradients(model: torch.nn.Module,
                            layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Monitor gradients in a model's parameters.
    
    Args:
        model: PyTorch model
        layer_names: Optional list of specific layer names to monitor
    
    Returns:
        Dictionary containing gradient statistics
    """
    grad_stats = OrderedDict()
    
    for name, param in model.named_parameters():
        if layer_names is None or any(layer in name for layer in layer_names):
            if param.grad is not None:
                stats = get_tensor_stats(param.grad, f"grad_{name}", detailed=False)
                grad_stats.update(stats)
                
                # Add gradient flow indicators
                grad_norm = param.grad.norm(2).item()
                grad_stats[f"grad_{name}_norm"] = grad_norm
                grad_stats[f"grad_{name}_is_exploding"] = grad_norm > 100
                grad_stats[f"grad_{name}_is_vanishing"] = grad_norm < 1e-7
    
    return grad_stats


def get_activation_stats(activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Get statistics for a dictionary of activations (useful for debugging).
    
    Args:
        activations: Dictionary of layer_name -> activation tensor
    
    Returns:
        Dictionary containing activation statistics
    """
    act_stats = OrderedDict()
    
    for name, activation in activations.items():
        if activation is not None:
            stats = get_tensor_stats(activation, f"act_{name}", detailed=True)
            act_stats.update(stats)
            
            # Add activation-specific metrics
            act_stats[f"act_{name}_sparsity"] = (activation == 0).float().mean().item()
            
            # Check for dead neurons (if activation is 2D or higher)
            if activation.dim() >= 2:
                # Check along feature dimension (usually dim 1)
                dead_features = (activation.abs().sum(dim=0) == 0).float().mean().item()
                act_stats[f"act_{name}_dead_features"] = dead_features
    
    return act_stats


# Example usage functions
def example_basic_usage():
    """Example of basic tensor statistics."""
    # Create a sample tensor
    tensor = torch.randn(100, 50, 32)
    
    # Get basic stats
    stats = get_tensor_stats(tensor, "sample_tensor")
    
    # Print formatted stats
    print_tensor_stats(tensor, "sample_tensor")
    
    return stats


def example_model_monitoring():
    """Example of monitoring model parameters and gradients."""
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    
    # Forward pass with dummy data
    x = torch.randn(32, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Monitor gradients
    grad_stats = monitor_tensor_gradients(model)
    
    # Get parameter statistics
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats.update(get_tensor_stats(param, f"param_{name}", detailed=False))
    
    return grad_stats, param_stats


if __name__ == "__main__":
    # Run examples
    print("Running basic tensor statistics example...")
    basic_stats = example_basic_usage()
    
    print("\nRunning model monitoring example...")
    grad_stats, param_stats = example_model_monitoring()
    
    print("\nGradient statistics:")
    for key, value in list(grad_stats.items())[:10]:
        print(f"  {key}: {value}")
