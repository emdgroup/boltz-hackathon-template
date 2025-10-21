#!/usr/bin/env python
"""
Speed Comparison Tool for Flow Matching vs Score-Based Diffusion

This script helps you measure actual wall-clock time differences between
flow matching and score-based diffusion predictions.

Usage:
    python measure_speed_comparison.py
"""

import sys
import torch
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from boltz.utils.flow_matching_utils import DataLoader, PDBParser


def run_prediction_with_timing(
    input_yaml: Path,
    output_dir: Path,
    checkpoint_path: Path,
    sampling_steps: int,
    method_name: str
) -> Tuple[bool, float]:
    """
    Run a prediction and measure wall-clock time.
    
    Args:
        input_yaml: Path to input YAML file
        output_dir: Output directory
        checkpoint_path: Path to checkpoint file
        sampling_steps: Number of sampling steps
        method_name: Name of the method for logging
        
    Returns:
        Tuple of (success, elapsed_time)
    """
    print(f"\nRunning {method_name} prediction...")
    print(f"  Steps: {sampling_steps}")
    print(f"  Input: {input_yaml}")
    
    # Prepare command
    cmd = [
        sys.executable, "-m", "boltz.main", "predict",
        str(input_yaml),
        "--out_dir", str(output_dir),
        "--checkpoint", str(checkpoint_path),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", "1",
        "--recycling_steps", "3",
        "--output_format", "pdb",
        "--devices", "1",
        "--accelerator", "cpu",  # Use CPU for consistent timing
        "--use_msa_server",
    ]
    
    # Measure wall-clock time
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    elapsed = time.time() - start_time
    
    # Check if prediction was successful
    protein_id = input_yaml.stem
    result_dir = output_dir / f"boltz_results_{protein_id}" / "predictions" / protein_id
    pdb_files = list(result_dir.glob("*.pdb")) if result_dir.exists() else []
    
    success = result.returncode == 0 and len(pdb_files) > 0
    
    if success:
        print(f"  ✓ SUCCESS! Generated {len(pdb_files)} structure(s) in {elapsed:.1f}s")
    else:
        print(f"  ✗ FAILED in {elapsed:.1f}s")
        if result.stderr:
            print(f"    Error: {result.stderr[:200]}")
    
    return success, elapsed


def measure_speed_comparison(
    protein_id: str,
    flow_checkpoint: Path,
    original_checkpoint: Path,
    flow_steps: int = 20,
    score_steps: int = 200
) -> Dict[str, Any]:
    """
    Measure actual speed comparison between flow matching and score-based diffusion.
    
    Args:
        protein_id: Protein identifier to test
        flow_checkpoint: Path to flow matching checkpoint
        original_checkpoint: Path to original checkpoint
        flow_steps: Number of flow matching steps
        score_steps: Number of score-based steps
        
    Returns:
        Dictionary with timing results
    """
    print("=" * 80)
    print(f"MEASURING SPEED COMPARISON FOR {protein_id}")
    print("=" * 80)
    
    # Setup paths
    input_dir = Path("boltz_inputs")
    flow_output_dir = Path("speed_test_flow")
    score_output_dir = Path("speed_test_score")
    
    input_yaml = input_dir / f"{protein_id}.yaml"
    
    if not input_yaml.exists():
        print(f"✗ Input YAML not found: {input_yaml}")
        return {"error": "Input file not found"}
    
    # Create output directories
    flow_output_dir.mkdir(exist_ok=True)
    score_output_dir.mkdir(exist_ok=True)
    
    results = {
        "protein_id": protein_id,
        "flow_steps": flow_steps,
        "score_steps": score_steps,
    }
    
    # Test flow matching
    print(f"\n1. Testing Flow Matching ({flow_steps} steps)")
    print("-" * 50)
    
    flow_success, flow_time = run_prediction_with_timing(
        input_yaml,
        flow_output_dir,
        flow_checkpoint,
        flow_steps,
        "Flow Matching"
    )
    
    results["flow_matching"] = {
        "success": flow_success,
        "time": flow_time,
        "steps": flow_steps
    }
    
    # Test score-based diffusion
    print(f"\n2. Testing Score-Based Diffusion ({score_steps} steps)")
    print("-" * 50)
    
    score_success, score_time = run_prediction_with_timing(
        input_yaml,
        score_output_dir,
        original_checkpoint,
        score_steps,
        "Score-Based Diffusion"
    )
    
    results["score_based"] = {
        "success": score_success,
        "time": score_time,
        "steps": score_steps
    }
    
    # Calculate speedup if both succeeded
    if flow_success and score_success:
        speedup = score_time / flow_time
        results["speedup"] = speedup
        results["time_saved"] = score_time - flow_time
        
        print(f"\n3. SPEED COMPARISON RESULTS")
        print("-" * 50)
        print(f"Flow Matching:     {flow_time:.1f}s ({flow_steps} steps)")
        print(f"Score-Based:       {score_time:.1f}s ({score_steps} steps)")
        print(f"Speedup:           {speedup:.1f}x faster")
        print(f"Time saved:        {score_time - flow_time:.1f}s")
        
        if speedup > 1.5:
            print(f"✓ Flow matching is significantly faster!")
        elif speedup > 1.1:
            print(f"✓ Flow matching is moderately faster")
        else:
            print(f"⚠ Flow matching speedup is minimal")
    else:
        print(f"\n✗ Cannot calculate speedup - one or both methods failed")
        results["speedup"] = None
    
    return results


def main():
    """Main function to run speed comparison."""
    print("\n🚀 SPEED COMPARISON TOOL 🚀\n")
    
    # Setup paths
    flow_checkpoint = Path("flow_matching_boltz2.ckpt")
    original_checkpoint = Path("/home/nab/.boltz/boltz2_conf.ckpt")
    
    if not flow_checkpoint.exists():
        print(f"✗ Flow matching checkpoint not found: {flow_checkpoint}")
        print("Run the main script first to create the flow matching checkpoint")
        return
    
    if not original_checkpoint.exists():
        print(f"✗ Original checkpoint not found: {original_checkpoint}")
        print("Please run a Boltz prediction first to download the model")
        return
    
    # Test on available proteins
    input_dir = Path("boltz_inputs")
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        print("Run the main script first to prepare input files")
        return
    
    yaml_files = list(input_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"✗ No YAML files found in {input_dir}")
        return
    
    print(f"Found {len(yaml_files)} proteins to test")
    
    # Test first protein
    first_protein = yaml_files[0].stem
    print(f"Testing on: {first_protein}")
    
    results = measure_speed_comparison(
        first_protein,
        flow_checkpoint,
        original_checkpoint,
        flow_steps=20,
        score_steps=200
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if "speedup" in results and results["speedup"] is not None:
        print(f"✓ Measured speedup: {results['speedup']:.1f}x")
        print(f"✓ Time saved: {results['time_saved']:.1f}s per prediction")
        print(f"✓ Flow matching uses {results['flow_steps']} steps vs {results['score_steps']} for score-based")
    else:
        print("✗ Could not measure speedup - check error messages above")
    
    print(f"\nTo test more proteins, modify the script to loop through all YAML files.")


if __name__ == "__main__":
    main()
