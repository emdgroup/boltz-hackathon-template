"""
Common utilities for Boltz flow matching.

This module provides shared functionality across all scripts to eliminate
duplication and ensure consistency.
"""

import torch
import numpy as np
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class PDBParser:
    """Standardized PDB parsing utilities."""
    
    RESIDUE_MAP = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    
    @classmethod
    def load_coordinates(cls, pdb_file: Path) -> Dict[str, Any]:
        """
        Load coordinates and metadata from PDB file.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary containing coordinates, atom info, and metadata
        """
        ca_coords = []
        all_coords = []
        residue_ids = []
        atom_names = []
        chain_sequences = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26].strip())
                    chain_id = line[21]
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    all_coords.append([x, y, z])
                    atom_names.append(atom_name)
                    
                    # Keep CA atoms for coarse-grained representation
                    if atom_name == 'CA':
                        ca_coords.append([x, y, z])
                        residue_ids.append(res_id)
                        
                        # Build chain sequences
                        if chain_id not in chain_sequences:
                            chain_sequences[chain_id] = []
                        if res_name in cls.RESIDUE_MAP:
                            one_letter = cls.RESIDUE_MAP[res_name]
                            if not chain_sequences[chain_id] or chain_sequences[chain_id][-1][1] != res_id:
                                chain_sequences[chain_id].append((one_letter, res_id))
        
        return {
            'ca_coords': np.array(ca_coords),
            'all_coords': np.array(all_coords),
            'residue_ids': residue_ids,
            'atom_names': atom_names,
            'chain_sequences': chain_sequences,
            'num_residues': len(ca_coords),
            'num_atoms': len(all_coords),
        }
    
    @classmethod
    def extract_sequences(cls, pdb_file: Path) -> List[Dict[str, Any]]:
        """
        Extract protein sequences from PDB file.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            List of sequence dictionaries
        """
        structure_data = cls.load_coordinates(pdb_file)
        chain_sequences = structure_data['chain_sequences']
        
        sequences = []
        for chain_id in sorted(chain_sequences.keys()):
            seq = ''.join([res[0] for res in chain_sequences[chain_id]])
            sequences.append({
                'protein': {
                    'id': chain_id,
                    'sequence': seq,
                }
            })
        
        return sequences


class DataLoader:
    """Standardized data loading utilities."""
    
    @staticmethod
    def load_protein_ids(data_dir: Path) -> List[str]:
        """
        Load protein IDs from hackathon dataset.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            List of protein identifiers
        """
        jsonl_file = data_dir / "abag_public.jsonl"
        protein_ids = []
        
        if jsonl_file.exists():
            with open(jsonl_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    protein_ids.append(entry['datapoint_id'])
        
        return protein_ids
    
    @staticmethod
    def prepare_boltz_input(protein_id: str, pdb_file: Path) -> Dict[str, Any]:
        """
        Prepare YAML input for Boltz from PDB file.
        
        Args:
            protein_id: Protein identifier
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary containing Boltz input data
        """
        sequences = PDBParser.extract_sequences(pdb_file)
        
        return {
            'version': 1,
            'sequences': sequences,
        }
    
    @staticmethod
    def save_yaml_input(input_data: Dict[str, Any], output_path: Path) -> None:
        """
        Save input data as YAML file.
        
        Args:
            input_data: Input data dictionary
            output_path: Path to save YAML file
        """
        with open(output_path, 'w') as f:
            yaml.dump(input_data, f, sort_keys=False)


class FlowMatchingConverter:
    """Analytical conversion utilities for flow matching."""
    
    def __init__(
        self,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,
        conversion_method: str = 'noise_based'
    ):
        """
        Initialize converter with Boltz-2 parameters.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            sigma_data: Data scaling parameter
            conversion_method: Conversion method to use
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.conversion_method = conversion_method
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert noise level sigma to flow time t ∈ [0,1].
        
        Args:
            sigma: Noise level tensor
            
        Returns:
            Flow time tensor
        """
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma)
        
        log_sigma = torch.log(sigma + 1e-10)
        log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=sigma.device))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=sigma.device))
        
        t = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
        return torch.clamp(t, 0.0, 1.0)
    
    def t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert flow time t to noise level sigma.
        
        Args:
            t: Flow time tensor
            
        Returns:
            Noise level tensor
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t)
        
        log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=t.device))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=t.device))
        
        log_sigma = log_sigma_min + t * (log_sigma_max - log_sigma_min)
        return torch.exp(log_sigma)
    
    def convert_to_velocity(self, x_t: torch.Tensor, denoised_coords: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert score model output to velocity field.
        
        Args:
            x_t: Noisy coordinates
            denoised_coords: Denoised coordinates from score model
            sigma: Noise level
            
        Returns:
            Velocity field tensor
        """
        # Handle sigma dimensions
        if isinstance(sigma, (int, float)):
            sigma_expanded = sigma
        elif sigma.dim() == 1:
            sigma_expanded = sigma.reshape(-1, 1, 1)
        else:
            sigma_expanded = sigma
        
        # Noise-based conversion (most accurate)
        epsilon = (x_t - denoised_coords) / (sigma_expanded + 1e-8)
        velocity = epsilon - denoised_coords
        
        return velocity


class CheckpointManager:
    """Utilities for managing Boltz checkpoints."""
    
    @staticmethod
    def convert_to_flow_matching(
        original_checkpoint: Path,
        output_checkpoint: Path,
        flow_steps: int = 20,
        device: str = 'cpu'
    ) -> Path:
        """
        Convert Boltz checkpoint to flow matching configuration.
        
        Args:
            original_checkpoint: Path to original checkpoint
            output_checkpoint: Path to save converted checkpoint
            flow_steps: Number of flow matching steps
            device: Device to use for loading
            
        Returns:
            Path to converted checkpoint
        """
        if output_checkpoint.exists():
            print(f"✓ Flow matching checkpoint already exists: {output_checkpoint}")
            return output_checkpoint
        
        print(f"\nLoading checkpoint: {original_checkpoint}")
        checkpoint = torch.load(original_checkpoint, map_location=device, weights_only=False)
        
        print(f"✓ Loaded checkpoint")
        
        # Modify sampling steps in hyperparameters
        hparams = checkpoint['hyper_parameters']
        
        if 'structure_args' in hparams:
            structure_args = hparams['structure_args']
            
            # Store original for comparison
            original_steps = structure_args.get('num_sampling_steps', 200)
            
            # Change to flow matching steps
            structure_args['num_sampling_steps'] = flow_steps
            
            print(f"\n✓ Modified hyperparameters:")
            print(f"  Sampling steps: {original_steps} → {flow_steps} (flow matching)")
            print(f"  Expected speedup: {original_steps / flow_steps:.1f}x")
        
        # Save modified checkpoint
        print(f"\nSaving flow matching checkpoint: {output_checkpoint}")
        torch.save(checkpoint, output_checkpoint)
        
        print(f"✓ Flow matching checkpoint saved")
        
        return output_checkpoint


class ResultAnalyzer:
    """Utilities for analyzing prediction results."""
    
    @staticmethod
    def print_timing_summary(results: List[Dict[str, Any]], flow_steps: int, score_steps: int) -> None:
        """
        Print comprehensive timing analysis.
        
        Args:
            results: List of result dictionaries
            flow_steps: Number of flow matching steps
            score_steps: Number of score-based steps
        """
        successful = [r for r in results if r['success']]
        
        if not successful:
            print("\n✗ No successful predictions to analyze")
            return
        
        print(f"\n✓ Successfully predicted {len(successful)}/{len(results)} proteins")
        print(f"\nTiming with flow matching ({flow_steps} steps):")
        print(f"  {'Protein':<15} {'Time (s)':<12}")
        print("  " + "-"*40)
        
        for r in successful:
            print(f"  {r['protein_id']:<15} {r['time']:<12.1f}")
        
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print("  " + "-"*40)
        print(f"  Average: {avg_time:.1f}s with flow matching")
    
    @staticmethod
    def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute summary statistics from results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary of computed statistics
        """
        successful = [r for r in results if r['success']]
        
        if not successful:
            return {
                'success_rate': 0.0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'total_proteins': len(results)
            }
        
        times = [r['time'] for r in successful]
        
        return {
            'success_rate': len(successful) / len(results),
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_proteins': len(results),
            'successful_proteins': len(successful)
        }


class ErrorHandler:
    """Standardized error handling utilities."""
    
    @staticmethod
    def handle_subprocess_error(result: subprocess.CompletedProcess, context: str = "") -> None:
        """
        Handle subprocess errors with consistent formatting.
        
        Args:
            result: Completed subprocess result
            context: Additional context for error message
        """
        if result.returncode != 0:
            print(f"✗ Process failed (return code: {result.returncode})")
            if context:
                print(f"   Context: {context}")
            
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            
            if result.stdout and "Error" in result.stdout:
                lines = result.stdout.split('\n')
                error_lines = [line for line in lines if 'Error' in line or 'Failed' in line]
                for line in error_lines[:5]:
                    print(f"   {line}")
    
    @staticmethod
    def handle_file_not_found(file_path: Path, context: str = "") -> None:
        """
        Handle file not found errors consistently.
        
        Args:
            file_path: Path to missing file
            context: Additional context
        """
        print(f"✗ File not found: {file_path}")
        if context:
            print(f"   Context: {context}")
    
    @staticmethod
    def handle_model_loading_error(error: Exception, checkpoint_path: Path) -> None:
        """
        Handle model loading errors with helpful suggestions.
        
        Args:
            error: The exception that occurred
            checkpoint_path: Path to checkpoint that failed to load
        """
        print(f"✗ Error loading model: {error}")
        print(f"\nThis may be due to:")
        print(f"  1. Missing checkpoint: {checkpoint_path}")
        print(f"  2. Version mismatch between model and checkpoint")
        print(f"  3. Missing dependencies")
        print(f"\nTo fix:")
        print(f"  1. Run a Boltz prediction first to download the model")
        print(f"  2. Check that the checkpoint exists and is valid")
        print(f"  3. Ensure all dependencies are installed")
