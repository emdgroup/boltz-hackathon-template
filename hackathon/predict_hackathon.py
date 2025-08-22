# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")



ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")

args = ap.parse_args()

# ---- Participants may modify only these four functions ----------------------

def get_custom_args(datapoint_id: str) -> List[str]:
    """
    Return extra CLI flags for boltz predict, e.g. ["--recycling_steps", "10"].
    By default returns args that make 5 models like AF3/Chai baselines.
    
    Args:
        datapoint_id: The unique identifier for this datapoint
        
    Returns
    -------
        List of additional command line arguments for boltz predict
    """
    # NOTE: --diffusion_samples controls #models; --output_format selects pdb vs mmcif
    return ["--diffusion_samples", "5"]

def inputs_to_yaml(
    datapoint_id: str,
    proteins: Iterable[Protein],
    ligand: Optional[SmallMolecule] = None,
    msa_dir: Optional[Path] = None,
) -> Path:
    """
    Writes inputs/{datapoint_id}.yaml for Boltz.
    
    This function converts the hackathon datapoint format into Boltz-compatible YAML.
    Participants can modify this function to customize the YAML structure,
    add additional fields, or implement custom preprocessing logic.
    
    Args:
        datapoint_id: Unique identifier for this prediction
        proteins: List of protein sequences
        ligand: Optional small molecule/ligand
        out_dir: Directory to write YAML files (default: "inputs")
        msa_dir: Directory containing MSA files (used to compute relative paths)
    
    Returns
    -------
        Path to the created YAML file
    """
    out_dir = args.intermediate_dir / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    ypath = out_dir / f"{datapoint_id}.yaml"

    seqs = []
    for p in proteins:
        # Compute relative path from YAML file to MSA file
        if msa_dir and p.msa:
            # If msa is just a filename, construct the full path
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa

            # Compute relative path from YAML location to MSA file
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa

        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path  # Use relative path
            }
        }
        # Add modifications if present and not None
        if getattr(p, "modifications", None) and p.modifications:
            entry["protein"]["modifications"] = p.modifications
        seqs.append(entry)

    if ligand:
        l = {
            "ligand": {
                "id": ligand.id,
                "smiles": ligand.smiles
            }
        }
        seqs.append(l)

    doc = {
        "version": 1,
        "sequences": seqs,
    }

    with open(ypath, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)

    return ypath

def predict_protein_complex(datapoint_id: str, proteins: List[Protein], msa_dir: Optional[Path] = None) -> None:
    """
    Predict structure for a protein complex.
    Baseline: write YAML, run boltz, copy the top-k models into submission/.
    
    Participants can modify this function to add custom pre/post-processing.
    
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        msa_dir: Directory containing MSA files (for computing relative paths)
    """
    yaml_path = inputs_to_yaml(datapoint_id, proteins=proteins, ligand=None,
                              msa_dir=msa_dir)
    _run_boltz_and_collect(datapoint_id, yaml_path)

def predict_protein_ligand(datapoint_id: str, protein: Protein, ligand: SmallMolecule, msa_dir: Optional[Path] = None) -> None:
    """
    Predict structure for a protein-ligand complex.
    Baseline: write YAML, run boltz, copy the top-k models into submission/.
    
    Participants can modify this function to add custom pre/post-processing.
    
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligand: The small molecule/ligand
        msa_dir: Directory containing MSA files (for computing relative paths)
    """
    yaml_path = inputs_to_yaml(datapoint_id, proteins=[protein], ligand=ligand,
                              msa_dir=msa_dir)
    _run_boltz_and_collect(datapoint_id, yaml_path)

# ---- End of participant section ---------------------------------------------

def _run_boltz_and_collect(datapoint_id: str, input_yaml: Path) -> None:
    """
    Runs boltz predict with fixed args + custom args, then copies
    <intermediate_dir>/predictions/boltz_results_{datapoint_id}/predictions/{datapoint_id}/{datapoint_id}_model_{0..}.[pdb|cif]
    to submission_dir/{datapoint_id}/model_{0..}.pdb
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Fixed args (safe defaults)
    cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    fixed = [
        "boltz", "predict", str(input_yaml),
        "--devices", "1",
        "--out_dir", str(out_dir),
        "--cache", cache,
        "--no_kernels",
        "--output_format", "pdb",          # We want PDB files for submissions
    ]
    cmd = fixed + get_custom_args(datapoint_id)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Find model files
    pred_root = out_dir / f"boltz_results_{datapoint_id}" / "predictions" / datapoint_id
    pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.pdb"))
    # If none found (e.g., mmCIF), try cif (we won't rename to .pdb)
    if not pdbs:
        pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.cif"))

    if not pdbs:
        raise FileNotFoundError(f"No model files found for {datapoint_id} under {pred_root}")

    # Copy to submission_dir/{datapoint_id}/model_{i}.pdb (if cif, keep .cif to be honest)
    for i, p in enumerate(pdbs):
        target = subdir / (f"model_{i}.pdb" if p.suffix == ".pdb" else f"model_{i}{p.suffix}")
        shutil.copy2(p, target)
        print(f"Saved: {target}")

    subprocess.run(["chmod", "-R", "777", str(subdir)], check=False, capture_output=True, text=True)

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _process_single_datapoint(datapoint: Datapoint, msa_dir: Optional[Path] = None):
    """Process a single datapoint specification."""
    datapoint_id = datapoint.datapoint_id
    task_type = datapoint.task_type

    # Route based on task_type field
    if task_type == "protein_ligand":
        if not datapoint.ligand:
            raise ValueError(f"Datapoint {datapoint_id} has task_type='protein_ligand' but no ligand specified")

        ligand = datapoint.ligand
        proteins = datapoint.proteins

        if len(proteins) != 1:
            raise ValueError(f"Datapoint {datapoint_id}: protein_ligand task expects exactly one protein")

        print(f"Processing protein-ligand datapoint: {datapoint_id}")
        predict_protein_ligand(datapoint_id, proteins[0], ligand, msa_dir)

    elif task_type == "protein_complex":
        proteins = datapoint.proteins

        if len(proteins) < 2:
            raise ValueError(f"Datapoint {datapoint_id}: protein_complex task expects at least two proteins")

        print(f"Processing protein complex datapoint: {datapoint_id}")
        predict_protein_complex(datapoint_id, proteins, msa_dir)

    else:
        raise ValueError(f"Datapoint {datapoint_id}: unknown task_type '{task_type}'. Expected 'protein_complex' or 'protein_ligand'")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _process_single_datapoint(datapoint, msa_dir)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _process_single_datapoint(datapoint, msa_dir)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    if args.input_json:
        _process_json(args.input_json, args.msa_dir)
    elif args.input_jsonl:
        _process_jsonl(args.input_jsonl, args.msa_dir)

if __name__ == "__main__":
    main()
