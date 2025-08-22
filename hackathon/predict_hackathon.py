# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

# ---- Participants may modify only these four functions ----------------------

def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> tuple[dict, List[str]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        Tuple of (final input dict, list of CLI args)
    """
    # Default: add 5 models
    cli_args = ["--diffusion_samples", "5"]
    return input_dict, cli_args

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligand: SmallMolecule, input_dict: dict, msa_dir: Optional[Path] = None) -> tuple[dict, List[str]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligand: The small molecule/ligand
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        Tuple of (final input dict, list of CLI args)
    """
    cli_args = ["--diffusion_samples", "5"]
    return input_dict, cli_args

def post_process_protein_complex(datapoint: Datapoint, input_dict: dict[str, Any], prediction_dir: Path) -> List[str]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dict: The input dictionary used for prediction
        prediction_dir: The directory containing prediction results
    Returns: 
        Sorted pdb file names that should be used as your submission.
    """
    datapoint_id = datapoint.datapoint_id
    pred_root = Path(prediction_dir)
    pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.pdb"))
    if not pdbs:
        pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.cif"))
    return [p.name for p in pdbs]

def post_process_protein_ligand(datapoint: Datapoint, input_dict: dict[str, Any], prediction_dir: Path) -> List[str]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dict: The input dictionary used for prediction
        prediction_dir: The directory containing prediction results
    Returns: 
        Sorted pdb file names that should be used as your submission.
    """
    datapoint_id = datapoint.datapoint_id
    pred_root = Path(prediction_dir)
    pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.pdb"))
    if not pdbs:
        pdbs = sorted(pred_root.glob(f"{datapoint_id}_model_*.cif"))
    return [p.name for p in pdbs]

# ---- End of participant section ---------------------------------------------


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
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligand: Optional[SmallMolecule] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
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
                "msa": msa_relative_path
            }
        }
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
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligand, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        input_dict, cli_args = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        input_dict, cli_args = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligand, input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Write input YAML
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / f"{datapoint.datapoint_id}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(input_dict, f, sort_keys=False)

    # Run boltz
    cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    fixed = [
        "boltz", "predict", str(yaml_path),
        "--devices", "1",
        "--out_dir", str(out_dir),
        "--cache", cache,
        "--no_kernels",
        "--output_format", "pdb",
    ]
    cmd = fixed + cli_args
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Compute prediction subfolder
    pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}" / "predictions" / datapoint.datapoint_id

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, input_dict, pred_subfolder)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, input_dict, pred_subfolder)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files):
        p = pred_subfolder / file_path
        target = subdir / (f"model_{i}.pdb" if p.suffix == ".pdb" else f"model_{i}{p.suffix}")
        shutil.copy2(p, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

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
        _run_boltz_and_collect(datapoint)
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
