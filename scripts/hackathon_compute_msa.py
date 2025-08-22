#!/usr/bin/env python3
"""
Script to compute MSAs for protein sequences in a JSONL file using ColabFold search.

This script processes a JSONL file containing protein sequences, generates MSAs for each
unique protein sequence using ColabFold search, and outputs a new JSONL file with
updated MSA paths.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import shutil

from generate_local_msa import LocalColabFoldConfig, run_colabfold_search, A3MProcessor


def create_fasta_from_sequences(sequences: List[str], seq_ids: List[str], output_path: Path) -> str:
    """Create a FASTA file from protein sequences, concatenating multiple sequences with ':'."""
    if len(sequences) == 1:
        # Single sequence
        fasta_content = f">{seq_ids[0]}\n{sequences[0]}\n"
    else:
        # Multiple sequences - concatenate with ':'
        concatenated_sequence = ":".join(sequences)
        concatenated_id = "_".join(seq_ids)
        fasta_content = f">{concatenated_id}\n{concatenated_sequence}\n"
    
    output_path.write_text(fasta_content)
    return str(output_path)


def get_entry_hash(entry: Dict[str, Any]) -> str:
    """Generate a hash for a JSONL entry based on its protein sequences."""
    return entry["datapoint_id"]


def extract_entry_sequences(jsonl_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract protein sequences for each JSONL entry.
    
    Returns:
        Dict mapping entry hash to entry data with sequences and IDs
    """
    entry_data = {}
    
    for entry in jsonl_data:
        if "proteins" in entry:
            sequences = []
            seq_ids = []
            
            for protein in entry["proteins"]:
                if "sequence" in protein:
                    sequences.append(protein["sequence"])
                    seq_ids.append(protein.get("id", f"protein_{len(seq_ids)}"))
            
            if sequences:  # Only process entries with protein sequences
                entry_hash = get_entry_hash(entry)
                entry_data[entry_hash] = {
                    "sequences": sequences,
                    "seq_ids": seq_ids,
                    "original_entry": entry
                }
    
    return entry_data


def process_msa_generation(
    entry_data: Dict[str, Dict[str, Any]],
    temp_dir: Path,
    msa_dir: Path,
    colabsearch_path: str,
    db_dir: str,
    mmseqs_path: Optional[str] = None,
    db1: str = "uniref30_2302_db",
    db2: Optional[str] = None,
    db3: str = "colabfold_envdb_202108_db",
) -> Dict[str, str]:
    """
    Generate MSAs for all entries.
    
    Returns:
        Dict mapping entry hash to MSA CSV file path(s)
    """
    msa_paths = {}
    
    for entry_hash, data in entry_data.items():
        sequences = data["sequences"]
        seq_ids = data["seq_ids"]
        
        print(f"Processing entry {entry_hash} with {len(sequences)} sequences...")
        
        # Create FASTA file
        fasta_path = temp_dir / f"{entry_hash}.fasta"
        create_fasta_from_sequences(sequences, seq_ids, fasta_path)
        
        # Create temporary results directory for this entry
        temp_results_dir = temp_dir / f"results_{entry_hash}"
        temp_results_dir.mkdir(exist_ok=True)
        
        # Configure ColabFold search
        config = LocalColabFoldConfig(
            colabsearch=colabsearch_path,
            query_fpath=str(fasta_path),
            db_dir=db_dir,
            results_dir=str(temp_results_dir),
            mmseqs_path=mmseqs_path,
            db1=db1,
            db2=db2,
            db3=db3,
        )
        
        try:
            # Run ColabFold search
            a3m_file = run_colabfold_search(config)
            
            # Process A3M file to generate CSV
            processor = A3MProcessor(a3m_file, str(temp_results_dir))
            processor.split_sequences()
            
            # Move the CSV files to the final MSA directory
            csv_files = list(temp_results_dir.glob("msa_*.csv"))
            if csv_files:
                if len(sequences) == 1:
                    # Single sequence - use one CSV file
                    source_csv = csv_files[0]
                    target_csv = msa_dir / f"{entry_hash}.csv"
                    shutil.move(str(source_csv), str(target_csv))
                    msa_paths[entry_hash] = str(target_csv)
                else:
                    # Multiple sequences - store paths to multiple CSV files
                    csv_paths = []
                    for i, source_csv in enumerate(csv_files):
                        target_csv = msa_dir / f"{entry_hash}_{i}.csv"
                        shutil.move(str(source_csv), str(target_csv))
                        csv_paths.append(str(target_csv))
                    msa_paths[entry_hash] = csv_paths
                    
                print(f"MSA generated for {entry_hash}: {msa_paths[entry_hash]}")
            else:
                print(f"Warning: No CSV files generated for entry {entry_hash}")
                
        except Exception as e:
            print(f"Error processing entry {entry_hash}: {e}")
            continue
    
    return msa_paths


def update_jsonl_with_msa_paths(
    entry_data: Dict[str, Dict[str, Any]],
    msa_paths: Dict[str, str],
    output_jsonl: Path,
) -> None:
    """Update JSONL data with new MSA paths (filenames only) and write to output file."""
    
    updated_entries = []
    
    for entry_hash, data in entry_data.items():
        entry = data["original_entry"].copy()
        
        if entry_hash in msa_paths:
            msa_path_data = msa_paths[entry_hash]
            
            if "proteins" in entry:
                if isinstance(msa_path_data, list):
                    # Multiple MSA files for multiple proteins
                    for i, protein in enumerate(entry["proteins"]):
                        if i < len(msa_path_data):
                            # Store only the filename, not the full path
                            protein["msa"] = Path(msa_path_data[i]).name
                        else:
                            print(f"Warning: No MSA file for protein {i} in entry {entry.get('datapoint_id', 'unknown')}")
                else:
                    # Single MSA file - assign to all proteins (for concatenated sequences)
                    for protein in entry["proteins"]:
                        # Store only the filename, not the full path
                        protein["msa"] = Path(msa_path_data).name
        else:
            print(f"Warning: No MSA found for entry {entry.get('datapoint_id', 'unknown')}")
        
        updated_entries.append(entry)
    
    # Write updated JSONL
    with output_jsonl.open("w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute MSAs for protein sequences in a JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input JSONL file containing protein sequences"
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL file with updated MSA paths"
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        required=True,
        help="Directory to store final MSA CSV files"
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        required=True,
        help="Directory containing ColabFold databases"
    )
    
    # Optional arguments
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for intermediate files (default: system temp dir)"
    )
    parser.add_argument(
        "--colabsearch",
        type=str,
        default="colabfold_search",
        help="Path to colabfold_search executable"
    )
    parser.add_argument(
        "--mmseqs-path",
        type=str,
        default="mmseqs",
        help="Path to MMseqs2 binary"
    )
    parser.add_argument(
        "--db1",
        type=str,
        default="uniref30_2302_db",
        help="First database name"
    )
    parser.add_argument(
        "--db2",
        type=str,
        default=None,
        help="Templates database"
    )
    parser.add_argument(
        "--db3",
        type=str,
        default="colabfold_envdb_202108_db",
        help="Environmental database"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate input file
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {args.input_jsonl}")
    
    # Create output directories
    args.msa_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if not args.temp_dir:
            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                _process_msa_workflow(args, temp_dir)
        else:
            args.temp_dir.mkdir(parents=True, exist_ok=True)
            _process_msa_workflow(args, args.temp_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        raise


def _process_msa_workflow(args: argparse.Namespace, temp_dir: Path):
    """Process the MSA generation workflow."""
    print(f"Loading JSONL data from {args.input_jsonl}")
    jsonl_data = load_jsonl(args.input_jsonl)
    
    print("Extracting entry sequences...")
    entry_data = extract_entry_sequences(jsonl_data)
    print(f"Found {len(entry_data)} entries with protein sequences")
    
    if not entry_data:
        print("No protein sequences found in input file")
        return
    
    print("Generating MSAs...")
    msa_paths = process_msa_generation(
        entry_data=entry_data,
        temp_dir=temp_dir,
        msa_dir=args.msa_dir,
        colabsearch_path=args.colabsearch,
        db_dir=args.db_dir,
        mmseqs_path=args.mmseqs_path,
        db1=args.db1,
        db2=args.db2,
        db3=args.db3,
    )
    
    print(f"Successfully generated MSA files for {len(msa_paths)} entries")
    
    print(f"Updating JSONL with MSA paths and writing to {args.output_jsonl}")
    update_jsonl_with_msa_paths(entry_data, msa_paths, args.output_jsonl)
    
    print("MSA generation complete!")


if __name__ == "__main__":
    main()
