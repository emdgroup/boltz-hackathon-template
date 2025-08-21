# Hackathon MSA Computation Script

This script (`hackathon_compute_msa.py`) computes Multiple Sequence Alignments (MSAs) for protein sequences in a JSONL file using ColabFold search.

## Features

- Processes JSONL files with protein sequence data
- Generates MSAs for each entry separately (not by unique sequences)
- Handles both single and multiple protein sequences per entry
- For multiple sequences in one entry, concatenates them with ":" in the FASTA file
- Outputs CSV files in MSA format compatible with the Boltz model
- Preserves original JSONL structure while updating MSA paths

## Requirements

- ColabFold search executable (`colabfold_search`)
- MMseqs2 (`mmseqs`)
- ColabFold databases (e.g., `uniref30_2302_db`, `colabfold_envdb_202108_db`)

## Usage

```bash
python hackathon_compute_msa.py \
    --input-jsonl examples/test_dataset.jsonl \
    --output-jsonl output_with_msa.jsonl \
    --msa-dir ./msa_outputs \
    --db-dir /path/to/colabfold/databases \
    --temp-dir ./temp_msa
```

### Required Arguments

- `--input-jsonl`: Input JSONL file containing protein sequences
- `--output-jsonl`: Output JSONL file with updated MSA paths
- `--msa-dir`: Directory to store final MSA CSV files
- `--db-dir`: Directory containing ColabFold databases

### Optional Arguments

- `--temp-dir`: Temporary directory for intermediate files (default: system temp dir)
- `--colabsearch`: Path to colabfold_search executable (default: `colabfold_search`)
- `--mmseqs-path`: Path to MMseqs2 binary (default: `mmseqs`)
- `--db1`: First database name (default: `uniref30_2302_db`)
- `--db2`: Templates database (optional)
- `--db3`: Environmental database (default: `colabfold_envdb_202108_db`)

## Input Format

The input JSONL file should contain entries with protein sequences:

```json
{"datapoint_id": "affinity", "task_type": "protein_ligand", "proteins": [{"id": "A", "sequence": "MVTPEGNVSLVDESLLVGV", "chain_id": "A"}], "ligand": {"id": "B", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O", "chain_id": "Z"}}
{"datapoint_id": "complex", "task_type": "protein_complex", "proteins": [{"id": "A", "sequence": "MVTPEGNVSLVDESLLVGV", "chain_id": "A"}, {"id": "B", "sequence": "MVTPEGNVSLVDESLLVGK", "chain_id": "B"}]}
```

## Output Format

The output JSONL file will have the same structure but with updated `msa_path` fields:

```json
{"datapoint_id": "affinity", "task_type": "protein_ligand", "proteins": [{"id": "A", "sequence": "MVTPEGNVSLVDESLLVGV", "chain_id": "A", "msa_path": "./msa_outputs/1c14d6d69fe4.csv"}], "ligand": {"id": "B", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O", "chain_id": "Z"}}
```

## How It Works

1. **Entry Processing**: Each line in the JSONL file is processed separately, creating a unique hash for each entry based on its protein sequences.

2. **FASTA Generation**: 
   - For single protein entries: Creates a standard FASTA file
   - For multiple protein entries: Concatenates sequences with ":" separator

3. **MSA Generation**: Uses ColabFold search to generate MSAs, then processes the A3M output into CSV format using the existing `A3MProcessor`.

4. **Path Assignment**:
   - Single protein entries: One MSA file shared across all proteins in the entry
   - Multiple protein entries: Multiple MSA files, one per protein

5. **Output**: Updates the original JSONL with MSA paths and writes to the output file.

## Example

For the test dataset with two entries:
- Entry 1 (affinity): 1 protein sequence → 1 MSA file
- Entry 2 (complex): 2 protein sequences → 2 MSA files (or 1 shared file for concatenated sequences)

The script will create MSA files named with entry hashes (e.g., `1c14d6d69fe4.csv`, `4c0f4851dea2_0.csv`, `4c0f4851dea2_1.csv`) in the specified MSA directory.
