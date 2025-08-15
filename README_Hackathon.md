# Boltz Hackathon Template 🧬

Welcome to the Boltz Hackathon! This template provides a scaffold for participants to improve the [Boltz](https://github.com/jwohlwend/boltz) protein structure prediction model.

## Quick Start

```bash
# Run a single protein-ligand prediction
python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json

# Run a single protein complex prediction  
python predict_hackathon.py --input-json examples/specs/example_protein_complex.json

# Run multiple predictions from a dataset
python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl

# Check your submissions
python runner/validate_submission.py
```

After running, check the results:
- `inputs/` - Generated YAML files for Boltz
- `predictions/` - Raw Boltz outputs
- `submission/` - Clean submission files for evaluation

## For Participants 🎯

You only need to modify **four functions** in `predict_hackathon.py`:

1. **`get_custom_args(datapoint_id)`** - Add custom CLI arguments for Boltz
2. **`inputs_to_yaml(datapoint_id, proteins, ligand, out_dir)`** - Customize YAML generation for Boltz
3. **`predict_protein_complex(datapoint_id, proteins)`** - Custom logic for protein complexes  
4. **`predict_protein_ligand(datapoint_id, protein, ligand)`** - Custom logic for protein-ligand interactions

### Example Modifications

```python
def get_custom_args(datapoint_id: str) -> List[str]:
    """Customize Boltz prediction parameters."""
    return [
        "--diffusion_samples", "10",      # Generate 10 models instead of 5
        "--recycling_steps", "5",         # More recycling steps
        "--num_sampling_steps", "200"     # Custom sampling
    ]

def inputs_to_yaml(datapoint_id: str, proteins: Iterable[Protein], 
                   ligand: Optional[SmallMolecule] = None, 
                   out_dir: Path = Path("inputs")) -> Path:
    """Customize YAML generation for Boltz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ypath = out_dir / f"{datapoint_id}.yaml"
    
    seqs = []
    for p in proteins:
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": p.msa_path,
                "chain_id": p.chain_id,
                # Add custom fields like structure templates, constraints, etc.
                # "template": "path/to/template.pdb",
                # "constraints": {"distance": [{"atom1": "A:1:CA", "atom2": "B:10:CA", "distance": 8.0}]}
            }
        }
        seqs.append(entry)
    
    if ligand:
        l = {
            "ligand": {
                "id": ligand.id,
                "smiles": ligand.smiles,
                "chain_id": ligand.chain_id,
                # Add custom ligand properties
                # "conformers": 10,
                # "charge": 0
            }
        }
        seqs.append(l)
    
    doc = {
        "version": 1,
        "sequences": seqs,
        # Add global settings
        # "pocket": {"center": [0, 0, 0], "radius": 10.0}
    }
    
    with open(ypath, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    
    return ypath

def predict_protein_complex(datapoint_id: str, proteins: List[Protein]) -> None:
    """Add custom pre/post-processing for complexes."""
    # Your custom preprocessing here
    print(f"Processing {len(proteins)} proteins for complex {datapoint_id}")
    
    # Call the standard pipeline (or replace with your approach)
    yaml_path = inputs_to_yaml(datapoint_id, proteins=proteins, ligand=None, out_dir=DEFAULT_INPUTS_DIR)
    _run_boltz_and_collect(datapoint_id, yaml_path)
    
    # Your custom postprocessing here
    print(f"Finished complex prediction for {datapoint_id}")
```

## Input Format 📝

Each datapoint must include a `task_type` field with either `"protein_complex"` or `"protein_ligand"`:

**Required Fields:**
- `datapoint_id`: Unique identifier for the prediction
- `task_type`: Either `"protein_complex"` or `"protein_ligand"`
- `proteins`: List of protein objects with `id`, `sequence`, `msa_path`, and `chain_id`
- `ligand` (for protein_ligand tasks): Ligand object with `id`, `smiles`, and `chain_id`

### Protein-Ligand Example
```json
{
  "datapoint_id": "my_ligand_prediction",
  "task_type": "protein_ligand",
  "proteins": [
    {
      "id": "A",
      "sequence": "MVTPEGNVSLVDESLLVGV...",
      "msa_path": "path/to/seq1.a3m",
      "chain_id": "A"
    }
  ],
  "ligand": {
    "id": "ligand1", 
    "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "chain_id": "Z"
  }
}
```

### Protein Complex Example  
```json
{
  "datapoint_id": "my_complex_prediction",
  "task_type": "protein_complex",
  "proteins": [
    {
      "id": "A",
      "sequence": "MVTPEGNVSLVDESLLVGV...",
      "msa_path": "path/to/seq1.a3m",
      "chain_id": "A"
    },
    {
      "id": "B", 
      "sequence": "GKTPEGNVSLVDESLLVGV...",
      "msa_path": "path/to/seq2.a3m",
      "chain_id": "B"
    }
  ]
}
```

## Technical Details ⚙️

### Command Line Interface

The script accepts two mutually exclusive input options:
- `--input-json path/to/single_datapoint.json` - Process a single datapoint
- `--input-jsonl path/to/dataset.jsonl` - Process multiple datapoints

### Under the Hood

For each datapoint, the pipeline:
1. Routes to the appropriate prediction function based on `task_type`
2. Converts inputs to Boltz YAML format
3. Runs: `boltz predict inputs/{id}.yaml --devices 1 --out_dir predictions --cache $BOLTZ_CACHE --output_format pdb [custom_args]`
4. Copies model files to `submission/{datapoint_id}/model_{0-4}.pdb`

### Environment Variables
- `BOLTZ_CACHE` - Cache directory for Boltz models (defaults to `~/.boltz`)

### Dependencies
Add any additional Python packages to `pyproject.toml` under the `[project.dependencies]` section.

## Submission Format 📦

Your final predictions should be in:
```
submission/
├── {datapoint_id_1}/
│   ├── model_0.pdb
│   ├── model_1.pdb
│   ├── model_2.pdb
│   ├── model_3.pdb
│   └── model_4.pdb
└── {datapoint_id_2}/
    ├── model_0.pdb
    └── ...
```

## Validation 🧪

```bash
# Validate your submission format
python runner/validate_submission.py

# Check if your modifications work
python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json
```

## Tips for Success 💡

1. **Start Simple** - The default implementation already runs vanilla Boltz predictions
2. **Test Frequently** - Use the validation script to catch issues early  
3. **Check Memory** - Boltz can be memory-hungry on large proteins
4. **MSA Quality** - MSA files are provided, but you can potentially improve them
5. **Sampling** - Experiment with `--diffusion_samples`, `--recycling_steps` etc.
6. **Postprocessing** - Consider confidence-based model selection or refinement
7. **YAML Customization** - The `inputs_to_yaml` function is a powerful way to add Boltz features like templates, constraints, or pocket definitions

## YAML Customization Examples 🔧

The `inputs_to_yaml` function gives you full control over the input to Boltz. Here are some advanced features you can enable:

### Structure Templates
```python
entry = {
    "protein": {
        "id": p.id,
        "sequence": p.sequence,
        "msa": p.msa_path,
        "template": "path/to/reference_structure.pdb"  # Use existing structure as template
    }
}
```

### Distance Constraints  
```python
doc = {
    "version": 1,
    "sequences": seqs,
    "constraints": {
        "distance": [
            {"atom1": "A:1:CA", "atom2": "B:10:CA", "distance": 8.0, "tolerance": 1.0}
        ]
    }
}
```

### Pocket Prediction
```python
doc = {
    "version": 1,
    "sequences": seqs,
    "pocket": {
        "center": [25.0, 30.0, 15.0],  # Coordinates of pocket center
        "radius": 12.0                  # Pocket radius in Angstroms  
    }
}
```

### Affinity Prediction
```python
doc = {
    "version": 1,
    "sequences": seqs,
    "affinity": {
        "label": True,  # Enable affinity prediction
        "target": 7.5   # Optional: target affinity value for training
    }
}
```

## Common Boltz Parameters 🔧

Some useful flags for `get_custom_args()`:
- `--diffusion_samples N` - Generate N models (default 5)
- `--recycling_steps N` - More recycling for better accuracy
- `--num_sampling_steps N` - Diffusion sampling steps
- `--output_format pdb|mmcif` - Output format (we use PDB for submissions)

## Directory Structure

```
boltz-hackathon-template/
├── predict_hackathon.py              # Main script (modify the 4 functions here!)
├── hackathon_api.py                  # Data models (Protein, SmallMolecule)
├── requirements.txt                  # Add your dependencies here
├── examples/
│   ├── specs/                        # Example single datapoints
│   └── test_dataset.jsonl           # Example dataset
├── runner/
│   └── validate_submission.py       # Validate output format
├── inputs/                          # Generated YAML files (auto-created)
├── predictions/                     # Raw Boltz outputs (auto-created) 
└── submission/                      # Final submission (auto-created)
```

## Need Help? 🆘

1. Check the example files in `examples/`
2. Run `python predict_hackathon.py --help`
3. Test with simple cases first
4. Review the [Boltz documentation](https://github.com/jwohlwend/boltz)

Good luck! 🚀
