# Boltz Hackathon Template 🧬

Welcome to the Boltz Hackathon!
It is great to have you here!
This template provides a scaffold for participating in the antibody-antigen complex prediction challenge and the allosteric-orthosteric ligand challenge.
Please read these instructions carefully before you start.

This repository is a fork of the [Boltz](https://github.com/jwohlwend/boltz) repository and has been modified for the hackathon to allow a straightforward evaluation of your contributions.

## Setup

Different from the original installation instructions, please set up your environment by first using conda/mamba to create the environment and then use pip to install the Boltz package.

```
git clone https://github.com/emdgroup/boltz-hackathon-template.git
cd boltz
conda env create -f environment.yml --name boltz
conda activate boltz
pip install -e .[cuda]
```

## Entrypoints for Participants

### `hackathon/predict_hackathon.py`

We will evaluate your contributions by calling `hackathon/predict_hackathon.py` that will do the following three main steps for each data point of a dataset: 

- generate input yaml files and CLI arguments
- call Boltz
- move predicted structures to a submission folder

Inside `hackathon/predict_hackathon.py`, you can modify the following functions for the antibody-antigen complex prediction challenge:

`def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> tuple[dict, List[str]]:`

This function gets as input:

- `datapoint_id`: The ID of the current datapoint
- `proteins`: A list of protein objects to be processed
- `input_dict`: A pre-filled dictionary containing the YAML definition for that data point
- `msa_dir`: The directory with the MSA files

This function should output two things:

- A modified `input_dict` with any changes made during preparation
- A list of CLI arguments that should be passed to Boltz for this data point.

You can modify this function, e.g., to tailor the CLI args like changing the number of diffusion samples or recycling steps. Or you could add constraints to the yaml file through modifications to the `input_dict`.

With the provided information, the script will then call Boltz to make the prediction. 
Afterwards, the following function gets called:

`def post_process_protein_complex(datapoint: Datapoint, input_dict: dict[str, Any], cli_args: list[str], prediction_dir: Path) -> List[str]:` 

In addition to the input dictionary and the CLI arguments that were used for this data point, the function also receives the path to the directory containing the predicted structures. 
The function should return a list of file names pointing to the PDB files of the predicted structures.
The order is important!
The first file name will be your top 1 prediction, and we will evaluate up to 5 predictions for each data point.

For the allosteric-orthosteric ligand challenge, there are similar functions:

`def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> tuple[dict, List[str]]:`

and

`def post_process_protein_ligand(datapoint: Datapoint, input_dict: dict[str, Any], cli_args: list[str], prediction_dir: Path) -> List[str]:`

These functions serve as quick start entrypoints. 
Feel free to modify any other part of `hackathon/predict_hackathon.py` as long as the final predictions are stored like

```
{submission_dir}/
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
