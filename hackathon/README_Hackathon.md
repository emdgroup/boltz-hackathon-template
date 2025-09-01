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
You can use `post_process_protein_complex`, e.g., to modify, combine or re-rank the predicted structures.

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

### Boltz code

You are also welcome to make modifications to the Boltz code as needed.

### Dependencies

Add any additional Python packages to `pyproject.toml` under the `[project.dependencies]` section.
If you need non-Python dependencies, you can add those in `environment.yml`.
We strongly advice against adding non-Python dependencies that are not available through any public conda channel.
If you still want to add them, please install them directly in your machine and make sure that you modify `Dockerfile` accordingly.

## Evaluation Limits

When evaluating your contributions your code will run in an environment with the following hardware specs:

- 1x NVIDIA L40 GPU (48GB)
- 32 CPU cores
- 300 GB RAM

On this machine the full end-to-end prediction for a single datapoint, including pre-processing, Boltz prediction, post-processing, should complete within 15 minutes on average. 
As a reference, one typical antibody-antigen complex with 5 diffusion samples and default settings takes around 80-90 seconds end-to-end on that kind of hardware.

## Validation Sets

For both challenges we provide an evaluation set.

For the antibody-antigen complex challenge, we have 12 public PDB structures, all released after the cut-off date for Boltz training data.
For the allosteric-orthosteric ligand challenge, we have 40 structures that were also used in the recent paper of Nittinger et. al [1].

To run the prediction and evaluation for the antibody-antigen complex challenge, use the following command:

```bash
TODO
```

To run the prediction and evaluation for the allosteric-orthosteric ligand challenge, use the following command:

```bash
TODO
```

## Submission Format 📦

Your final predictions should be in:
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

## Directory Structure

**TODO**

## Need Help? 🆘

**TODO**

## References

1. Nittinger, Eva, et al. "Co-folding, the future of docking – prediction of allosteric and orthosteric ligands." Artificial Intelligence in the Life Sciences, vol. 8, 2025, p. 100136. Elsevier,

Good luck! 🚀
