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

## Quick Start

To participate in the hackathon:

1. **Modify the code**: Edit the functions in `hackathon/predict_hackathon.py`:
   - `prepare_protein_complex()` or `prepare_protein_ligand()` - Customize input configurations and CLI arguments
   - `post_process_protein_complex()` or `post_process_protein_ligand()` - Re-rank or post-process predictions
   - You can also modify any Boltz source code in `src/boltz/` as needed

2. **Run predictions**: Execute the prediction script on a validation dataset:
   ```bash
   python hackathon/predict_hackathon.py \
       --input-jsonl hackathon_datasets/abag_public/abag_public_dataset_msa.jsonl \
       --msa-dir hackathon_datasets/abag_public/msa/ \
       --submission-dir ./my_predictions \
       --intermediate-dir ./tmp/ \
       --result-folder ./my_results
   ```

3. **Evaluate**: Results will be automatically computed and saved to the `--result-folder` directory. 
Review the metrics to assess your improvements.

4. **Iterate**: Refine your approach based on evaluation results and repeat!

## Entrypoints for Participants

### `hackathon/predict_hackathon.py`

We will evaluate your contributions by calling `hackathon/predict_hackathon.py` that will do the following three main steps for each data point of a dataset: 

- generate input yaml files and CLI arguments
- call Boltz
- move predicted structures to a submission folder

Inside `hackathon/predict_hackathon.py`, you can modify the following functions for the antibody-antigen complex prediction challenge:

`def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:`

This function gets as input:

- `datapoint_id`: The ID of the current datapoint
- `proteins`: A list of protein objects to be processed
- `input_dict`: A pre-filled dictionary containing the YAML definition for that data point
- `msa_dir`: The directory with the MSA files

This function should output a **list of tuples**, where each tuple contains:

- A modified `input_dict` with any changes made during preparation
- A list of CLI arguments that should be passed to Boltz for this configuration.

By returning multiple tuples, you can run Boltz with different configurations for the same datapoint (e.g., different sampling strategies, different constraints, different hyperparameters). Each configuration will be run separately with its own YAML file.

You can modify this function, e.g., to tailor the CLI args like changing the number of diffusion samples or recycling steps. Or you could add constraints to the yaml file through modifications to the `input_dict`.

With the provided information, the script will then call Boltz once for each configuration. 
Afterwards, the following function gets called:

`def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:` 

This function receives:
- `datapoint`: The original datapoint object
- `input_dicts`: A list of input dictionaries used (one per configuration)
- `cli_args_list`: A list of CLI arguments used (one per configuration)
- `prediction_dirs`: A list of directories containing prediction results (one per configuration)

The function should return a list of **Path objects** pointing to the PDB files of the predicted structures across all configurations.
The order is important!
The first path will be your top 1 prediction, and we will evaluate up to 5 predictions for each data point.
You can use `post_process_protein_complex`, e.g., to modify, combine or re-rank the predicted structures from multiple configurations.

For the allosteric-orthosteric ligand challenge, there are similar functions:

`def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:`

This function also returns a **list of tuples** to support multiple configurations per datapoint.

and

`def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:`

This function receives lists of configurations and returns a list of **Path objects** pointing to the ranked PDB files.

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

For both challenges we provide a validation data set that you can use to test your contributions and track your progress.

The validation set for the antibody-antigen complex challenge comprises of 12 public PDB structures, all released after the cut-off date for Boltz training data.
The validation set for the allosteric-orthosteric ligand challenge comprises of 40 structures that were also used in the recent paper of Nittinger et. al [1].

To run the prediction and evaluation for the antibody-antigen complex challenge, use:

```bash
python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_datasets/abag_public/abag_public_dataset_msa.jsonl \
    --msa-dir hackathon_datasets/abag_public/msa/ \
    --submission-dir SUBMISSION_DIR \
    --intermediate-dir ./tmp/ \
    --result-folder RESULT_DIR
```

To run the prediction and evaluation for the allosteric-orthosteric ligand challenge, use:

```bash
python hackathon/predict_hackathon.py \
    --input-jsonl hackathon_datasets/asos_public/asos_public_dataset_msa.jsonl \
    --msa-dir hackathon_datasets/asos_public/msa/ \
    --submission-dir SUBMISSION_DIR \
    --intermediate-dir ./tmp/ \
    --result-folder RESULT_DIR
```
In both cases, replace `SUBMISSION_DIR` with the path to a directory where you want to store your predictions and `RESULT_DIR` with the path to a directory where you want to store the evaluation results.
If you do not provide `--result-folder`, the script will only run the predictions and not the evaluation.

## Submission Format 📦

If you make deeper changes to the provided code, make sure your final predictions are organized in the following structure:
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

## Need Help? 🆘

**TODO**

## References

1. Nittinger, Eva, et al. "Co-folding, the future of docking – prediction of allosteric and orthosteric ligands." Artificial Intelligence in the Life Sciences, vol. 8, 2025, p. 100136. Elsevier,

Good luck! 🚀
