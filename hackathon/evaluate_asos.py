#!/usr/bin/env python3
"""
Evaluate ASOS predictions by calculating ligand RMSD values.

This script aligns predicted structures with experimental structures using PyMOL,
then calculates RMSD values for ligands using the Hungarian algorithm for optimal assignment.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from Bio import PDB
import json
import matplotlib.pyplot as plt
import tempfile
import shutil


def best_fit_rmsd(exp, pred):
    """
    Calculate RMSD using the Hungarian algorithm for optimal atom assignment.
    
    Args:
        exp: Experimental coordinates (numpy array)
        pred: Predicted coordinates (numpy array)
    
    Returns:
        RMSD value
    """
    # Calculate pairwise distances
    distances = np.linalg.norm(exp[:, np.newaxis] - pred[np.newaxis, :], axis=2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(distances)

    # Calculate the RMSD based on the optimal assignment
    rmsd = np.sqrt(np.mean(np.sum((exp[row_indices] - pred[col_indices]) ** 2, axis=1)))

    return rmsd


def get_coordinates(structure, ligand_name):
    """
    Extract coordinates for a specific ligand from a PDB structure.
    
    Args:
        structure: BioPython structure object
        ligand_name: Residue name of the ligand
    
    Returns:
        List of coordinates [[x, y, z], ...]
    """
    coordinates = []  # Initialize an empty list to store coordinates

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == f"{ligand_name}":
                    for atom in residue.get_atoms():
                        vector = atom.get_vector()  # Get the vector for the atom
                        coordinates.append(vector)  # Append the vector to the list

    # Now, if you want to extract just the x, y, z values:
    coordinates_list = [[vector[0], vector[1], vector[2]] for vector in coordinates]

    return coordinates_list


def get_ligand_rmsd(exp_file, pred_file, ligand_name_exp, ligand_name_pred):
    """
    Calculate ligand RMSD between experimental and predicted structures.
    
    Args:
        exp_file: Path to experimental structure file
        pred_file: Path to predicted structure file
        ligand_name_exp: Ligand residue name in experimental structure
        ligand_name_pred: Ligand residue name in predicted structure
    
    Returns:
        RMSD value
    """
    parser = PDB.PDBParser()
    structure_exp = parser.get_structure('experiment', exp_file)

    parser = PDB.PDBParser()
    structure_pred = parser.get_structure('model', pred_file)

    coordinates_exp = get_coordinates(structure_exp, f"{ligand_name_exp}")
    coordinates_pred = get_coordinates(structure_pred, f"{ligand_name_pred}")

    exp = np.array(coordinates_exp)
    pred = np.array(coordinates_pred)

    # Calculate best-fit RMSD
    rmsd = best_fit_rmsd(exp, pred)

    return rmsd


def load_dataset(dataset_file):
    """
    Load the ASOS dataset and extract ligand information.
    
    Args:
        dataset_file: Dataset filename
    
    Returns:
        Tuple of (dataset, ligand_info)
    """
    dataset = [json.loads(line) for line in open(dataset_file, 'r')]
    print(f"Loaded {len(dataset)} samples from the dataset.")

    ligand_info = {}
    for datapoint in dataset:
        for ligand in datapoint['ground_truth']['ligand_types']:
            ligand_id = datapoint['datapoint_id']
            if ligand_id not in ligand_info:
                ligand_info[ligand_id] = []  # Initialize a list for each ligand ID
            ligand_info[ligand_id].append({  # Append the type and CCD information
                'type': ligand['type'],
                'ccd': ligand['ccd'],
                'chain_prot': ligand['chain'],
                'chain_lig': ligand.get("ligand_chain") or ligand["ligand_id"],  # Default to "L" if not provided
            })

    return dataset, ligand_info


def align_structures(dataset, ligand_info, dataset_folder, submission_folder, tempfolder):
    """
    Align predicted structures with experimental structures using PyMOL.
    
    Args:
        dataset: Dataset list
        ligand_info: Dictionary of ligand information
        dataset_folder: Path to dataset folder
        submission_folder: Path to submission folder with predictions
        tempfolder: Temporary folder for aligned structures
    """
    # Clean and create temp folder
    shutil.rmtree(tempfolder, ignore_errors=True)
    os.makedirs(tempfolder, exist_ok=True)

    # Generate PyMOL alignment script
    for datapoint in dataset:
        datapoint_id = datapoint['datapoint_id']
        gt_structure = datapoint['ground_truth']["structure"]
        keep = []
        for chain in ligand_info[datapoint_id]:
            keep.append(chain["chain_prot"])
            keep.append(chain["chain_lig"])

        keep_selection = "+".join(keep)
        with open(os.path.join(tempfolder, "align.pml"), "a") as a:
            a.write(f"load {dataset_folder}/ground_truth/{gt_structure}, exp\n")
            a.write(f"sele not chain {keep_selection}\n")
            a.write(f"remove sele\n")
            for model in range(5):
                a.write(f"load {submission_folder}/{datapoint_id}/model_{model}.pdb, pred_{model}\n")
                a.write(f"align pred_{model}, exp\n")
                a.write(f"save {tempfolder}/{datapoint_id}_model_{model}.pdb, pred_{model}\n")
            a.write(f"save {tempfolder}/{datapoint_id}_exp.pdb, exp\n")
            a.write("delete all\n")

    # Run PyMOL alignment
    sh_file = os.path.join(tempfolder, "pymol_align.sh")
    with open(sh_file, "w") as p:
        p.write(f"pymol -c {os.path.join(tempfolder, 'align.pml')}\n")

    os.system("chmod u+x " + sh_file)
    print("Running PyMOL alignment...")
    os.system(sh_file)
    print("PyMOL alignment complete.")


def calculate_rmsds(dataset, ligand_info, tempfolder):
    """
    Calculate RMSD values for all ligands across all models.
    
    Args:
        dataset: Dataset list
        ligand_info: Dictionary of ligand information
        tempfolder: Temporary folder with aligned structures
    
    Returns:
        Dictionary of RMSD values
    """
    rmsds = {}
    for datapoint in dataset:
        exp_file = os.path.join(tempfolder, f"{datapoint['datapoint_id']}_exp.pdb")
        try:
            for ligand_i in range(len(ligand_info[datapoint['datapoint_id']])):
                ligand = ligand_info[datapoint['datapoint_id']][ligand_i]
                rmsds_ligand = []
                for model in range(5):
                    pred_file = os.path.join(tempfolder, f"{datapoint['datapoint_id']}_model_{model}.pdb")
                    rmsds_ligand.append(get_ligand_rmsd(exp_file, pred_file, ligand["ccd"], "LIG"))

                rmsds[f"{datapoint['datapoint_id']}"] = {
                    "rmsd": rmsds_ligand,
                    "type": ligand['type']
                }

        except Exception as e:
            print(f"Unsuccessful for {datapoint['datapoint_id']}")
            import traceback
            traceback.print_exc()
    
    return rmsds


def plot_results(rmsds, result_folder):
    """
    Create boxplots for orthosteric and allosteric ligand RMSD values.
    
    Args:
        rmsds: Dictionary of RMSD values
        result_folder: Folder to save plots
    """
    os.makedirs(result_folder, exist_ok=True)
    
    # Create a horizontal boxplot for orthosteric ligands
    rmsd_subset = {key: value["rmsd"] for key, value in rmsds.items() if value["type"] == "orthosteric"}
    if rmsd_subset:
        data = [values for values in rmsd_subset.values()]
        labels = list(rmsd_subset.keys())
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, vert=False)  # vert=False makes the boxplot horizontal
        plt.yticks(range(1, len(labels) + 1), labels)  # Set the y-ticks to the model names
        plt.xlabel('RMSD Values')
        plt.title('Orthosteric Ligand RMSD Values')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, 'orthosteric_rmsd.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved orthosteric RMSD plot to {result_folder}/orthosteric_rmsd.png")

    # Create a horizontal boxplot for allosteric ligands
    rmsd_subset = {key: value["rmsd"] for key, value in rmsds.items() if value["type"] == "allosteric"}
    if rmsd_subset:
        data = [values for values in rmsd_subset.values()]
        labels = list(rmsd_subset.keys())
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, vert=False)  # vert=False makes the boxplot horizontal
        plt.yticks(range(1, len(labels) + 1), labels)  # Set the y-ticks to the model names
        plt.xlabel('RMSD Values')
        plt.title('Allosteric Ligand RMSD Values')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, 'allosteric_rmsd.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved allosteric RMSD plot to {result_folder}/allosteric_rmsd.png")


def save_results(rmsds, result_folder):
    """
    Save RMSD results to a JSON file.
    
    Args:
        rmsds: Dictionary of RMSD values
        result_folder: Folder to save results
    """
    os.makedirs(result_folder, exist_ok=True)

    # CSV with combined results
    output_file = os.path.join(result_folder, 'combined_results.csv')
    df_metrics = pd.DataFrame([
        {
            "datapoint_id": key,
            "type": value["type"],
            "top1_rmsd": value["rmsd"][0],
            "top5_mean_rmsd": np.mean(value["rmsd"][:5]),
            "top5_min_rmsd": np.min(value["rmsd"][:5]),
            "rmsd_under_2A": any(rmsd < 2.0 for rmsd in value["rmsd"][:5]),
            "rmsd_model_0": value["rmsd"][0],
            "rmsd_model_1": value["rmsd"][1],
            "rmsd_model_2": value["rmsd"][2],
            "rmsd_model_3": value["rmsd"][3],
            "rmsd_model_4": value["rmsd"][4],
        }
        for key, value in rmsds.items()
    ])
    df_metrics.to_csv(output_file, index=False)
    print(f"Saved metrics results to {output_file}")

    return df_metrics

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ASOS predictions by calculating ligand RMSD values.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset-folder',
        required=False,
        help='Path to the dataset folder containing the JSONL file and ground_truth subdirectory'
    )
    parser.add_argument(
        '--dataset-file',
        required=True,
        help='Name of the dataset JSONL file'
    )
    parser.add_argument(
        '--submission-folder',
        required=True,
        help='Path to the submission folder containing predicted structures'
    )
    parser.add_argument(
        '--result-folder',
        default='./evaluation_results',
        help='Path to save evaluation results and plots'
    )
    parser.add_argument(
        '--temp-folder',
        default='./tmp',
        help='Path to temporary folder for aligned structures'
    )
    
    args = parser.parse_args()

    print("=" * 80)
    print("ASOS Ligand RMSD Evaluation")
    print("=" * 80)
    print(f"Dataset folder: {args.dataset_folder}")
    print(f"Dataset file: {args.dataset_file}")
    print(f"Submission folder: {args.submission_folder}")
    print(f"Result folder: {args.result_folder}")
    print(f"Temp folder: {args.temp_folder}")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    if not args.dataset_folder:
        args.dataset_folder = os.path.dirname(os.path.abspath(args.dataset_file))
        
    dataset, ligand_info = load_dataset(args.dataset_file)
    print(f"Found ligand information for {len(ligand_info)} datapoints")

    # Align structures
    print("\n2. Aligning structures with PyMOL...")
    align_structures(dataset, ligand_info, args.dataset_folder, args.submission_folder, args.temp_folder)

    # Calculate RMSDs
    print("\n3. Calculating ligand RMSD values...")
    rmsds = calculate_rmsds(dataset, ligand_info, args.temp_folder)
    print(f"Calculated RMSD for {len(rmsds)} ligands")

    # Save results
    print("\n4. Saving results...")
    df_metrics = save_results(rmsds, args.result_folder)

    # Plot results
    print("\n5. Generating plots...")
    plot_results(rmsds, args.result_folder)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Top 1 model RMSD summary:")
    # all, orthosteric, allosteric
    for ligand_type in [None, "orthosteric", "allosteric"]:
        if ligand_type:
            df_filtered = df_metrics[df_metrics['type'] == ligand_type]
            label = ligand_type.capitalize()
        else:
            df_filtered = df_metrics
            label = "All"
        
        if not df_filtered.empty:
            mean_top1_rmsd = df_filtered['top1_rmsd'].mean()
            mean_top5_min_rmsd = df_filtered['top5_min_rmsd'].mean()
            num_below_2A = df_filtered['rmsd_under_2A'].sum()
            total = len(df_filtered)
            print(f"{label} ligands - Mean Top 1 RMSD: {mean_top1_rmsd:.2f}, Mean Top 5 Min RMSD: {mean_top5_min_rmsd:.2f}, "
                  f"RMSD < 2Å in Top 5: {num_below_2A}/{total} ({(num_below_2A/total)*100:.1f}%)")
        else:
            print(f"No data for {label} ligands.")
    print("=" * 80)


if __name__ == "__main__":
    main()
