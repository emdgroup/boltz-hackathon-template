#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
import pandas as pd

from hackathon_api import Datapoint

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel CAPRI-Q evaluation runner (Python port)")
    parser.add_argument('--dataset-file', type=str, default=str(Path.cwd() / 'inputs'), help='Path to input JSONL file')
    parser.add_argument('--result-folder', type=str, default=str(Path.cwd() / 'outputs'), help='Directory to store result files')
    parser.add_argument('--submission-folder', type=str, default=str(Path.cwd() / 'predictions'), help='Directory containing prediction files')
    parser.add_argument('--njobs', type=int, default=50, help='Number of parallel jobs to run')
    parser.add_argument('--nsamples', type=int, default=5, help='Number of samples to evaluate per structure')
    return parser.parse_args()


def run_evaluation(gt_dir, gt_structures: dict[str, Any], structure_name: str, i: int, args) -> Optional[pd.DataFrame]:
    output_subdir = Path(args.result_folder) / f"{structure_name}_{i}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    prediction_file = Path(args.submission_folder) / structure_name / f"model_{i}.pdb"
    if not prediction_file.exists():
        print(f"No prediction file {prediction_file} found. Skipping.")
        return None
    

    capriq_cmd = [
        "/capri-q/bin/capriq",
        "-a", "--dontwrite",
        "-t", f"/app/ground_truth/{gt_structures['structure_complex']}",
        "-u", f"/app/ground_truth/{gt_structures['structure_ab']}",
        "-u", f"/app/ground_truth/{gt_structures['structure_ligand']}",
        "-z", "/app/outputs/",
        "-p", "65",
        "-o", f"/app/outputs/{structure_name}_{i}_results.txt",
        "-l", f"/app/outputs/{structure_name}_{i}_errors.txt",
        f"/app/predictions/prediction.pdb",
        "&&",
        "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/app/outputs"
    ]

    docker_cmd = [
        "docker", "run", "--group-add", str(os.getgid()), "--rm", "--network", "none",
        "-v", f"{Path(gt_dir).absolute()}:/app/ground_truth/",
        "-v", f"{output_subdir.absolute()}:/app/outputs",
        "-v", f"{prediction_file.absolute()}:/app/predictions/prediction.pdb",
        "gitlab-registry.in2p3.fr/cmsb-public/capri-q",
        "/bin/bash", "-c", 
        f"{' '.join(capriq_cmd)}"
    ]
    print(f"Evaluating {structure_name} model {i}... Prediction file: {prediction_file}")
    # print(f"Docker command: {' '.join(docker_cmd)}")
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Docker run failed for {structure_name} model {i}. Error: {e}", file=sys.stderr)
        return pd.DataFrame({
            'structure_name': [structure_name],
            'structure_index': [i],
            'nclash': [None],
            'clash_fraction': [None],
            'classification': ['error'],
            'error': [str(e)]
        })

    # load result file
    result_file = output_subdir / f"{structure_name}_{i}_results.txt"
    if not result_file.exists():
        print(f"No result file {result_file} found. Skipping.")
        return pd.DataFrame({
            'structure_name': [structure_name],
            'structure_index': [i],
            'nclash': [None],
            'clash_fraction': [None],
            'classification': ['error'],
            'error': ['Result file not found']
        })
    
    df = pd.read_csv(result_file, sep='\\s+')
    df['clash_fraction'] = df['model'].str.replace("/", "").astype(float) / df['nclash']
    df['nclash'] = df['model'].str.replace("/", "").astype(int)
    df.drop(columns=['model'], inplace=True)
    df['structure_name'] = structure_name
    df['structure_index'] = i
    return df

def load_dataset(input_jsonl: str) -> list[Datapoint]:
    with open(input_jsonl, 'r') as f:
        data = [Datapoint.from_json(line) for line in f]
    return data

def main():
    args = parse_args()
    input_jsonl = args.dataset_file
    dataset = load_dataset(input_jsonl)
    gt_dir = Path(args.dataset_file).parent / 'ground_truth'
    result_dfs = []
    with ThreadPoolExecutor(max_workers=args.njobs) as executor:
        futures = []
        for datapoint in dataset:
            structure_name = datapoint.datapoint_id
            gt_structures = datapoint.ground_truth
            for i in range(args.nsamples):
                futures.append(executor.submit(run_evaluation, gt_dir, gt_structures, structure_name, i, args))
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                result_dfs.append(result)
    combined_results = pd.concat(result_dfs, ignore_index=True)
    combined_results.to_csv(Path(args.result_folder) / 'combined_results.csv', index=False)

    # select structure 0 and count "classification"
    nsuccessful = 0
    good_classes = ['high', 'medium', 'acceptable']
    bad_classes = ['incorrect', 'error']
    for classification in good_classes + bad_classes:
        n = len(combined_results[(combined_results['structure_index'] == 0) & (combined_results['classification'].str.contains(classification))])
        if classification in good_classes:
            nsuccessful += n
        print(f"Number of {classification} classifications in top 1: {n}")

    # print number of successful top 1 predictions
    print(f"Number of successful top 1 predictions: {nsuccessful} out of {len(dataset)}")
    
    print("All evaluations completed.")

if __name__ == "__main__":
    main()
