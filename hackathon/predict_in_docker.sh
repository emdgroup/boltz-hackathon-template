
#!/bin/bash

# Usage:
#   ./predict_in_docker.sh --image <docker_image_tag> --dataset <dataset_path> --submission-dir <submission_dir> --msa-dir <msa_dir> --boltz-cache-dir <boltz_cache_dir>

set -e

show_help() {
    echo "Usage: $0 --image <docker_image_tag> --dataset <dataset_path> --submission-dir <submission_dir> --msa-dir <msa_dir> --boltz-cache-dir <boltz_cache_dir>"
    exit 1
}

# Parse named parameters
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --image)
            DOCKER_IMAGE_TAG="$2"
            shift; shift
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift; shift
            ;;
        --submission-dir)
            SUBMISSION_DIR="$2"
            shift; shift
            ;;
        --msa-dir)
            MSA_DIR="$2"
            shift; shift
            ;;
        --boltz-cache-dir)
            BOLTZ_CACHE_DIR="$2"
            shift; shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            ;;
    esac
done

# Check required parameters
if [[ -z "$DOCKER_IMAGE_TAG" || -z "$DATASET_PATH" || -z "$SUBMISSION_DIR" || -z "$MSA_DIR" || -z "$BOLTZ_CACHE_DIR" ]]; then
    show_help
fi

# Make sure submission directory exists
mkdir -p "$SUBMISSION_DIR"

# Run the docker container with the required mounts and arguments
# set -x
docker run --rm \
    --gpus ${CUDA_VISIBLE_DEVICES:+device=$CUDA_VISIBLE_DEVICES} ${CUDA_VISIBLE_DEVICES:-all} \
    --network none \
    --shm-size=16G \
    --mount type=bind,source=$HOME/.ssh/cacert.pem,target=/etc/ssl/certs/cacert.pem,readonly \
    -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/cacert.pem \
    -e CURL_CA_BUNDLE=/etc/ssl/certs/cacert.pem \
    -e SSL_CERT_FILE=/etc/ssl/certs/cacert.pem \
    -e BOLTZ_CACHE=/db/boltz \
    -v "${DATASET_PATH}:/app/dataset.jsonl:ro" \
    -v "${SUBMISSION_DIR}:/app/submission:rw" \
    -v "${MSA_DIR}:/app/msa:ro" \
    -v "${BOLTZ_CACHE_DIR}:/db/boltz:rw" \
    -it \
    "${DOCKER_IMAGE_TAG}" \
    conda run -n boltz python hackathon/predict_hackathon.py --input-jsonl "/app/dataset.jsonl" --msa-dir "/app/msa"
