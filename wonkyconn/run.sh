#!/bin/bash

#SBATCH --job-name=wonkyconn
#SBATCH --output=logs/wonkyconn_%j.log
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load required environment
cd /home/seann/scratch/wonkyconn

source /home/seann/scratch/wonkyconn/wonkyconnenv/bin/activate

# Define paths
BIDS_DIR="/home/seann/scratch/wonkyconn/giga_connectome_outputs/connectome_Schaefer2018"  # Replace with your fMRIPrep output directory
OUTPUT_DIR="/home/seann/scratch/wonkyconn/wonkyconn_outputs/test1"          # Replace with desired output directory
PHENOTYPES="${BIDS_DIR}/participants.tsv"       # Replace with path to your phenotypes file
ATLAS_PATH="${BIDS_DIR}/atlases/sub-1/func/sub-1_seg-Schaefer2018400Parcels7Networks_dseg.nii.gz"  # Replace with path to your atlas

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run WONKYCONN
wonkyconn ${BIDS_DIR} ${OUTPUT_DIR} group \
    --phenotypes ${PHENOTYPES} \
    --atlas Schaefer2018400Parcels7Networks ${ATLAS_PATH} \
    --group-by seg desc \
    --verbosity 2
