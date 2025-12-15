#!/bin/bash


# source ~/miniconda3/etc/profile.d/conda.sh
source activate base
conda activate autoalbument

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/search_$TIMESTAMP.log"

export CUBLAS_WORKSPACE_CONFIG=:0:0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to log messages
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# Start search
log "Starting AutoAlbument search..."
log "Configuration directory: $(pwd)"
log "Output directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

# Ensure local dataset.py is found first
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run AutoAlbument search
CONFIG_DIR=$(pwd)
# check if config dir ends with brightness_low_task
if [[ $CONFIG_DIR != *brightness_low_task* ]]; then
    echo "Please run this script from the brightness_low_task directory"
    exit 1
fi
log "Running: autoalbument-search --config-dir $CONFIG_DIR"

conda activate autoalbument

autoalbument-search \
    --config-dir "$CONFIG_DIR"
