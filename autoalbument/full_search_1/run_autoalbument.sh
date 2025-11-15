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

# Run AutoAlbument search
log "Running: autoalbument-search --config-dir /mnt/synrs3d/SynRS3D/autoalbument/full_search"

conda activate autoalbument

autoalbument-search \
    --config-dir /mnt/synrs3d/SynRS3D/autoalbument/full_search_1
