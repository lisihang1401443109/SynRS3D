#!/bin/bash

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
# source ~/miniconda3/etc/profile.d/conda.sh
source activate base
conda activate autoalbument

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/search_$TIMESTAMP.log"

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
log "Running: autoalbument-search --config-dir $(pwd) --config-name search hydra.run.dir=$OUTPUT_DIR"

conda activate autoalbument

autoalbument-search \
    --config-dir . \
    --config-name search \
    hydra.run.dir="$OUTPUT_DIR"

# Check if search was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "AutoAlbument search completed successfully!"
    
    # Check for best policy
    BEST_POLICY="$OUTPUT_DIR/policy/best_policy.json"
    if [ -f "$BEST_POLICY" ]; then
        log "Best policy saved to: $BEST_POLICY"
        log "To use this policy in training, set policy_dir to: $(dirname "$BEST_POLICY")"
        
        # Create a symlink to the latest results
        LATEST_LINK="outputs/latest"
        rm -f "$LATEST_LINK"
        ln -s "$(basename "$OUTPUT_DIR")" "$LATEST_LINK"
        log "Created symlink: $LATEST_LINK -> $(basename "$OUTPUT_DIR")"
    else
        log "Warning: Best policy not found in expected location: $BEST_POLICY"
    fi
else
    log "AutoAlbument search failed. Check the logs in: $LOG_FILE"
    exit 1
fi
