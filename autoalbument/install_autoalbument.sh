#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up AutoAlbument environment...${NC}"

# source activate base

# # Check if conda is installed
# if ! command -v conda &> /dev/null; then
#     echo -e "${YELLOW}Conda not found. Please install Miniconda or Anaconda first.${NC}"
#     echo "You can download it from: https://docs.conda.io/en/latest/miniconda.html"
#     exit 1
# fi


# Create conda environment
ENV_NAME="autoalbument"
echo -e "\n${GREEN}Creating conda environment '${ENV_NAME}'...${NC}"
conda create -n $ENV_NAME python=3.8 -y

# Activate the environment
echo -e "\n${GREEN}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME


# Install AutoAlbument and dependencies
echo -e "\n${GREEN}Installing AutoAlbument and dependencies...${NC}"
# pip install --upgrade pip
python -m pip install --upgrade pip==23
pip install -U autoalbument


# Install specific versions of required packages
echo -e "\n${GREEN}Installing required packages with specific versions...${NC}"
pip install six
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install segmentation-models-pytorch==0.1.3
pip install hydra-core==1.0.6
pip install PyYAML==5.3.1


# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import autoalbument; print(f'AutoAlbument version: {autoalbument.__version__}')"

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "To activate the environment, run: ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo -e "To run the search, execute: ${YELLOW}./run_autoalbument.sh${NC}"
