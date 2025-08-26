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

# Install PyTorch with CUDA support
echo -e "\n${GREEN}Installing PyTorch with CUDA support...${NC}"
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install AutoAlbument and dependencies
echo -e "\n${GREEN}Installing AutoAlbument and dependencies...${NC}"
pip install --upgrade pip
pip install timm==0.4.12
pip install autoalbument
pip install albumentations==1.3.0
pip install hydra-core --upgrade
pip install omegaconf
pip install opencv-python-headless
pip install tqdm

# Install development tools (optional)
echo -e "\n${GREEN}Installing development tools...${NC}"
pip install black isort flake8 jupyter

# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import autoalbument; print(f'AutoAlbument version: {autoalbument.__version__}')"

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "To activate the environment, run: ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo -e "To run the search, execute: ${YELLOW}./run_autoalbument.sh${NC}"
