# Makefile for DSC180A-Q1-Project
# Usage:
#   make download      # Download and convert MoCap data
#   make single_frame  # Run matching and visualization on one frame
#   make multi_frame   # Run multi-frame animation
#   make clean         # Remove __pycache__ and temp files

PYTHON = python3

# Default target
all: download multi_frame

# 1. Download and convert MoCap data
download:
	$(PYTHON) scripts/download_and_convert.py

# 2. Multi-frame OT animation
multi_frame:
	$(PYTHON) scripts/multi_frame_ot.py

# 3. Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf *.egg-info