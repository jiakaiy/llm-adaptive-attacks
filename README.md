# LLM Adaptive Attacks

A research project focused on adaptive attack methods for Large Language Models.

## Overview

This repository contains code and experiments for studying adaptive attacks on LLMs, including various attack strategies and evaluation methods.

## Project Structure

- `*.py` - Python scripts for different attack methods and utilities
- `*.jsonl` - Data files containing prompts, results, and experimental data
- `experiments/` - Experimental scripts and configurations
- `attack_logs/` - Log files from attack experiments
- `harmful_behaviors/` - Dataset of harmful behavior prompts
- `jailbreak_artifacts/` - Generated jailbreak artifacts
- `images/` - Visualization and result images

## Key Components

### Main Scripts
- `main.py` - Main execution script for attacks
- `sft_Qwen_train.py` - Training script for Qwen model fine-tuning
- `conversers.py` - Conversation handling utilities
- `language_models.py` - Language model interfaces
- `judges.py` - Evaluation and judging utilities

### Attack Methods
- Various phase-based attack implementations
- Batch processing scripts for different models
- Transfer learning approaches

## Setup

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys for model access (if needed)

## Usage

[Add specific usage instructions here based on your workflow]

## Results

[Add information about your experimental results]

## Contributing

[Add contribution guidelines if applicable]

## License

[Specify your license]