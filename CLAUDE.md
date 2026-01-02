# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the implementation of WASB (Widely Applicable Strong Baseline for Sports Ball Detection and Tracking), a sports ball detection and tracking system that works across multiple sports categories including soccer, tennis, badminton, volleyball, and basketball.

## Common Development Commands

### Environment Setup
- Use the provided Dockerfile for consistent environment setup: `docker build -t wasb .`
- For manual setup, ensure you have Python 3.8, CUDA 11.3, and the dependencies listed in the Dockerfile

### Running Evaluations
- Basic evaluation command structure:
  ```
  cd src
  python main.py --config-name=eval dataset=<sport> model=<model_name> detector.model_path=<path_to_weights>
  ```

- Example for evaluating WASB on tennis:
  ```
  cd src
  python main.py --config-name=eval dataset=tennis model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar
  ```

- Example for evaluating with step=1:
  ```
  cd src
  python main.py --config-name=eval dataset=tennis model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1
  ```

### Available Models
- WASB (ours)
- MonoTrack
- ResTrackNetV2
- TrackNetV2
- BallSeg
- DeepBall
- DeepBall-Large

### Available Datasets
- Soccer
- Tennis
- Badminton
- Volleyball
- Basketball

## Code Architecture and Structure

### Core Components

1. **Models (`src/models/`)**
   - Contains implementations of different neural network architectures for ball detection
   - Key models: wasb.py, deepball.py, tracknetv2.py, monotrack.py

2. **Detectors (`src/detectors/`)**
   - Wrappers around models that handle inference and post-processing
   - Main files: detector.py (base class), deepball_detector.py, deepball_postprocessor.py, postprocessor.py

3. **Datasets (`src/datasets/`)**
   - Sport-specific dataset loaders and preprocessing pipelines
   - Files for each sport: badminton.py, basketball.py, soccer.py, tennis.py, volleyball.py

4. **Dataloaders (`src/dataloaders/`)**
   - Handles data loading, transformations, and batching
   - Includes heatmaps generation and image transforms

5. **Runners (`src/runners/`)**
   - Execution controllers that orchestrate the evaluation process
   - Main evaluation logic in eval.py with VideosInferenceRunner class

6. **Configs (`src/configs/`)**
   - YAML configuration files organized by component type
   - Hierarchical configuration using Hydra

7. **Loss Functions (`src/losses/`)**
   - Various loss implementations for training models
   - Heatmap losses, segmentation losses, focal losses

8. **Trackers (`src/trackers/`)**
   - Post-processing modules for temporal consistency
   - intra_frame_peak.yaml, online.yaml configurations

### Data Flow

1. Configuration is loaded via Hydra using `--config-name=eval`
2. Dataset loader prepares data based on config specifications
3. Model is instantiated and loaded with pretrained weights
4. Runner orchestrates the evaluation process:
   - Detector processes batches of frames
   - Tracker applies temporal smoothing
   - Results are evaluated against ground truth
   - Visualizations are generated if requested

### Entry Point

- `src/main.py`: Main entry point that initializes the runner based on configuration
- Uses Hydra for configuration management
- Selects appropriate runner from `src/runners/__init__.py`

## Key Implementation Details

- Uses PyTorch as the deep learning framework
- Implements heatmap-based ball detection
- Supports multi-frame input processing (step parameter)
- Generates visualization outputs when requested
- Calculates evaluation metrics (precision, recall, F1, accuracy, RMSE)