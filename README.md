# mental_health_risk_models
Data processing and Training Scripts for MCI and Depression Risk Models

## Paper
[To be added]
## Usage

### Setup
In order to run any of the scripts, you must create the files `.env` and `.slurm`.
See `env.example` and `slurm.example` for how to create each of these files respectively.

After creating `.env` and `.slurm`, your `src` directory should look like the following:
```bash
$ cd src
$ tree . -a
.
├── create_dataset.sh
├── data_pipeline.py
├── .env
├── __init__.py
├── preprocess.py
├── __pycache__
├── run_training.sh
├── .slurm
├── submit_job.sh
└── training.py
```

#### Install environment

Run the setup script to create the conda environment and install dependencies:
```bash
./setup_env.sh
```

This will:
1. Create a conda environment named `mental_health_models`
2. Activate the environment
3. Install all required packages from `requirements.txt`

**Manual installation (alternative):**
```bash
conda create -n mental_health_models python=3.13.2
conda activate mental_health_models
pip install -r requirements.txt
```

**Note:** Always activate the conda environment before running any scripts:
```bash
conda activate mental_health_models
```

### MCI Modeling

#### Quick Start

For convenience in reproduction, we provide `submit_job.sh` which runs both dataset creation and training sequentially. To run the complete pipeline for MCI with the default configuration:

```bash
./submit_job.sh
```

#### Running Individual Components

##### Dataset Creation

To run sample generation and labeling for MCI with the default configuration:

```bash
./create_dataset.sh
```

##### Training and Testing

To run training and testing for MCI using the dataset created by `create_dataset.sh` with the default configuration:

```bash
./run_training.sh
```

#### Script Options

All scripts support the `--help` flag to display available arguments and options:

**Dataset creation:**
```bash
./create_dataset.sh --help
```

**Training:**
```bash
./run_training.sh --help
```

**Complete pipeline:**
```bash
./submit_job.sh --help
```

### Depression Modeling
[To be added]

### General Configuration

#### Environment Variables (.env)
These variables must be configured in your `.env` file:
- `OUTPUT_DIR`: Path general output directory of dataset and model checkpoints
- `DATASET_DIR`: Path to dataset directory of dataset created from `create_dataset.sh`
- `MODEL_NAME`: Huggingface pretrained model name
- `MODEL_DIR`: Path to model output directory 
- `MCI_ICD_REGEX`: Regex for phenotyping MCI patients via icds
- `MCI_MED_REGEX`: Regex for phenotyping MCI patients via medications
- `MCI_QA_MEDKEY_PATH`: Path to map from medication id to medication name. Must be json.
- `PT_MESSAGES_PATH`: Path to patient message histories. Can be json or csv.
- `PT_ICDS_PATH`: Path to patient icd histories.
- `PT_DEMO_PATH`: Path to patient demographic information. Can be json or csv.
- `PT_MED_PATH`: Path to patient mediciation histories from general dataset. Can be json or csv.
- `PT_QA_MEDS_PATH`: Path to patient medication histories from qa dataset. Can be json or csv.

#### Slurm Configuration (.slurm)
All variables must be configured in your `.slurm` file, unless stated otherwise:
- `USER_EMAIL`: Email to send slurm job notifications to
- `CONDA_MODULE`: Path of conda module to load
- `CONDA_ENV`: Name/Path of conda environment
- `HF_HOME`: Path of huggingface cache (optional)
- `EXPERIMENT_NAME`: Name of experiment
- `LOGS`: Path to SLURM output logs and error logs 

#### Output Files
Generated outputs will be saved to:
- **Datasets**: `OUTPUT_DIR/DATASET_DIR/`
- **Model Outputs**: `OUTPUT_DIR/MODEL_DIR/`
- **Logs**: `LOGS`

#### Monitoring Jobs
Check job status:
```bash
watch -n 1 "squeue -u $USER"
```

View logs:
```bash
tail -f logs/job_output.log
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing `.env` file | Copy `env.example` to `.env` and fill in values |
| Missing `.slurm` file | Copy `slurm.example` to `.slurm` and fill in values |
| Permission denied | Run `chmod +x *.sh` to make scripts executable |
| Module not found | Ensure you've activated the conda environment and run `pip install -r requirements.txt` |
| Invalid option error | Run script with `--help` to see available options |
| Conda environment not found | Run `./setup_env.sh` to create the environment |
