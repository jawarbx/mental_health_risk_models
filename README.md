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
$ tree . -a
.
├── create_dataset.sh
├── LICENSE
├── README.md
├── requirements.txt
├── run_full_pipeline.sh
├── run_training.sh
├── setup_env.sh
├── .slurm
└── src
    ├── data_pipeline.py
    ├── .env
    ├── __init__.py
    ├── preprocess.py
    └── training.py
```
#### Install environment

Go to the project root and run the `setup_env.sh` to create the conda environment and install dependencies:

```bash
$ ./setup_env.sh
```

This will:
1. Create a conda environment named `mental_health_models`
2. Activate the environment
3. Install all required packages from `requirements.txt`

**Manual installation (alternative):**
```bash
$ conda create -n mental_health_models python=3.13.2
$ conda activate mental_health_models
$ pip install -r requirements.txt
```

**Note:** Always activate the conda environment before running any scripts:
```bash
$ conda activate mental_health_models
```

### MCI Modeling

#### Quick Start

For convenience in reproduction, we provide `run_full_pipeline.sh` which runs both dataset creation and training sequentially. To run the complete pipeline for MCI with the default configuration:

```bash
$ source .slurm
$ ./run_full_pipeline.sh
```
To run a batch job on slurm, run the following command:
```bash
source .slurm
sbatch run_full_pipeline.sh
```

#### Running Individual Components

##### Dataset Creation

To run sample generation and labeling for MCI with the default configuration:

```bash
$ source .slurm
$ ./create_dataset.sh
```

To run a batch job on slurm, run the following command:
```bash
$ source .slurm
$ sbatch --parsable \
	--output=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.out \
	--error=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.err \
	--job-name=${EXPERIMENT_NAME}_dataset \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=${USER_EMAIL} \
	create_dataset.sh
```

##### Training and Testing

To run training and testing for MCI using the dataset created by `create_dataset.sh` with the default configuration:

```bash
$ source .slurm
$ ./run_training.sh
```

To run a batch job on slurm, run the following command:

```bash
$ source .slurm
$ sbatch --parsable \
	--output=${LOGS}/${EXPERIMENT_NAME}_training_%j.out \
	--error=${LOGS}/${EXPERIMENT_NAME}_training_%j.err \
	--job-name=${EXPERIMENT_NAME}_training \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=${USER_EMAIL} \
	run_training.sh
```

**Note** `create_dataset.sh` must have been run at least **once** for `run_training.sh` to work!

#### Script Options

All scripts support the `--help` flag to display available arguments and options:

**Dataset creation:**
```bash
$ ./create_dataset.sh --help
```

**Training:**
```bash
$ ./run_training.sh --help
```

**Complete pipeline:**
```bash
$ ./run_full_pipeline.sh --help
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
$ watch -n 1 "squeue -u $USER"
```

View logs:
```bash
$ tail -f logs/job_output.log
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
