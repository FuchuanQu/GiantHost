# GiantHost

GiantHost is a lightweight release for NCLDV host prediction inference.
This folder is ready to publish as a standalone GitHub repository.

## What This Release Contains

- `run.py`: simple local entrypoint
- `gianthost/`: GiantHost pipeline package
- `src/`: minimal inference-only modules (already slimmed)
- `scripts/parallel-prodigal-gv.py`: helper script used during feature extraction
- `pyproject.toml` and `requirements.txt`: install metadata

## Quick Start

### Fastest installation path (recommended first)

If you use `mamba` or `conda`, this is the fastest and preferred way to install GiantHost (if you use `conda`, just replace `mamba` with `conda`):

```bash
git clone https://github.com/FuchuanQu/GiantHost.git
cd GiantHost
mamba env create -f environment.yml
mamba activate gianthost
pip install -e .
```

This path installs Python dependencies plus `blast` and `prodigal-gv` in one go.

---

### Manual installation
If you prefer manual step-by-step installation, follow the sections below.

#### 1) Create a new conda environment (mamba recommended)

Using `mamba` (recommended):

```bash
mamba create -n gianthost python=3.10 -y
mamba activate gianthost
```

Using `conda`:

```bash
conda create -n gianthost python=3.10 -y
conda activate gianthost
```

#### 2) Install BLAST and prodigal-gv in the environment

Using `mamba` (recommended):

```bash
mamba install -c bioconda blast prodigal-gv -y
```

Using `conda` with your commands:

```bash
conda install -c bioconda prodigal-gv blast -y
```

#### 3) Install GiantHost

```bash
cd final_release
pip install -r requirements.txt
pip install -e .
```

<!-- Optional (one command with `environment.yml`, includes BLAST and prodigal-gv):

```bash
mamba env create -f environment.yml
mamba activate gianthost
``` -->

After installation, you can run either:

- `python run.py ...`
- `gianthost ...`

### Download database

You can download all required database files manually:

- Main database package: `https://github.com/FuchuanQu/GiantHost/releases/download/0.1.0/gianthost_db.zip`

Or use the helper script:

```bash
bash scripts/download_database.sh /path/to/db
```

## Run Inference

### Contig mode

```bash
gianthost \
  --input /path/to/contigs.fasta \
  --input-type contig \
  --db /path/to/db \
  --midfolder /path/to/mid \
  --output /path/to/predictions.csv \
  --min-length 5000 \
  --strictness normal
```

### MAG mode

```bash
gianthost \
  --input /path/to/mag_folder \
  --input-type mag \
  --db /path/to/db \
  --midfolder /path/to/mid \
  --output /path/to/predictions.csv \
  --strictness strict
```

### Command-line Arguments

GiantHost supports the following command-line arguments.

### Required arguments

- `--input`
  - Required: Yes
  - Description: Input path. For `contig` mode, this should be a FASTA file. For `mag` mode, this should be a folder containing FASTA files.
  - Example: `--input /path/to/contigs.fasta`

- `--db`
  - Required: Yes
  - Description: Database folder containing `gvog.complete.hmm`, `taxonomy.csv`, `ncldv_complete.fasta`, and a model subfolder.
  - Example: `--db /path/to/db`

- `--output`
  - Required: Yes
  - Description: Output CSV file path for prediction results.
  - Example: `--output /path/to/predictions.csv`

- `--midfolder`
  - Required: Yes
  - Description: Working directory for temporary/intermediate files (feature files, BLAST outputs, etc.).
  - Example: `--midfolder /path/to/mid`

### Optional arguments

- `--input-type`
  - Required: No
  - Default: `contig`
  - Choices: `contig`, `mag`
  - Description: Selects whether input is single-contig FASTA or a MAG folder.

- `--model-subdir`
  - Required: No
  - Default: `model`
  - Description: Model subdirectory name under `--db`.
  - Example: `--model-subdir model_256`

- `--model-file`
  - Required: No
  - Default: None
  - Description: Explicit checkpoint path. If set, GiantHost uses this model file instead of automatic search.
  - Example: `--model-file /path/to/model_final.pt`

- `--min-length`
  - Required: No
  - Default: `5000`
  - Description: Minimum sequence length cutoff in `contig` mode. Shorter contigs are filtered out.

- `--strictness`
  - Required: No
  - Default: `normal`
  - Choices: `normal`, `strict`, `very_strict`
  - Description: Controls conformal filtering strictness for host predictions.

- `--alpha`
  - Required: No
  - Default: None
  - Description: Overrides alpha in model inference config.
  - Example: `--alpha 0.1`

- `--threads`
  - Required: No
  - Default: `8`
  - Description: Number of CPU threads used by external tools and feature extraction.

- `--device`
  - Required: No
  - Default: `cpu`
  - Choices: `cpu`, `cuda`
  - Description: Inference device.

- `--debug`
  - Required: No
  - Default: Disabled
  - Description: Enables verbose debug logging.


## Output

Main output is a compact CSV file with one row per sample.
Columns are:

- `sample_id`
  - Input sequence/MAG identifier.

- `predict_final`
  - Final host prediction label.
  - Derived from model output `host_group_2`.

- `predict_set`
  - Conformal prediction set in JSON list format.
  - Examples: `[]`, `["Mammalia"]`, `["Aves", "Mammalia"]`.

- `predict_lca`
  - Lowest common ancestor summary of the prediction set.
  - Derived from `host_group_2_lca`.

<!-- ## Publish to GitHub

```bash
cd final_release
git init
git add .
git commit -m "Initial GiantHost release"
```

Then add your remote repository and push. -->
