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

### 1) Install system tools

Please make sure these commands are available in your environment:

- `blastn`
- `makeblastdb`
- `prodigal-gv`

### 2) Install Python dependencies

```bash
cd final_release
pip install -r requirements.txt
pip install -e .
```

After installation, you can run either:

- `python run.py ...`
- `gianthost ...`

## Database Layout

Your `--db` directory should contain:

- `ncldv_complete.fasta`
- `taxonomy.csv`
- `gvog.complete.hmm`
- `model/` (or your custom `--model-subdir`) with:
  - `inference_config.json`
  - recommended: `nested_kfold_bundle.pt`
  - fallback: `*_final.pt` + `conformal_scores.npz`
  - optional: `label_encoders.json`

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

## Output

Main output is a CSV file with prediction results per sample.
Core columns include:

- `sample_id`
- `virus_order`
- `host_group_1`
- `host_group_2`
- `host_group_2_lca`

## Publish to GitHub

```bash
cd final_release
git init
git add .
git commit -m "Initial GiantHost release"
```

Then add your remote repository and push.
