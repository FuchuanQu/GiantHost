#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from gianthost.pipeline import ReleasePredictor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GiantHost inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="contig mode input fasta; mag mode input folder")
    parser.add_argument("--input-type", choices=["contig", "mag"], default="contig", help="input mode")
    parser.add_argument("--db", required=True, help="database folder containing gvog.complete.hmm/taxonomy.csv/ncldv_complete.fasta/model")
    parser.add_argument("--model-subdir", default="model", help="model subdirectory under db")
    parser.add_argument("--model-file", default=None, help="optional model checkpoint path overriding auto search")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--midfolder", required=True, help="temporary working directory")
    parser.add_argument("--min-length", type=int, default=5000, help="minimum contig length for contig mode")
    parser.add_argument("--strictness", choices=["normal", "strict", "very_strict"], default="normal", help="conformal strictness")
    parser.add_argument("--alpha", type=float, default=None, help="optional alpha override")
    parser.add_argument("--threads", type=int, default=8, help="thread number")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="inference device")
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


def _setup_logger(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = _parse_args()
    _setup_logger(args.debug)

    predictor = ReleasePredictor(args)
    result = predictor.run()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    front = [c for c in ["sample_id", "virus_order", "host_group_1", "host_group_2"] if c in result.columns]
    rest = [c for c in result.columns if c not in front]
    result = result[front + rest]

    result.to_csv(out_path, index=False)
    logging.info("Done. %d samples written to %s", len(result), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
