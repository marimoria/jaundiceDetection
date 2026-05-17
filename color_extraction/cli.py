"""
cli.py
Command-line entry point for the color_extraction package.

Two modes
---------
  training   Build the full training-ready CSV from a folder of images
             and the NeoJaundice clinical metadata CSV.

  debug      Process a single image with full diagnostic output.

Usage examples
--------------
  python -m color_extraction training \\
      --image_dir    data/neo/images \\
      --clinical_csv data/neo/neo.csv \\
      --output       out/training_data.csv

  python -m color_extraction training \\
      --image_dir    data/neo/images \\
      --clinical_csv data/neo/neo.csv \\
      --output       out/training_data.csv \\
      --start_from   0032-3.jpg

  python -m color_extraction debug \\
      --image      data/neo/images/0003-1.jpg \\
      --debug_dir  out/debug

  python -m color_extraction training \\
      --image_dir    data/neo/images \\
      --clinical_csv data/neo/neo.csv \\
      --output       out/training_data.csv \\
      --debug \\
      --debug_dir    out/debug \\
      --save_every   100
"""

import argparse
import logging
import sys
from datetime import datetime


def _setup_root_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("jaundice_extractor")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def run_training_pipeline(args: argparse.Namespace, log: logging.Logger) -> None:
    from .dataset_pipeline import run_full_training_data_pipeline

    run_full_training_data_pipeline(
        image_dir=args.image_dir,
        clinical_csv=args.clinical_csv,
        output_csv=args.output,
        n_augmented_variants=args.n_augmented_variants,
        checkpoint_file=args.output.replace(".csv", "_checkpoint.csv"),
        save_every=args.save_every,
        debug=args.debug,
        debug_dir=args.debug_dir,
        start_from=getattr(args, "start_from", None),
    )


def run_single_image_debug(args: argparse.Namespace, log: logging.Logger) -> None:
    import os
    from pathlib import Path

    import cv2

    from .augmentation import generate_brightness_augmented_variants
    from .image_processor import process_single_image

    log.info("mode=debug  image=%s", args.image)

    row = process_single_image(args.image, debug=True, debug_dir=args.debug_dir)
    for k, v in row.items():
        if k not in ("patient_id", "image_idx", "is_augmented"):
            log.info("  %-16s %s", k, v)

    if not getattr(args, "augment", False):
        return

    raw_bgr  = cv2.imread(args.image)
    variants = generate_brightness_augmented_variants(raw_bgr, n_variants=3)  # type: ignore
    stem     = Path(args.image).stem
    ext      = Path(args.image).suffix
    os.makedirs(args.debug_dir, exist_ok=True)

    for i, aug_bgr in enumerate(variants):
        temp_path = os.path.join(args.debug_dir, f"{stem}_aug{i}{ext}")
        cv2.imwrite(temp_path, aug_bgr)
        aug_row = process_single_image(temp_path, debug=True, debug_dir=args.debug_dir)
        log.info("augment %d:", i)
        for k, v in aug_row.items():
            if k not in ("patient_id", "image_idx", "is_augmented"):
                log.info("  %-16s %s", k, v)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color_extraction",
        description="Neonatal jaundice color feature extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("training", help="Build the full training-ready CSV.")
    train_p.add_argument("--image_dir",            required=True, metavar="DIR")
    train_p.add_argument("--clinical_csv",         required=True, metavar="CSV")
    train_p.add_argument("--output",               required=True, metavar="CSV")
    train_p.add_argument("--n_augmented_variants", type=int, default=3,   metavar="N")
    train_p.add_argument("--save_every",           type=int, default=50,  metavar="N")
    train_p.add_argument("--debug",                action="store_true")
    train_p.add_argument("--debug_dir",            default="debug",       metavar="DIR")
    train_p.add_argument("--log_file",             metavar="PATH")
    train_p.add_argument("--start_from",           metavar="FILENAME")

    debug_p = sub.add_parser("debug", help="Process a single image with diagnostic output.")
    debug_p.add_argument("--image",     required=True, metavar="PATH")
    debug_p.add_argument("--debug_dir", default="debug", metavar="DIR")
    debug_p.add_argument("--log_file",  metavar="PATH")
    debug_p.add_argument("--augment",   action="store_true",
                         help="Also generate and debug brightness-augmented variants.")

    return parser


def main() -> None:
    parser = build_argument_parser()
    args   = parser.parse_args()
    log    = _setup_root_logger(getattr(args, "log_file", None))

    log.info("started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if args.mode == "training":
        run_training_pipeline(args, log)
    elif args.mode == "debug":
        run_single_image_debug(args, log)
    else:
        parser.print_help()
        sys.exit(1)

    log.info("done.")


if __name__ == "__main__":
    main()