"""
python evaluation_binary_improved.py \
  --gt /data2/letrunglinh/despecular/diff-train-style/SHDR-E2E-SHDNet/trainsets/SHIQ/Test/Mask/ \
  --pr /data2/letrunglinh/despecular/diff-train-style/SHR-specmamba-anyir/results/Unified_R2MoE_SD2_v4_rdim64/mask

"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as a grayscale float array."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a normalized mask to 0/1."""
    return (arr >= threshold).astype(np.uint8)


@dataclass
class Counts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = pred.astype(np.uint8).reshape(-1)
        gt = gt.astype(np.uint8).reshape(-1)

        self.tp += int(np.sum((pred == 1) & (gt == 1)))
        self.tn += int(np.sum((pred == 0) & (gt == 0)))
        self.fp += int(np.sum((pred == 1) & (gt == 0)))
        self.fn += int(np.sum((pred == 0) & (gt == 1)))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def metrics_from_counts(c: Counts) -> Dict[str, float]:
    total = c.tp + c.tn + c.fp + c.fn
    precision_den = c.tp + c.fp
    recall_den = c.tp + c.fn
    specificity_den = c.tn + c.fp
    iou_den = c.tp + c.fp + c.fn
    dice_den = 2 * c.tp + c.fp + c.fn

    dice = 1.0 if dice_den == 0 else (2.0 * c.tp) / dice_den
    iou = 1.0 if iou_den == 0 else c.tp / iou_den
    precision = 1.0 if precision_den == 0 and recall_den == 0 else safe_div(c.tp, precision_den)
    recall = 1.0 if recall_den == 0 else safe_div(c.tp, recall_den)
    specificity = 1.0 if specificity_den == 0 else safe_div(c.tn, specificity_den)
    accuracy = 1.0 if total == 0 else safe_div(c.tp + c.tn, total)
    ber = 1.0 - 0.5 * (recall + specificity)
    mae = 0.0 if total == 0 else safe_div(c.fp + c.fn, total)
    f1 = dice

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "ber": float(ber),
        "mae": float(mae),
        "f1": float(f1),
    }


def pair_files(gt_dir: Path, pr_dir: Path, suffix: str = "") -> List[Tuple[Path, Path]]:
    pairs = []
    for gt_path in sorted(gt_dir.iterdir()):
        if gt_path.name.startswith(".") or not gt_path.is_file():
            continue

        stem = gt_path.stem
        candidates = sorted(pr_dir.glob(f"{stem}{suffix}.*"))
        if len(candidates) != 1:
            raise AssertionError(f"Expected exactly one prediction for {stem}, got: {candidates}")

        pairs.append((gt_path, candidates[0]))

    if not pairs:
        raise RuntimeError(f"No files found in {gt_dir}")

    return pairs


def per_image_row(gt_path: Path, pr_path: Path, counts: Counts) -> Dict[str, float | str]:
    metrics = metrics_from_counts(counts)
    return {
        "image": gt_path.name,
        "stem": gt_path.stem,
        "gt_path": str(gt_path),
        "pr_path": str(pr_path),
        **metrics,
        "tp": float(counts.tp),
        "tn": float(counts.tn),
        "fp": float(counts.fp),
        "fn": float(counts.fn),
    }


def evaluate_threshold(
    gt_dir: Path,
    pr_dir: Path,
    threshold: float,
    suffix: str = "",
) -> Tuple[Dict[str, float], List[Dict[str, float | str]]]:
    pairs = pair_files(gt_dir, pr_dir, suffix=suffix)

    global_counts = Counts()
    per_image_metrics: List[Dict[str, float]] = []
    per_image_rows: List[Dict[str, float | str]] = []

    for gt_path, pr_path in tqdm(pairs, desc=f"Threshold {threshold}", unit="img", leave=False):
        gt = load_grayscale(gt_path)
        pr = load_grayscale(pr_path)

        if gt.shape != pr.shape:
            raise AssertionError(
                f"Shape mismatch for {gt_path.name}: gt={gt.shape}, pr={pr.shape}"
            )

        gt_bin = binarize(gt, 0.5)
        pr_bin = binarize(pr, threshold / 255.0 if threshold > 1 else threshold)

        counts = Counts()
        counts.update(pr_bin, gt_bin)
        global_counts.update(pr_bin, gt_bin)
        metrics = metrics_from_counts(counts)
        per_image_metrics.append(metrics)
        per_image_rows.append(per_image_row(gt_path, pr_path, counts))

    global_metrics = metrics_from_counts(global_counts)
    mean_metrics = {
        f"mean_{k}": float(np.mean([m[k] for m in per_image_metrics]))
        for k in per_image_metrics[0].keys()
    }

    result = {
        "threshold": float(threshold),
        "num_images": float(len(pairs)),
        **global_metrics,
        **mean_metrics,
        "tp": float(global_counts.tp),
        "tn": float(global_counts.tn),
        "fp": float(global_counts.fp),
        "fn": float(global_counts.fn),
    }
    return result, per_image_rows


def write_summary_csv(csv_path: Path, result: Dict[str, float]) -> None:
    fieldnames = [
        "threshold",
        "num_images",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "ber",
        "mae",
        "f1",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_specificity",
        "mean_accuracy",
        "mean_ber",
        "mean_mae",
        "mean_f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result)


def write_per_image_csv(csv_path: Path, rows: List[Dict[str, float | str]]) -> None:
    fieldnames = [
        "image",
        "stem",
        "gt_path",
        "pr_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "ber",
        "mae",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary segmentation masks with standard metrics."
    )
    parser.add_argument("--gt", required=True, type=str, help="Ground-truth mask directory")
    parser.add_argument("--pr", required=True, type=str, help="Prediction mask directory")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional prediction filename suffix before the extension",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold on a normalized 0..1 scale, or 0..255 if > 1",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="evaluation_results_improved.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--per-image-csv",
        type=str,
        default="",
        help="Optional CSV path for per-image metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt)
    pr_dir = Path(args.pr)

    threshold = args.threshold
    result, per_image_rows = evaluate_threshold(gt_dir, pr_dir, threshold, suffix=args.suffix)

    print(f"Threshold: {threshold}")
    print(f"  Dice        : {result['dice']:.6f}")
    print(f"  IoU         : {result['iou']:.6f}")
    print(f"  Precision   : {result['precision']:.6f}")
    print(f"  Recall      : {result['recall']:.6f}")
    print(f"  Accuracy    : {result['accuracy']:.6f}")
    print(f"  Specificity : {result['specificity']:.6f}")
    print(f"  BER         : {result['ber']:.6f}")
    print(f"  MAE         : {result['mae']:.6f}")
    print(f"  TP/TN/FP/FN  : {int(result['tp'])}/{int(result['tn'])}/{int(result['fp'])}/{int(result['fn'])}")
    
    print("mean per image metric ")
    print("mean Dice        : {:.6f}".format(result['mean_dice']))
    print("mean IoU         : {:.6f}".format(result['mean_iou']))
    print("mean Precision   : {:.6f}".format(result['mean_precision']))
    print("mean Recall      : {:.6f}".format(result['mean_recall']))
    print("mean Specificity : {:.6f}".format(result['mean_specificity']))
    print("mean Accuracy    : {:.6f}".format(result['mean_accuracy']))
    print("mean BER         : {:.6f}".format(result['mean_ber']))
    print("mean MAE         : {:.6f}".format(result['mean_mae']))

    csv_path = Path(args.csv)
    write_summary_csv(csv_path, result)

    if args.per_image_csv:
        per_image_csv_path = Path(args.per_image_csv)
    else:
        per_image_csv_path = csv_path.with_name(f"{csv_path.stem}_per_image{csv_path.suffix}")
    write_per_image_csv(per_image_csv_path, per_image_rows)
    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved per-image CSV to: {per_image_csv_path}")


if __name__ == "__main__":
    main()

# python evaluation_binary_improved.py --gt ./trainsets/Test/Mask --pr ./outputInp --threshold 0.5
