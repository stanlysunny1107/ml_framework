import csv
from pathlib import Path
import numpy as np
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdi_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import LABELS, write_json


def confusion(truth, pred):
    valid = truth != 0
    t = np.searchsorted(LABELS, truth[valid])
    p = np.searchsorted(LABELS, pred[valid])
    return np.bincount(3 * t + p, minlength=9).reshape(3, 3)


def metrics(cm):
    tp = cm.diagonal()
    actual = cm.sum(1)
    predicted = cm.sum(0)

    def div(a, b):
        return [float(x / y) if y else None for x, y in zip(a, b)]

    iou = div(tp, actual + predicted - tp)
    return {
        "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None,
        "mean_iou": (
            float(np.mean([v for v in iou if v is not None]))
            if any(v is not None for v in iou)
            else None
        ),
        "per_class": {
            str(k): {
                "dice": div(2 * tp, actual + predicted)[i],
                "iou": iou[i],
                "precision": div(tp, predicted)[i],
                "recall": div(tp, actual)[i],
                "support": int(actual[i]),
            }
            for i, k in enumerate(LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_order": LABELS,
    }


def export_report(path, report):
    write_json(path.with_suffix(".json"), report)
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "patient_id",
                "class_id",
                "dice",
                "iou",
                "precision",
                "recall",
                "support",
                "pixel_accuracy",
                "mean_iou",
            ],
        )
        w.writeheader()
        for scope, pid, m in [
            ("patient", pid, m) for pid, m in report["patients"].items()
        ] + [("pooled_pixels", "ALL", report["pooled_pixels"])]:
            for k, v in m["per_class"].items():
                w.writerow(
                    {
                        "scope": scope,
                        "patient_id": pid,
                        "class_id": k,
                        **v,
                        "pixel_accuracy": m["pixel_accuracy"],
                        "mean_iou": m["mean_iou"],
                    }
                )
        w.writerow(
            {
                "scope": "patient_macro",
                "patient_id": "ALL",
                "mean_iou": report["patient_macro_mean_iou"],
                "pixel_accuracy": report["patient_macro_pixel_accuracy"],
            }
        )


def overlay(cube, labels, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Downsample preview only; prediction exports remain native resolution.
    rgb = np.asarray(
        cube[
            [cube.shape[0] * 3 // 4, cube.shape[0] // 2, cube.shape[0] // 4], ::4, ::4
        ],
        float,
    ).transpose(1, 2, 0)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    from matplotlib.colors import ListedColormap

    encoded = np.zeros(labels.shape, np.uint8)
    for i, k in enumerate(LABELS):
        encoded[labels == k] = i + 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(rgb)
    ax[0].imshow(
        np.ma.masked_equal(encoded[::4, ::4], 0),
        cmap=ListedColormap(["black", "orange", "cyan", "magenta"]),
        vmin=0,
        vmax=3,
        alpha=0.4,
    )
    im = ax[1].imshow(
        encoded,
        cmap=ListedColormap(["black", "orange", "cyan", "magenta"]),
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax[1], ticks=[0, 1, 2, 3]).ax.set_yticklabels(
        ["0 (ignored)", "103", "104", "105"]
    )
    fig.suptitle(
        "Research predictions / annotations — not validated clinical diagnoses"
    )
    for a in ax:
        a.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = [r["epoch"] for r in history]
    for k in ("train_loss", "validation_loss"):
        ax[0].plot(x, [r[k] for r in history], label=k)
    ax[1].plot(
        x, [r["validation_mean_iou"] for r in history], label="validation mean IoU"
    )
    for a in ax:
        a.legend()
        a.set_xlabel("Epoch")
    fig.savefig(path)
    plt.close(fig)
