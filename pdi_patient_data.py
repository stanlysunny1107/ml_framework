"""Real patient images, annotations, and measured spectra for the PDI viewer."""

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
from PyQt6.QtGui import QImage, QPixmap

from patient_dataset import load_cube


def load_patient_rasters(record):
    """Load aligned images at native resolution; do not synthesize predictions."""
    cube = load_cube(record["cube"])
    _, h, w = cube.shape
    labels = loadmat(record["gt"])[record.get("gt_key", "annotation_map")]
    if (
        labels.shape != (h, w)
        or not np.isfinite(labels).all()
        or not np.equal(labels, np.floor(labels)).all()
    ):
        raise ValueError("Annotation and cube dimensions differ.")
    image = QImage(record["rgb"] or "")
    if image.isNull() or image.width() != w or image.height() != h:
        raise ValueError(f'Missing or misaligned RGB image for {record["id"]}')
    reference = QImage(record["gt_png"] or "")
    if not reference.isNull() and (reference.width(), reference.height()) != (w, h):
        raise ValueError("Ground-truth PNG dimensions differ from cube.")
    if reference.isNull():
        rgba = label_colors(labels)
    else:
        reference = reference.convertToFormat(QImage.Format.Format_RGBA8888)
        ptr = reference.bits()
        ptr.setsize(reference.sizeInBytes())
        rgba = (
            np.frombuffer(ptr, np.uint8)
            .reshape(h, reference.bytesPerLine())[:, : w * 4]
            .reshape(h, w, 4)
            .copy()
        )
    rgba[labels == 0, 3] = 0
    predicted = np.zeros((h, w, 4), np.uint8)
    prediction = None
    prediction_path = Path(
        record.get("prediction_path")
        or (Path(record["cube"]).parent / "prediction.npy")
    )
    if prediction_path.exists():
        prediction = load_prediction(prediction_path, labels.shape)
        predicted = label_colors(prediction)
    elif record.get("prediction_path"):
        raise ValueError(f"Prediction file not found: {prediction_path}")
    text = Path(record["header"]).read_text()
    match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.I)
    wavelengths = (
        [float(v) for v in match.group(1).split(",")]
        if match
        else list(range(cube.shape[0]))
    )
    if len(wavelengths) != cube.shape[0]:
        raise ValueError("Wavelength count differs from band count.")

    def pixmap(array):
        return QPixmap.fromImage(
            QImage(array.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        )

    meta = {
        "width": w,
        "height": h,
        "real_patient": True,
        "cube": cube,
        "labels": labels,
        "prediction": prediction,
        "wavelengths": wavelengths,
        "spectral_axis_label": "WAVELENGTH (nm)" if match else "BAND INDEX",
        "metrics": classification_metrics(labels, prediction),
    }
    return QPixmap.fromImage(image), pixmap(rgba), pixmap(predicted), meta


def label_colors(labels):
    """Use reference display colors; retain numeric labels without clinical interpretation."""
    rgba = np.zeros((*labels.shape, 4), np.uint8)
    palette = {103: (0, 0, 255), 104: (0, 255, 0), 105: (255, 255, 255)}
    for value in np.unique(labels):
        if value != 0:
            color = palette.get(
                int(value),
                (int(value * 67) % 256, int(value * 131) % 256, int(value * 197) % 256),
            )
            rgba[labels == value] = (*color, 255)
    return rgba


def pixel_info(meta, px, py):
    """Return measured spectral values and numeric annotations at the selected pixel."""
    px, py = int(np.clip(px, 0, meta["width"] - 1)), int(
        np.clip(py, 0, meta["height"] - 1)
    )
    label = int(meta["labels"][py, px])
    prediction = meta["prediction"]
    spectrum = np.asarray(meta["cube"][:, py, px], float)
    return {
        "px": px,
        "py": py,
        "tissue": "Not classified",
        "behind_prediction": "Not classified",
        "gt": f"Label {label}" if label else "Unannotated",
        "pred": (
            f"Label {int(prediction[py,px])}"
            if prediction is not None
            else "No prediction loaded"
        ),
        "spectrum": list(zip(meta["wavelengths"], spectrum.tolist())),
    }


def load_prediction(path, shape):
    """Read original integer label IDs from a NumPy map or MATLAB prediction variable."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        prediction = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".mat":
        data = loadmat(path)
        if "prediction" not in data:
            raise ValueError(
                "MATLAB prediction file must contain a variable named prediction."
            )
        prediction = data["prediction"]
    else:
        raise ValueError("Select a .npy or .mat prediction label map.")
    if (
        prediction.shape != shape
        or not np.issubdtype(prediction.dtype, np.number)
        or not np.isfinite(prediction).all()
        or not np.equal(prediction, np.floor(prediction)).all()
    ):
        raise ValueError(
            "Prediction must contain finite integer class IDs with the same dimensions as the cube."
        )
    return prediction.astype(np.int64)


def classification_metrics(labels, prediction):
    """Compute multiclass metrics on annotated pixels only (ground truth != 0).

    Sensitivity, specificity, and F1 are macro averages over ground-truth classes.
    MCC uses the multiclass confusion matrix. AUC requires probability scores and
    is therefore unavailable for integer label maps. Undefined values remain N/A.
    """
    result = {
        key: "N/A"
        for key in ("Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC", "MCC")
    }
    if prediction is None:
        return result
    valid = labels != 0
    truth, pred = labels[valid].astype(np.int64), prediction[valid].astype(np.int64)
    if not truth.size:
        return result
    values = np.union1d(truth, pred)
    target = np.searchsorted(values, truth)
    output = np.searchsorted(values, pred)
    k = len(values)
    matrix = (
        np.bincount(k * target + output, minlength=k * k)
        .reshape(k, k)
        .astype(np.float64)
    )
    actual, assigned = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    total = matrix.sum()
    fp, fn = assigned - tp, actual - tp
    tn = total - tp - fp - fn
    present = actual > 0

    def ratio(numerator, denominator):
        values = np.divide(
            numerator, denominator, out=np.full(k, np.nan), where=denominator != 0
        )
        finite = values[present & np.isfinite(values)]
        return float(finite.mean()) if finite.size else None

    def fmt(value):
        return f"{value:.4f}" if value is not None and np.isfinite(value) else "N/A"

    denominator = np.sqrt(
        (total * total - np.dot(actual, actual))
        * (total * total - np.dot(assigned, assigned))
    )
    result.update(
        {
            "Accuracy": fmt(float(tp.sum() / total)),
            "Sensitivity": fmt(ratio(tp, actual)),
            "Specificity": fmt(ratio(tn, tn + fp)),
            "F1-Score": fmt(ratio(2 * tp, 2 * tp + fp + fn)),
            "MCC": fmt(
                float((tp.sum() * total - np.dot(actual, assigned)) / denominator)
                if denominator > 0
                else None
            ),
        }
    )
    return result
