"""BE-Net training and inference for ENVI hyperspectral cubes and MATLAB annotations.

Default dataset: Patient_47, 158 bands, 801 x 801 pixels.
Annotation variable: annotation_map; label 0 is ignored by default.
Original labels 103, 104, and 105 are restored when predictions are exported.

Examples:
    python BE-Net.py inspect
    python BE-Net.py train --epochs 20
    python BE-Net.py predict --checkpoint be_net_results/best_model.pt"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.io import loadmat, savemat


class SpectralGradientConv(nn.Module):
    """Apply a learned 5 x 5 Laplacian-weighted spatial filter and entropy-based channel attention.

    Input spectral bands are treated as channels. This operation computes spatial
    gradients; it does not explicitly differentiate along the wavelength axis."""

    def __init__(self, inch, outch, groups, ks=5, stride=1, first_layer="False"):
        super(SpectralGradientConv, self).__init__()
        self.first = first_layer
        self.padding = int(ks // 2)
        self.stride = (stride, stride)
        self.inch, self.outch, self.groups = (int(inch), int(outch), int(groups))
        self.weight = nn.Parameter(
            torch.Tensor(self.outch, self.inch // groups, ks, ks)
        )
        self.conv3x3 = nn.Conv2d(self.outch, self.outch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.outch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        epsilon = 1e-10
        # Repeat the spatial stencil for each input channel of the learned filters.
        list_x = []
        for i in range(self.weight.shape[1]):
            list_x.append(
                torch.tensor(
                    [
                        [-2, -4, -4, -4, -2],
                        [-4, 0, 8, 0, -4],
                        [-4, 8, 24, 8, -4],
                        [-4, 0, 8, 0, -4],
                        [-2, -4, -4, -4, -2],
                    ],
                    device=x.device,
                )
            )
        list_x = torch.stack(list_x, 0)
        weight_x = torch.mul(self.weight, list_x)
        input_x = F.conv2d(
            x, weight_x, stride=self.stride, padding=self.padding, groups=self.groups
        )
        conv = self.relu(self.bn(self.conv3x3(input_x)))
        gmax_group1, gavg_group1 = (self.gmaxpool(conv), self.gavgpool(conv))
        cat = torch.cat([gmax_group1.squeeze(-1), gavg_group1.squeeze(-1)], dim=2)
        B, C, N = cat.shape
        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy
        weights = 1 - normalized_entropy
        out = weights.unsqueeze(-1).unsqueeze(-1) * conv + conv
        return out


class ChannelBoundaryAttention(nn.Module):
    """Fuse encoder and decoder features using grouped multiscale convolutions
    and entropy-based channel attention."""

    def __init__(self, inchannel):
        super(ChannelBoundaryAttention, self).__init__()
        self.groupchannel = inchannel // 4
        self.groupconv1 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=1,
            padding=0,
        )
        self.groupconv3 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            padding=1,
        )
        self.groupconv5 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            dilation=2,
            padding=2,
        )
        self.groupconv7 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            dilation=3,
            padding=3,
        )
        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, skip, decoderx):
        skip_decoder_cat = torch.cat([skip, decoderx], dim=1)
        group1 = self.groupconv5(skip_decoder_cat[:, : self.groupchannel, :, :])
        group3 = self.groupconv7(skip_decoder_cat[:, self.groupchannel * 3 :, :, :])
        group5 = self.groupconv1(
            skip_decoder_cat[
                :, int(self.groupchannel * 2) : int(self.groupchannel * 3), :, :
            ]
        )
        group7 = self.groupconv3(
            skip_decoder_cat[
                :, int(self.groupchannel) : int(self.groupchannel * 2), :, :
            ]
        )
        groupconv = group1 + group3 + group5 + group7
        groupconv_ = self.relu(groupconv)
        gmax_group1, gavg_group1 = (
            self.gmaxpool(groupconv_),
            self.gavgpool(groupconv_),
        )
        cat = torch.cat([gmax_group1.squeeze(-1), gavg_group1.squeeze(-1)], dim=2)
        B, C, N = cat.shape
        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + self.epsilon)
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy
        weights = 1 - normalized_entropy
        out = weights.unsqueeze(-1).unsqueeze(-1) * groupconv_ + groupconv_
        return out


class PyramidDecoder(nn.Module):
    """Refine decoder features using dilated convolutions, channel entropy,
    and multiscale spatial pooling."""

    def __init__(self, in_channel, out_channel):
        super(PyramidDecoder, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                dilation=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                dilation=3,
                padding=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.maxpool5x5 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        self.maxpool7x7 = nn.MaxPool2d(kernel_size=7, padding=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x1)
        x3 = self.conv7x7(x2)
        out = x1 + x2 + x3
        B, C, H, W = out.shape
        probs = torch.softmax(out, dim=1)
        log_probs = torch.log(probs + self.epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)
        max_entropy = math.log(C)
        normalized_entropy = entropy / max_entropy
        area_ = self.sigmoid(out)
        a3x3 = self.maxpool3x3(area_)
        a5x5 = self.maxpool5x5(area_)
        a7x7 = self.maxpool7x7(area_)
        area = (a3x3 + a5x5 + a7x7) / 3 * area_
        output = out * (normalized_entropy.unsqueeze(1) + area) + out
        return output


class ConvBlock(nn.Module):
    """Apply two convolution, batch-normalization, and ReLU stages."""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """Upsample features by two and reduce the channel dimension."""

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class BENet(nn.Module):
    """BE-Net segmentation architecture adapted from the supplied implementation.

    Accept tensors with shape (batch, spectral_bands, height, width) and return
    unscaled class logits with shape (batch, classes, height, width).
    Use cross-entropy loss for training and argmax for class prediction."""

    def __init__(self, in_ch=3, out_ch=1):
        super(BENet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(in_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])
        self.Up5 = UpConv(filters[4], filters[3])
        self.Up_conv5 = PyramidDecoder(filters[4], filters[3])
        self.Up4 = UpConv(filters[3], filters[2])
        self.Up_conv4 = PyramidDecoder(filters[3], filters[2])
        self.Up3 = UpConv(filters[2], filters[1])
        self.Up_conv3 = PyramidDecoder(filters[2], filters[1])
        self.Up2 = UpConv(filters[1], filters[0])
        self.Up_conv2 = PyramidDecoder(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.spectralgradconv1 = SpectralGradientConv(
            in_ch, filters[0], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv2 = SpectralGradientConv(
            filters[0], filters[1], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv3 = SpectralGradientConv(
            filters[1], filters[2], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv4 = SpectralGradientConv(
            filters[2], filters[3], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv5 = SpectralGradientConv(
            filters[3], filters[4], 1, ks=5, stride=1, first_layer="False"
        )
        self.cba5 = ChannelBoundaryAttention(filters[4])
        self.cba4 = ChannelBoundaryAttention(filters[3])
        self.cba3 = ChannelBoundaryAttention(filters[2])
        self.cba2 = ChannelBoundaryAttention(filters[1])

    def forward(self, x):
        e1 = self.Conv1(x)
        se1 = self.spectralgradconv1(x)
        e1 = se1 + e1
        e2 = self.Maxpool1(e1)
        se2 = self.spectralgradconv2(e2)
        e2 = self.Conv2(e2)
        e2 = se2 + e2
        e3 = self.Maxpool2(e2)
        se3 = self.spectralgradconv3(e3)
        e3 = self.Conv3(e3)
        e3 = se3 + e3
        e4 = self.Maxpool3(e3)
        se4 = self.spectralgradconv4(e4)
        e4 = self.Conv4(e4)
        e4 = se4 + e4
        e5 = self.Maxpool4(e4)
        se5 = self.spectralgradconv5(e5)
        e5 = self.Conv5(e5)
        e5 = se5 + e5
        d5 = F.interpolate(self.Up5(e5), size=e4.shape[-2:], mode="nearest")
        d5 = self.cba5(e4, d5)
        d5 = self.Up_conv5(d5)
        d4 = F.interpolate(self.Up4(d5), size=e3.shape[-2:], mode="nearest")
        d4 = self.cba4(e3, d4)
        d4 = self.Up_conv4(d4)
        d3 = F.interpolate(self.Up3(d4), size=e2.shape[-2:], mode="nearest")
        d3 = self.cba3(e2, d3)
        d3 = self.Up_conv3(d3)
        d2 = F.interpolate(self.Up2(d3), size=e1.shape[-2:], mode="nearest")
        d2 = self.cba2(e1, d2)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)
        return out


def load_cube(path):
    """Read an ENVI cube as a memory-mapped array in band, row, column order.

    Read dimensions, data type, byte order, and interleave from the adjacent .hdr file.
    """
    path = Path(path)
    text = path.with_suffix(".hdr").read_text()
    meta = dict(
        (
            (k.strip().lower(), v.strip())
            for k, v in re.findall("^([^=\\n]+)=\\s*([^\\n]+)", text, re.M)
        )
    )
    h, w, b = (int(meta[k]) for k in ("lines", "samples", "bands"))
    types = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    dtype = np.dtype(
        ("<" if int(meta.get("byte order", 0)) == 0 else ">")
        + types[int(meta["data type"])]
    )
    offset = int(meta.get("header offset", 0))
    if path.stat().st_size != offset + h * w * b * dtype.itemsize:
        raise ValueError("Cube file size disagrees with ENVI header.")
    raw = np.memmap(path, dtype=dtype, mode="r", offset=offset)
    layout = meta["interleave"].lower()
    if layout == "bsq":
        return raw.reshape(b, h, w)
    if layout == "bil":
        return raw.reshape(h, b, w).transpose(1, 0, 2)
    if layout == "bip":
        return raw.reshape(h, w, b).transpose(2, 0, 1)
    raise ValueError(f"Unsupported interleave: {layout}")


def load_labels(path, shape, key=None):
    """Load an integer MATLAB annotation map aligned with the cube dimensions.

    Return the original annotation IDs and the selected MATLAB variable name."""
    data = loadmat(path)
    candidates = {
        k: v
        for k, v in data.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and (v.shape == shape)
    }
    if key is None:
        if len(candidates) != 1:
            raise ValueError(f"Select --gt-key; matching arrays: {list(candidates)}")
        key = next(iter(candidates))
    labels = np.asarray(data[key])
    if (
        labels.shape != shape
        or not np.isfinite(labels).all()
        or (not np.equal(labels, np.floor(labels)).all())
    ):
        raise ValueError(
            "Ground truth must be a finite integer H x W label map aligned with cube."
        )
    return (labels.astype(np.int64), key)


def tiles(h, start, stop, size):
    """Generate non-overlapping tile origins within the specified column interval."""
    return [(y, x) for y in range(0, h, size) for x in range(start, stop, size)]


class CubeTiles(torch.utils.data.Dataset):
    """Provide normalized, padded image tiles and aligned integer annotation tiles.

    Targets outside the image or spatial split use the loss ignore index -100."""

    def __init__(self, cube, labels, coords, size, mean, std, stop):
        self.cube, self.labels, self.coords = (cube, labels, coords)
        self.size, self.mean, self.std, self.stop = (size, mean, std, stop)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        y, x = self.coords[i]
        h = min(self.size, self.cube.shape[1] - y)
        w = min(self.size, self.stop - x)
        image = np.zeros((self.cube.shape[0], self.size, self.size), np.float32)
        image[:, :h, :w] = np.nan_to_num(
            (
                np.asarray(self.cube[:, y : y + h, x : x + w], dtype=np.float32)
                - self.mean
            )
            / self.std,
            nan=0,
            posinf=0,
            neginf=0,
        )
        target = np.full((self.size, self.size), -100, np.int64)
        if self.labels is not None:
            target[:h, :w] = self.labels[y : y + h, x : x + w]
        return (torch.from_numpy(image), torch.from_numpy(target))


def evaluate(model, loader, device, classes):
    """Compute pixel accuracy, per-class IoU, and a confusion matrix.

    Exclude ignored annotation pixels. Confusion matrix rows are ground-truth
    classes and columns are predicted classes, in checkpoint class order."""
    model.eval()
    confusion = np.zeros((classes, classes), np.int64)
    with torch.no_grad():
        for image, target in loader:
            pred = model(image.to(device)).argmax(1).cpu().numpy()
            truth = target.numpy()
            valid = truth != -100
            confusion += np.bincount(
                classes * truth[valid] + pred[valid], minlength=classes**2
            ).reshape(classes, classes)
    tp = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(classes, np.nan), where=union != 0)
    return {
        "accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "mean_iou": float(np.nanmean(iou)),
        "iou": [None if np.isnan(v) else float(v) for v in iou],
        "confusion_matrix": confusion.tolist(),
    }


def main():
    """Run dataset inspection, supervised training, or checkpoint-based prediction."""
    parser = argparse.ArgumentParser(
        description="BE-Net hyperspectral pixel segmentation"
    )
    parser.add_argument("mode", choices=["inspect", "train", "predict"])
    parser.add_argument("--cube", default="D:\\PDI\\data\\Patient_47\\cube.dat")
    parser.add_argument("--gt", default="D:\\PDI\\data\\Patient_47\\Patient_47_GT.mat")
    parser.add_argument("--gt-key", default="annotation_map")
    parser.add_argument(
        "--ignore-label", type=int, default=0, help="Unannotated label; default 0"
    )
    parser.add_argument("--output", default="be_net_results")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke testing",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (
        args.patch_size < 32
        or args.patch_size % 16
        or args.epochs < 1
        or (args.batch_size < 1)
    ):
        parser.error(
            "Patch size must be a multiple of 16 >=32; epochs and batch size must be positive."
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cube = load_cube(args.cube)
    bands, h, w = cube.shape
    device = torch.device(args.device)
    output = Path(args.output)
    if args.mode == "inspect":
        labels, key = load_labels(args.gt, (h, w), args.gt_key)
        values, counts = np.unique(labels, return_counts=True)
        print(
            json.dumps(
                {
                    "cube_CHW": list(cube.shape),
                    "gt_key": key,
                    "label_counts": dict(zip(map(str, values), map(int, counts))),
                },
                indent=2,
            )
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "predict":
        if not args.checkpoint:
            parser.error("predict requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["bands"] != bands:
            raise ValueError("Checkpoint band count differs from cube.")
        values = np.array(checkpoint["class_values"])
        mean = np.array(checkpoint["mean"], np.float32).reshape(-1, 1, 1)
        std = np.array(checkpoint["std"], np.float32).reshape(-1, 1, 1)
        size = checkpoint["patch_size"]
        coords = tiles(h, 0, w, size)
        ds = CubeTiles(cube, None, coords, size, mean, std, w)
        model = BENet(bands, len(values)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        prediction = np.empty((h, w), np.int64)
        with torch.no_grad():
            for i, (y, x) in enumerate(coords):
                image, _ = ds[i]
                pred = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
                hh, ww = (min(size, h - y), min(size, w - x))
                prediction[y : y + hh, x : x + ww] = values[pred[:hh, :ww]]
        np.save(output / "prediction.npy", prediction)
        savemat(output / "prediction.mat", {"prediction": prediction})
        print(f"Saved prediction in {output.resolve()}")
        return
    labels, key = load_labels(args.gt, (h, w), args.gt_key)
    # Reserve disjoint spatial regions, separated by an unused tile-width buffer.
    split = int(w * 0.7)
    val_start = split + args.patch_size
    if val_start >= w:
        raise ValueError("Patch size leaves no validation region.")
    values = np.unique(labels[:, :split])
    values = values[values != args.ignore_label]
    if len(values) < 2:
        raise ValueError("Training region needs at least two annotated classes.")
    unknown = set(np.unique(labels[:, val_start:])) - set(values) - {args.ignore_label}
    if unknown:
        raise ValueError(f"Validation contains classes absent from training: {unknown}")
    # Cross-entropy requires contiguous targets; retain original IDs for export.
    encoded = np.full((h, w), -100, np.int64)
    for i, v in enumerate(values):
        encoded[labels == v] = i
    # Estimate normalization using training columns only.
    means, stds = ([], [])
    for band in cube:
        data = np.asarray(band[:, :split], np.float32)
        finite = data[np.isfinite(data)]
        if not finite.size:
            raise ValueError("Training band has no finite values.")
        means.append(float(finite.mean()))
        stds.append(max(float(finite.std()), 1e-06))
    mean = np.array(means, np.float32).reshape(-1, 1, 1)
    std = np.array(stds, np.float32).reshape(-1, 1, 1)
    loaders = []
    for start, stop in [(0, split), (val_start, w)]:
        coords = [
            (y, x)
            for y, x in tiles(h, start, stop, args.patch_size)
            if np.any(
                encoded[y : y + args.patch_size, x : min(x + args.patch_size, stop)]
                != -100
            )
        ]
        if not coords:
            raise ValueError("A split has no annotated tiles.")
        ds = CubeTiles(cube, encoded, coords, args.patch_size, mean, std, stop)
        loaders.append(
            torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, shuffle=start == 0
            )
        )
    counts = np.bincount(
        encoded[:, :split][encoded[:, :split] >= 0], minlength=len(values)
    )
    weights = counts.sum() / (len(values) * counts)
    model = BENet(bands, len(values)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=-100,
    )
    history, best = ([], -1.0)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, (image, target) in enumerate(loaders[0]):
            optimizer.zero_grad()
            loss = loss_fn(model(image.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.max_steps and step + 1 >= args.max_steps:
                break
        metrics = evaluate(model, loaders[1], device, len(values))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "bands": bands,
                    "class_values": values.tolist(),
                    "mean": means,
                    "std": stds,
                    "patch_size": args.patch_size,
                    "gt_key": key,
                    "ignore_label": args.ignore_label,
                    "split_columns": [split, val_start],
                    "seed": args.seed,
                    "epoch": epoch + 1,
                    "metrics": metrics,
                },
                output / "best_model.pt",
            )
        (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best checkpoint: {(output / 'best_model.pt').resolve()}")


# Compatibility names for callers of the original implementation.
U_Net = BENet
spectralgradconv = SpectralGradientConv
pyrimad_decoder = PyramidDecoder
conv_block = ConvBlock
up_conv = UpConv

if __name__ == "__main__":
    main()
