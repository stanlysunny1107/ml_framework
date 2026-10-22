# BE-Net patient pipeline

The BE-Net classes in model.py are copied exactly from the supplied Downloads/BE-Net.py. original_be_net.py is the unmodified source snapshot. No architecture change was needed: instantiate BENet(158, 3). The existing attention, encoder, decoder, spatial stencil, and interpolation stay intact. Checkpoints at different epochs are versions of this same architecture, not different architectures. Old single-patient checkpoints are deliberately unsupported because their preprocessing/split provenance is incompatible.

## Current dataset findings

The updated local archive audit found all 77 archives, including Patient_19. Fifty-one patients contain additional label IDs; all observed IDs are 0, 100, 101, 102, 103, 104, 105, 200, 201, 202, 300. See available_archive_audit.json for per-patient details. This is a snapshot of files available during this session. This checks headers, declared cube byte sizes and every annotation_map, without decompressing all cube values. It does not establish spectral consistency, finite cube values, or anatomical alignment; the inspect command performs the further computational checks and generates overlays. Extra label IDs are rejected, pending an explicit label policy. Do not silently map them to 0. Only ID 0 is ignored. The default mapping is contiguous training indices 0/1/2 to original IDs 103/104/105. No clinical names are assigned.

A successful full prepare is required before training. It requires exactly 77 unique patients and makes a seeded 53/12/12 patient split, records class coverage, rejects identical cubes, and computes pixel-weighted per-band population mean/std using all training cube pixels only. Zero-variance bands use std >= 1e-6. Nonfinite input is rejected. No data upload occurs in these tools.

## Local setup (PowerShell, from D:/Code)

Use Python 3.11 or 3.12 in a fresh environment for the pinned baseline dependencies. GPU users should install the CUDA-enabled PyTorch build appropriate for their driver before the other packages. The existing workspace Python is 3.14 and requires newer compatible packages; its verified versions are in verification_environment.txt. Reproducibility requires the same package versions, device type and hardware; CPU and GPU results need not be bitwise identical. Deterministic algorithms are requested and may raise if the selected device lacks an implementation.

    py -3.11 -m venv .venv-pdi
    .\.venv-pdi\Scripts\Activate.ps1
    python -m pip install -r pdi_pipeline/requirements.txt
    python -m pdi_pipeline.audit_archives --root D:/PDI/data --output pdi_pipeline/archive_audit.json
    python -m pdi_pipeline.import_archives --source D:/PDI/data --destination D:/Code/pdi_data

Import creates a private local copy of required cube/header/mask files only; allow roughly 31 GB for float32 158 x 801 x 801 cubes from 77 patients. Existing matching files are verified by CRC; conflicts fail. Folder discovery expects exactly one .dat and one .mat per patient. Do not export predictions back into this input tree. A patient ID must represent one person; reconcile pseudonyms/repeat acquisitions before use. Identical-cube hashes detect exact duplicates, not all possible repeat acquisitions.

    pdi_data/
      Patient_1/
        cube.dat
        cube.hdr
        Patient_1_GT.mat
      Patient_2/
        cube.dat
        cube.hdr
        Patient_2_GT.mat
      ...
      Patient_77/
        cube.dat
        cube.hdr
        Patient_77_GT.mat

Run `Copy-Item pdi_pipeline/config.example.json pdi_pipeline/config.json` in PowerShell to create the working configuration. Copy config.example.json to config.json and set dataset_root, output, and the truthful preprocessing identifier. The identifier records upstream calibration/preprocessing; it is an operator assertion, not an automatic determination from intensity values. Ordered wavelength values and units must match exactly in all cubes and inference headers. The pipeline never guesses band order or resamples bands. Missing wavelength metadata must be supplied from verified acquisition records. Never bypass this using invented values.

## Commands

    python -m pdi_pipeline.cli inspect --config pdi_pipeline/config.json
    python -m pdi_pipeline.cli prepare --config pdi_pipeline/config.json
    python -m pdi_pipeline.checks

Inspect scans every voxel for finiteness, validates ENVI dimensions/file size, 158 bands, spectral order/units, annotation shape, integer labels and allowed IDs, fingerprints files, and exports per-patient label counts and overlay previews. No implicit transpose, registration, crop or resize is applied. Review every overlay against known orientation/acquisition landmarks before training: equal square dimensions cannot prove image-mask alignment.

For a short check, copy config.json to config.smoke.json, set output to pdi_runs/smoke and epochs to 1, then run these commands. This still evaluates all validation patches/patients; max-steps limits only training and must not be used for real model selection.

    python -m pdi_pipeline.cli prepare --config pdi_pipeline/config.smoke.json
    python -m pdi_pipeline.cli train --config pdi_pipeline/config.smoke.json --max-steps 2

Full training and resume (epochs is the total target, not additional epochs):

    python -m pdi_pipeline.cli train --config pdi_pipeline/config.json
    python -m pdi_pipeline.cli train --config pdi_pipeline/config.json --resume pdi_runs/full/last_model.pt
    python -m pdi_pipeline.cli validate --config pdi_pipeline/config.json --checkpoint pdi_runs/full/best_model.pt
    python -m pdi_pipeline.cli compare --config pdi_pipeline/config.json

Train uses 64 x 64 nonoverlapping tiles, default batch size 2, ignored padding, unweighted cross entropy, Adam, and StepLR (halve every 30 epochs). Tiles with no annotations are omitted from training/loss; inference covers every pixel. Augmentation uses joint 90-degree rotations and horizontal flips only in training. Statistics and augmentation are unchanged by validation/test content. Full-patient predictions use the same nonoverlapping padded tiling for validation/test/inference; possible tile seams should be assessed using validation only. CUDA mixed precision is enabled when configured; CPU uses float32. Initial training and loader/RNG states are seeded; worker count 0 avoids worker RNG ambiguity.

Every epoch produces train/validation loss, validation pooled-pixel mean IoU, patient metrics, history CSV/JSON and curves.png. best_model.pt selects maximum pooled-pixel validation mean IoU; ties keep the earlier checkpoint. last_model.pt saves every epoch; epoch_NNNN.pt uses checkpoint_interval (default 5). All checkpoints include architecture, label mapping, normalization, preprocessing signature, epoch, validation metrics, training configuration, manifest reference and snapshot/hash, model/optimizer/scheduler/scaler and RNG states. Resume restores epoch-boundary state. Use the same hardware/software for the closest reproduction; training stops only after completed epochs. Use a separate output directory for a new experiment.

After all settings are finalized using training/validation, explicitly freeze the best checkpoint and run final testing:

    python -m pdi_pipeline.cli finalize --config pdi_pipeline/config.json
    python -m pdi_pipeline.cli test --config pdi_pipeline/config.json --checkpoint pdi_runs/full/best_model.pt

Finalization records checkpoint SHA256 and config hash; testing rejects changed configuration or weights, short-check models, and overwriting existing test results. This is a workflow guard, not a substitute for study governance: do not tune on test results or repeatedly finalize experiments against the same test cohort. Testing is never called by training or checkpoint comparison.

## Metrics

Ignore GT label 0 and padding in loss and all comparisons. Confusion rows are truth and columns predictions in [103,104,105] order. Per-class Dice = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN). A zero denominator is JSON null and CSV blank. Classes absent in truth but predicted receive IoU/Dice 0; absent in both are undefined and excluded from mean IoU. All-ignored patients have null accuracy/mean IoU and are excluded from patient means. Pixel accuracy = correct / all nonignored pixels.

JSON reports each patient's per-class metrics/confusion matrix, pooled-pixel aggregate metrics/confusion, and equal-weight patient-macro mean IoU and accuracy. CSV has patient, pooled_pixels and patient_macro scope labels; it includes per-class metrics and patient/pooled accuracy, while full confusion matrices are in JSON. Patient-macro mean IoU first averages defined classes per patient, then patients. Class support and missing-class lists must accompany interpretation. Validation checkpoint comparison uses validation only and exports JSON/CSV. No performance numbers are supplied without training.

## Prediction and PDI interface

    python -m pdi_pipeline.cli predict --config pdi_pipeline/config.json --checkpoint pdi_runs/full/best_model.pt --cube D:/private/NewPatient/cube.dat --prediction-output predictions/NewPatient
    python -m pdi_pipeline.cli predict --config pdi_pipeline/config.json --registry pdi_pipeline/checkpoint_registry.example.json --name "BE-Net validation-selected best (example)" --cube D:/private/NewPatient/cube.dat --prediction-output predictions/NewPatient --gt D:/private/NewPatient/NewPatient_GT.mat

The registry example points to a future checkpoint, not an existing trained model. Paths resolve relative to registry location. Compatibility metadata and strict model loading are checked. Load only locally trusted checkpoints (weights_only=True). Normalization/class mapping come from the selected checkpoint. GT is optional and only used after prediction for separate comparison metrics. Outputs are native H x W prediction.npy, prediction.mat with variable prediction, labeled prediction.png and prediction_metadata.json. Predictions cover every pixel, including unannotated areas; only GT comparison ignores 0.

    from pdi_pipeline.inference import Predictor
    predictor = Predictor.from_registry("pdi_pipeline/checkpoint_registry.example.json",
                                        "BE-Net validation-selected best (example)")
    result = predictor.predict(cube="D:/private/NewPatient/cube.dat",
                               output="predictions/NewPatient",
                               preprocessing="raw_ENVI_values_no_spectral_reordering")
    mask = result["mask"]
    metadata = result["metadata"]
    prediction_path = result["prediction_path"]

I inspected D:/PDI/pdi_app.py and its patient data loader (the workspace and D:/PDI copies have matching SHA256 hashes). Its existing Load prediction mask (.npy / .mat) button accepts these exports (MAT variable prediction). Model selection widgets currently browse paths without running inference. No GUI inference wiring has been installed or claimed. A future GUI adapter should run Predictor off the UI thread, populate the registry selector, call predict, and pass prediction_path into the existing record loader. GT-free inference works through this API; the current PDI viewer still expects GT/RGB to display a patient. Always display metadata notice: Research predictions, not validated clinical diagnoses.

## Colab privacy

PDI_Colab.ipynb includes setup and all workflow commands. Using Colab places patient data on Google infrastructure. Use it only with explicit authorization and approved storage/access policy; this work has not uploaded patient data. Local execution is the default. Do not put private cubes in public repositories or notebook outputs. Checkpoints/manifests contain patient IDs and paths, so treat them as sensitive research artifacts too.
