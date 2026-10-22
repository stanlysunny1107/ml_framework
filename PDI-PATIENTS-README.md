# PDI viewer: real patient datasets

This update changes the PDI viewer only. BE-Net training is unchanged.

Start the updated viewer with run_patient_viewer.bat, or:

```powershell
D:\Code\be_net_env\Scripts\python.exe D:\PDI\pdi_app.py
```

Use the browse button next to IMPORT PATIENT ZIP / CUBE. Select multiple Patient_*.zip archives (Ctrl+A selects all files in the dialog), or select an extracted patient's cube.dat. Import reads the cube.hdr and matching *_GT.mat, *_GT.png, and *_RGB.png from the same patient folder. ZIP import runs in a background thread with progress; Stop finishes the current patient before stopping. Each archive retains its own Patient_<number> directory, so repeated cube.dat names do not collide. Existing files are not overwritten. Already extracted archives are accepted when their member paths and sizes match; existing file contents are not re-hashed.

Default patient folder: D:\PDI\data. To select another destination before launch:

```powershell
$env:PDI_DATA_ROOT = 'D:\MyPatientData'
D:\Code\be_net_env\Scripts\python.exe D:\PDI\pdi_app.py
```

Imported patient folders are discovered again at startup. Use the searchable patient selector to switch patients. The comparison dialog displays six patients per page. Only the active patient's full arrays are cached in the main view.

The viewer uses RGB for the tissue image, annotation_map from the MATLAB file for pixel label IDs, the GT PNG for overlay colors, and cube.dat for measured spectra. Image dimensions must match; images retain their original aspect ratio. Numeric annotation IDs are displayed without assigning clinical meanings. Background label 0 is transparent in the overlay. The opacity slider controls overlay visibility.

PATIENT ANNOTATIONS (.MAT) selects a replacement annotation_map for the active patient for the current session; it does not overwrite the source annotation file. Replacement maps must align with the cube and contain finite integer IDs. Their colors are generated from label IDs because the original GT PNG may no longer match.

Use Load prediction mask (.npy / .mat) to attach a prediction to the active patient. A .npy file contains an H x W array of original integer class IDs (for Patient_47, IDs such as 103, 104, 105). A .mat file must contain the same array in a variable named prediction. Shape and finite integer values are checked. A prediction.npy beside cube.dat is also loaded automatically. Selected external prediction paths are retained for the current session.

The viewer compares predictions with annotations only at pixels whose ground-truth ID is nonzero. Accuracy is pixel accuracy. Sensitivity, specificity, and F1 are macro averages over ground-truth classes; undefined terms are omitted. MCC is the multiclass Matthews correlation coefficient. AUC remains N/A because integer label maps contain no class probabilities. Metrics remain N/A without predictions. These scores measure agreement on the current image, not independent patient-level validation.

Model/checkpoint selectors retain file paths but do not execute models. Inference must be performed separately. The prediction panel remains empty in Mask only mode until predictions are loaded. RGB + mask and RGB only modes can show the acquisition even without predictions. Spectra are read directly from the cube and plotted with automatic axis limits; they are not normalized for display.

The 72 Patient_*.zip archives found in Downloads total about 29.3 GB uncompressed. They have not been bulk-extracted. Only Patient_47 has been exercised with real data; the implementation does not impose a 77-patient limit. Add any missing patient archives through the same import dialog.

Runtime dependencies: requirements-pdi-viewer.txt. Verification: test_patient_viewer.py exercises real image and annotation loading, measured spectra, archive path isolation, existing archive handling, and offscreen viewer rendering.

The layout follows the provided PDI reference: two stacked viewports, a right-hand data sidebar, opacity control, metrics panel, and spectrum diagnostics. SELECT PATIENT switches the complete acquisition and its paired annotation map. DISPLAY offers Mask only (default), RGB + mask, and RGB only. Masks are shown at their original aspect ratio. The patient selector is populated from imported or extracted data; ZIPs must be imported before their patients appear.
