# Verification record

- Source inspected: C:/Users/stanl/Downloads/BE-Net.py, D:/PDI/pdi_app.py, D:/PDI/pdi_patient_data.py.
- Architecture classes match the original source AST exactly; the original single-patient CubeTiles dataset is replaced, not part of the architecture.
- Five automated checks cover architecture preservation, ignored/absent labels, patient leakage, strict checkpoint reload and odd-sized native prediction, and synthetic training/resume equivalence.
- Synthetic fixture only: 77 generated 3-band 32 x 32 cubes. One training batch per epoch, two epochs. Resumed CPU weights/history match uninterrupted execution exactly. No synthetic performance numbers are offered as patient results.
- Patient_47 was extracted locally to D:/Code/pdi_data/Patient_47. Full inspection passed: 158 x 801 x 801, valid label IDs 0/103/104/105, finite voxels and header/file size consistency. Overlay rendered and visually inspected. Clinical/anatomical registration still requires qualified acquisition review.
- Archive audit inspected every available annotation and declared cube size: 76 archives, Patient_19 missing; 50 patients have extra label IDs. Cube values inside these other archives have not been fully scanned, and their alignment has not been visually verified.
- No real multi-patient training, GPU training, Colab runtime execution, final held-out testing, or PDI GUI inference integration was performed. Pending verified label policy and the complete 77-patient cohort.
- No patient data uploaded. All patient artifacts remain local. Dependency downloads were used to install matplotlib for local plot verification.

Run from D:/Code: `be_net_env\Scripts\python.exe -m pdi_pipeline.checks`.

Update: Patient_19.zip is now present at D:/PDI/data. Re-audit found all 77 expected archives with no header/annotation audit errors. Patient_19 has shape 158 x 801 x 801 and label IDs 0, 100, 101, 102, 202; ZIP CRC integrity check passed. Label policy remains unresolved. Full voxel scans and spatial review for the cohort remain pending.
