# BE-Net for Patient_47

Updated script: D:\Code\BE-Net.py. The Downloads original is preserved.

Run in PowerShell:

```powershell
D:\Code\be_net_env\Scripts\python.exe D:\Code\BE-Net.py inspect
D:\Code\be_net_env\Scripts\python.exe D:\Code\BE-Net.py train --epochs 20 --output D:\Code\be_net_results
D:\Code\be_net_env\Scripts\python.exe D:\Code\BE-Net.py predict --checkpoint D:\Code\be_net_results\best_model.pt --output D:\Code\be_net_results
```

The default files are Patient_47/cube.dat and Patient_47_GT.mat. Keep cube.hdr beside cube.dat. Other files can be selected using --cube and --gt; --gt-key defaults to annotation_map.

The model consumes all 158 spectral bands. Ground truth labels 103, 104, 105 are mapped to contiguous training targets and restored in prediction.npy and prediction.mat. Label 0 is ignored by default; use --ignore-label to change this. Clinical meanings of the numeric classes are not inferred. mask_outliers is not automatically excluded: decide whether label 105 should be trained or ignored from your annotation protocol.

Training uses the left 70% of the image, leaves a patch-wide unused buffer, and validates on the remaining right region. Normalization statistics and class weights use training data only. Patient_47's validation region contains only class 103; validation is incomplete for classes 104 and 105 and is not a test of generalization to other patients. Use separate patients for a reliable final evaluation. Ignored pixels still receive a predicted class during inference; predictions are not masked by ground truth.

Outputs: best_model.pt (model, normalization, label mapping and split metadata), history.json (loss, accuracy, IoU and confusion matrices), prediction.npy and prediction.mat (801 x 801 maps).

A one-batch smoke training run is saved separately in D:\Code\be_net_smoke. This verifies execution; it is not a trained model suitable for practical use. Tile predictions use non-overlapping patches and may show seams. This is experimental segmentation code, not a validated clinical system.

Dependencies are installed in D:\Code\be_net_env; requirements-be-net.txt lists them. CPU is supported; CUDA is selected automatically when available in the installed PyTorch environment.

The source uses formal English comments and docstrings, standard Python formatting, and descriptive classes: BENet, SpectralGradientConv, ChannelBoundaryAttention, PyramidDecoder, ConvBlock, and UpConv. Original constructor names remain available as compatibility aliases. Parameter names in saved state dictionaries remain compatible with earlier checkpoints.
