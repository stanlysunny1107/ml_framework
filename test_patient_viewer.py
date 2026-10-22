"""Checks for real patient loading, pixel spectra, and archive path isolation."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from patient_dataset import patient_record, import_patient_zip
from pdi_patient_data import load_patient_rasters, pixel_info
from pdi_app import PDIMainWindow, PatientImportWorker
from PyQt6.QtCore import QPointF

app = QApplication([])
from PyQt6.QtGui import QFontDatabase, QFont
for name in ('segoeui.ttf', 'segoeuib.ttf', 'arial.ttf'):
    QFontDatabase.addApplicationFont(str(Path(r'C:\Windows\Fonts')/name))
app.setFont(QFont('Segoe UI', 9))
record = patient_record(r'D:\PDI\data\Patient_47')
rasters = load_patient_rasters(record)
meta = rasters[3]
assert (meta['width'],meta['height']) == (801,801)
assert set(np.unique(meta['labels'])) == {0,103,104,105}
info = pixel_info(meta,200,350)
np.testing.assert_allclose([v for _,v in info['spectrum']], meta['cube'][:,350,200])
assert info['pred'] == 'No prediction loaded'
assert len(info['spectrum']) == 158
assert import_patient_zip(r'D:\PDI\data\Patient_47.zip',r'D:\PDI\data')['id']=='Patient_47'
with tempfile.TemporaryDirectory(dir=r'D:\Code') as temporary:
    root = Path(temporary)
    archive = root/'Patient_1.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('Patient_1/../../escape.txt','invalid')
    try:
        import_patient_zip(archive,root/'patients')
        raise AssertionError('Unsafe archive accepted')
    except ValueError:
        pass
    for patient in ('Patient_1','Patient_2'):
        archive = root/(patient+'.zip')
        with zipfile.ZipFile(archive,'w') as bundle:
            for filename,content in [('cube.dat',patient),('cube.hdr','ENVI'),(patient+'_GT.mat','test')]:
                bundle.writestr(patient+'/'+filename,content)
        import_patient_zip(archive,root/'patients')
    assert (root/'patients/Patient_1/cube.dat').read_text()=='Patient_1'
    assert (root/'patients/Patient_2/cube.dat').read_text()=='Patient_2'
window = PDIMainWindow()
window.resize(1536,960)
window.show()
app.processEvents()
assert window.active_patient_id == 'Patient_47'
assert window.gt_canvas.patient_meta['real_patient']
assert all(label.text()=='N/A' for label in window.metrics_labels.values())
point = QPointF(0.25, 0.75)
screen = window.gt_canvas.normalized_to_screen(point)
roundtrip = window.gt_canvas.screen_to_normalized(screen)
assert abs(roundtrip.x()-point.x()) < 1e-8 and abs(roundtrip.y()-point.y()) < 1e-8
rect = window.gt_canvas.image_rect()
assert abs(rect.width()/rect.height()-1) < 1e-8
received, errors = [], []
worker = PatientImportWorker([r'D:\PDI\data\Patient_47.zip'],Path(r'D:\PDI\data'))
worker.result.connect(received.append)
worker.error.connect(errors.append)
worker.start()
while worker.isRunning():
    app.processEvents()
    worker.wait(10)
app.processEvents()
assert len(received)==1 and not errors, errors
window.grab().save(r'D:\Code\pdi_real_patient_preview.png')
window.close()
print('PASS: real images, labels, measured spectra, ZIP isolation, and viewer rendering')

# Verify switching between two independent patient acquisitions and masks.
from scipy.io import savemat
from PyQt6.QtGui import QImage, QColor
with tempfile.TemporaryDirectory(dir=r'D:\Code') as temporary:
    root = Path(temporary)
    for number, label in [(901,103),(902,104)]:
        folder = root/f'Patient_{number}'
        folder.mkdir()
        cube = np.full((3,32,32),number/1000,np.float32)
        cube.tofile(folder/'cube.dat')
        (folder/'cube.hdr').write_text('ENVI\nlines = 32\nsamples = 32\nbands = 3\ndata type = 4\ninterleave = bsq\nbyte order = 0\nwavelength = {500, 600, 700}\n')
        savemat(folder/f'Patient_{number}_GT.mat', {'annotation_map':np.full((32,32),label,np.int16)})
        image = QImage(32,32,QImage.Format.Format_RGB32)
        image.fill(QColor('red'))
        image.save(str(folder/f'Patient_{number}_RGB.png'))
    os.environ['PDI_DATA_ROOT'] = str(root)
    viewer = PDIMainWindow()
    assert viewer.patient_selector.count()==2
    viewer.switch_patient('Patient_902')
    assert viewer.active_patient_id=='Patient_902'
    assert viewer.mask_input.line_edit.text().endswith('Patient_902_GT.mat')
    assert int(viewer.gt_canvas.patient_meta['labels'][0,0])==104
    assert np.isclose(viewer.gt_canvas.patient_meta['cube'][0,0,0],0.902)
    viewer.switch_patient('Patient_901')
    assert int(viewer.gt_canvas.patient_meta['labels'][0,0])==103
    viewer.display_selector.setCurrentText('RGB + mask')
    assert viewer.gt_canvas.show_tissue and viewer.gt_canvas.show_overlay
    viewer.display_selector.setCurrentText('RGB only')
    assert viewer.gt_canvas.show_tissue and not viewer.gt_canvas.show_overlay
    viewer.display_selector.setCurrentText('Mask only')
    assert not viewer.gt_canvas.show_tissue and viewer.gt_canvas.show_overlay
    viewer.gt_canvas.patient_meta = None
    viewer.pred_canvas.patient_meta = None
    viewer.patient_rasters_cache.clear()
    viewer.close()
    viewer.deleteLater()
    from PyQt6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del viewer
    import gc
    gc.collect()
    os.environ.pop('PDI_DATA_ROOT')
print('PASS: patient selection changes cube and annotation together; all display modes work')

# Validate metric arithmetic and prediction map formats independently of the UI.
from pdi_patient_data import classification_metrics, load_prediction
truth = np.array([[103,103,103,103,104,104,104,104]])
pred = np.array([[103,103,103,104,103,103,104,104]])
metrics = classification_metrics(truth,pred)
assert metrics['Accuracy']=='0.6250'
assert metrics['Sensitivity']=='0.6250'
assert metrics['Specificity']=='0.6250'
assert metrics['F1-Score']=='0.6190'
assert metrics['MCC']=='0.2582'
assert metrics['AUC']=='N/A'
assert classification_metrics(np.array([[103,104,0]]),np.array([[103,104,104]]))['Accuracy']=='1.0000'
with tempfile.TemporaryDirectory(dir=r'D:\Code') as temporary:
    root = Path(temporary)
    np.save(root/'prediction.npy',pred)
    savemat(root/'prediction.mat',{'prediction':pred})
    np.testing.assert_array_equal(load_prediction(root/'prediction.npy',truth.shape),pred)
    np.testing.assert_array_equal(load_prediction(root/'prediction.mat',truth.shape),pred)
    try:
        load_prediction(root/'prediction.npy',(32,32))
        raise AssertionError('Misaligned prediction accepted')
    except ValueError:
        pass
print('PASS: known multiclass metrics, ignored background, NPY/MAT prediction loading')
