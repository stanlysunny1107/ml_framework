import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from patient_dataset import discover_patients, patient_record, import_patient_zip
from pdi_patient_data import load_patient_rasters, pixel_info
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, pyqtSignal, QSize, QThread
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QIcon,
    QPixmap,
    QImage,
    QCursor,
    QLinearGradient,
    QAction,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QGridLayout,
    QFrame,
    QDialog,
    QSplitter,
    QToolTip,
    QMessageBox,
    QComboBox,
    QProgressDialog,
)

# ---------------------------------------------------------------------------
# COLOR PALETTE CONSTANTS
# ---------------------------------------------------------------------------
BG_WINDOW = QColor("#111113")
BG_TITLEBAR = QColor("#202125")
BG_SIDEBAR = QColor("#25272d")
BG_CARD = QColor("#25272d")
BG_CANVAS_GT = QColor("#17171a")
BG_CANVAS_PRED = QColor("#281648")
GRID_LINE_GT = QColor("#0e0e10")
GRID_LINE_PRED = QColor("#38195e")

TEXT_WHITE = QColor("#ffffff")
TEXT_MUTED = QColor("#8d909b")
TEXT_DIM = QColor("#666870")
ACCENT_BLUE = QColor("#2856d8")
ACCENT_CYAN = QColor("#29b6f6")

SWATCH_COLORS = [
    ("#0000ff", "Annotation label 103"),
    ("#00ff00", "Annotation label 104"),
    ("#ffffff", "Annotation label 105"),
    ("#000000", "Unannotated (0)"),
]

DEFAULT_METRICS = {
    key: "N/A"
    for key in ("Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC", "MCC")
}


def synthesize_patient_rasters(patient_dict, width=320, height=240):
    """Load native-resolution patient images; dimensions are retained for compatibility."""
    return load_patient_rasters(patient_dict)


def get_pixel_clinical_info(meta, px, py):
    """Read numeric annotations and measured spectra without inferring tissue types."""
    return pixel_info(meta, px, py)


# ---------------------------------------------------------------------------
# CUSTOM TITLE BAR
# ---------------------------------------------------------------------------
class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setFixedHeight(44)
        self.drag_position = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 10, 0)
        layout.setSpacing(10)

        # Title Label: "PDI   Prediction Diagnostics Interface"
        self.title_layout = QHBoxLayout()
        self.title_layout.setSpacing(10)

        title_pdi = QLabel("PDI")
        title_pdi.setStyleSheet(
            "color: #ffffff; font-size: 14px; font-weight: 800; font-family: 'Segoe UI', Arial, sans-serif;"
        )

        title_desc = QLabel("Prediction Diagnostics Interface")
        title_desc.setMinimumWidth(270)
        title_desc.setStyleSheet(
            "color: #ababaf; font-size: 13px; font-weight: 400; font-family: 'Segoe UI', Arial, sans-serif;"
        )

        self.title_layout.addWidget(title_pdi)
        self.title_layout.addWidget(title_desc)
        self.title_layout.addStretch()

        layout.addLayout(self.title_layout, 1)

        # Window Controls: Minimize, Maximize/Restore, Close
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)

        self.btn_min = QPushButton("—")
        self.btn_min.setFixedSize(36, 28)
        self.btn_min.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_min.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ababaf;
                border: none;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #2f323a;
                color: #ffffff;
            }
        """)
        self.btn_min.clicked.connect(self.minimize_window)

        self.btn_max = QPushButton("☐")
        self.btn_max.setFixedSize(36, 28)
        self.btn_max.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_max.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ababaf;
                border: none;
                font-size: 13px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #2f323a;
                color: #ffffff;
            }
        """)
        self.btn_max.clicked.connect(self.toggle_maximize)

        self.btn_close = QPushButton("X")
        self.btn_close.setFixedSize(36, 28)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ababaf;
                border: none;
                font-size: 13px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #e81123;
                color: #ffffff;
            }
        """)
        self.btn_close.clicked.connect(self.close_window)

        btn_layout.addWidget(self.btn_min)
        btn_layout.addWidget(self.btn_max)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.parent_window:
            self.drag_position = (
                event.globalPosition().toPoint()
                - self.parent_window.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self.drag_position
            and self.parent_window
        ):
            if not self.parent_window.isMaximized():
                self.parent_window.move(
                    event.globalPosition().toPoint() - self.drag_position
                )
                event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()

    def minimize_window(self):
        if self.parent_window:
            self.parent_window.showMinimized()

    def toggle_maximize(self):
        if self.parent_window:
            if self.parent_window.isMaximized():
                self.parent_window.showNormal()
                self.btn_max.setText("☐")
            else:
                self.parent_window.showMaximized()
                self.btn_max.setText("❐")

    def close_window(self):
        if self.parent_window:
            self.parent_window.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_TITLEBAR)
        painter.setPen(QColor("#18191c"))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()


# ---------------------------------------------------------------------------
# FLOATING TOOLBAR OVERLAY
# ---------------------------------------------------------------------------
class FloatingToolbar(QWidget):
    zoom_in_clicked = pyqtSignal()
    zoom_out_clicked = pyqtSignal()
    reset_view_clicked = pyqtSignal()
    tool_selected = pyqtSignal(str)
    toggle_tissue_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(38, 238)
        self.active_tool = "pointer"
        self.tissue_visible = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(5)

        # 1. Zoom In (+)
        self.btn_zoom_in = self.create_tool_button(
            "+", "Zoom In (Up to 40x Pixel Level)"
        )
        self.btn_zoom_in.clicked.connect(self.zoom_in_clicked.emit)

        # 2. Zoom Out (-)
        self.btn_zoom_out = self.create_tool_button("—", "Zoom Out")
        self.btn_zoom_out.clicked.connect(self.zoom_out_clicked.emit)

        # 3. Reset View 1:1
        self.btn_reset = self.create_tool_button("Fit", "Reset View 1:1 (Fit)")
        self.btn_reset.clicked.connect(self.reset_view_clicked.emit)

        # 4. Pointer / Cursor
        self.btn_pointer = self.create_tool_button("Ptr", "Pointer / Inspect Pixel")
        self.btn_pointer.clicked.connect(lambda: self.set_tool("pointer"))

        # 5. Pan Tool
        self.btn_pan = self.create_tool_button("Pan", "Pan Canvas (Drag)")
        self.btn_pan.clicked.connect(lambda: self.set_tool("pan"))

        # 6. ROI / Box Select
        self.btn_roi = self.create_tool_button("ROI", "ROI Box Select")
        self.btn_roi.clicked.connect(lambda: self.set_tool("roi"))

        # 7. Toggle Tissue / Features
        self.btn_tissue = self.create_tool_button(
            "RGB", "Toggle underlying patient RGB image"
        )
        self.btn_tissue.clicked.connect(self.toggle_tissue_clicked.emit)

        layout.addWidget(self.btn_zoom_in)
        layout.addWidget(self.btn_zoom_out)
        layout.addWidget(self.btn_reset)
        layout.addWidget(self.btn_pointer)
        layout.addWidget(self.btn_pan)
        layout.addWidget(self.btn_roi)
        layout.addWidget(self.btn_tissue)

        self.update_tool_styles()

    def create_tool_button(self, icon_text, tooltip):
        btn = QPushButton(icon_text)
        btn.setFixedSize(30, 26)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        return btn

    def set_tool(self, tool_name):
        self.active_tool = tool_name
        self.update_tool_styles()
        self.tool_selected.emit(tool_name)

    def update_tool_styles(self):
        base_style = """
            QPushButton {
                background: transparent;
                color: #ababaf;
                border: none;
                font-size: 13px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #343740;
                color: #ffffff;
            }
        """
        active_style = """
            QPushButton {
                background: #253347;
                color: #29b6f6;
                border: 1px solid #1976d2;
                font-size: 13px;
                border-radius: 4px;
                font-weight: bold;
            }
        """
        self.btn_zoom_in.setStyleSheet(base_style)
        self.btn_zoom_out.setStyleSheet(base_style)
        self.btn_reset.setStyleSheet(base_style)
        self.btn_tissue.setStyleSheet(base_style)
        self.btn_pointer.setStyleSheet(
            active_style if self.active_tool == "pointer" else base_style
        )
        self.btn_pan.setStyleSheet(
            active_style if self.active_tool == "pan" else base_style
        )
        self.btn_roi.setStyleSheet(
            active_style if self.active_tool == "roi" else base_style
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        painter.setBrush(QBrush(QColor(37, 39, 45, 235)))
        painter.setPen(QPen(QColor(60, 64, 74, 180), 1))
        painter.drawRoundedRect(rect, 8, 8)
        painter.end()


# ---------------------------------------------------------------------------
# GRID CANVAS WIDGET (Ground Truth & Predictions Viewport)
# Features:
# - Simultaneous bidirectional synchronization
# - Zoom to pixel level (up to 40x) with distinguishable pixel grid
# - Underlying tissue morphology (Blood vessels, Light glare, Cells) visible behind predictions
# ---------------------------------------------------------------------------
class GridCanvasWidget(QWidget):
    pixel_selected_detailed = pyqtSignal(dict)
    pixel_selected = pyqtSignal(float, float)  # Legacy compatible signal

    def __init__(self, mode="ground_truth", parent=None):
        super().__init__(parent)
        self.mode = mode  # "ground_truth" or "predictions"
        self.setMinimumSize(400, 200)
        self.setMouseTracking(True)

        # Viewport Transform State
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.opacity = 0.65
        self.active_tool = "pointer"
        self.show_tissue = False
        self.show_overlay = True
        self.show_pixel_grid = True

        # Mouse tracking & ROI
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.is_drawing_roi = False
        self.roi_start = QPointF()
        self.roi_current = QPointF()
        self.roi_rect = None  # QRectF in normalized coords

        # Hover & Selected Pixel
        self.is_hovering = False
        self.hover_norm_pos = QPointF(-1, -1)
        self.selected_pixel = None  # (px, py)
        self.selected_norm_pos = None

        # Data & Pixmaps
        self.shapes = []
        self.raster_w = 320
        self.raster_h = 240
        self.tissue_pixmap = None
        self.gt_pixmap = None
        self.pred_pixmap = None
        self.patient_meta = None

        # Partner Canvas for 2-way Real-Time Synchronization
        self.partner_canvas = None
        self._is_syncing = False

        # Floating Toolbar Overlay
        self.toolbar = FloatingToolbar(self)
        self.toolbar.zoom_in_clicked.connect(self.zoom_in)
        self.toolbar.zoom_out_clicked.connect(self.zoom_out)
        self.toolbar.reset_view_clicked.connect(self.reset_view)
        self.toolbar.tool_selected.connect(self.on_tool_selected)
        self.toolbar.toggle_tissue_clicked.connect(self.toggle_tissue)

    # -----------------------------------------------------------------------
    # BI-DIRECTIONAL SYNCHRONIZATION WITH PARTNER CANVAS
    # -----------------------------------------------------------------------
    def connect_partner(self, partner):
        """Pairs this canvas with the partner canvas for continuous simultaneous updates."""
        self.partner_canvas = partner

    def sync_view(self, zoom, pan_offset):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.zoom = zoom
            self.pan_offset = QPointF(pan_offset)
            self.update()
        finally:
            self._is_syncing = False

    def sync_hover(self, norm_pos, is_hovering):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.hover_norm_pos = QPointF(norm_pos)
            self.is_hovering = is_hovering
            self.update()
        finally:
            self._is_syncing = False

    def sync_selected_pixel(self, norm_pos, px, py):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.selected_norm_pos = QPointF(norm_pos) if norm_pos else None
            self.selected_pixel = (px, py) if px is not None else None
            self.update()
        finally:
            self._is_syncing = False

    def sync_tool(self, tool_name):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.active_tool = tool_name
            self.toolbar.set_tool(tool_name)
            self.update_cursor()
            self.update()
        finally:
            self._is_syncing = False

    def sync_roi(self, roi_rect, is_drawing):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.roi_rect = QRectF(roi_rect) if roi_rect else None
            self.is_drawing_roi = is_drawing
            self.update()
        finally:
            self._is_syncing = False

    def sync_tissue_toggle(self, show_tissue):
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            self.show_tissue = show_tissue
            self.update()
        finally:
            self._is_syncing = False

    # -----------------------------------------------------------------------
    # VIEWPORT CONTROL & ZOOM (Up to 40x Pixel Level)
    # -----------------------------------------------------------------------
    def apply_view(self, zoom, pan_offset):
        self.zoom = float(np.clip(zoom, 0.5, 40.0))
        self.pan_offset = QPointF(pan_offset)
        self.update()
        if self.partner_canvas and not self._is_syncing:
            self._is_syncing = True
            try:
                self.partner_canvas.sync_view(self.zoom, self.pan_offset)
            finally:
                self._is_syncing = False

    def zoom_in(self):
        new_zoom = min(40.0, self.zoom * 1.35)
        self.apply_view(new_zoom, self.pan_offset)

    def zoom_out(self):
        new_zoom = max(0.5, self.zoom / 1.35)
        self.apply_view(new_zoom, self.pan_offset)

    def reset_view(self):
        self.apply_view(1.0, QPointF(0, 0))

    def toggle_tissue(self):
        self.show_tissue = not self.show_tissue
        self.update()
        if self.partner_canvas and not self._is_syncing:
            self._is_syncing = True
            try:
                self.partner_canvas.sync_tissue_toggle(self.show_tissue)
            finally:
                self._is_syncing = False

    def set_opacity(self, opacity_val):
        self.opacity = opacity_val
        self.update()

    def set_shapes(self, shapes):
        self.shapes = shapes
        self.update()

    def set_raster_data(self, tissue_pix, gt_pix, pred_pix, meta):
        self.tissue_pixmap = tissue_pix
        self.gt_pixmap = gt_pix
        self.pred_pixmap = pred_pix
        self.patient_meta = meta
        if meta:
            self.raster_w = meta.get("width", 320)
            self.raster_h = meta.get("height", 240)
        self.update()

    def on_tool_selected(self, tool_name):
        self.active_tool = tool_name
        self.update_cursor()
        if self.partner_canvas and not self._is_syncing:
            self._is_syncing = True
            try:
                self.partner_canvas.sync_tool(tool_name)
            finally:
                self._is_syncing = False

    def update_cursor(self):
        if self.active_tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.active_tool == "roi":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # -----------------------------------------------------------------------
    # COORDINATE MAPPING HELPERS
    # -----------------------------------------------------------------------
    def image_rect(self):
        """Return the centered image bounds, preserving the acquisition aspect ratio."""
        scale = min(
            self.width() / max(self.raster_w, 1), self.height() / max(self.raster_h, 1)
        )
        width, height = self.raster_w * scale, self.raster_h * scale
        return QRectF(
            (self.width() - width) / 2, (self.height() - height) / 2, width, height
        )

    def screen_to_normalized(self, pos):
        """Invert image fitting, synchronized zoom, and pan for accurate pixel selection."""
        cx, cy = self.width() / 2, self.height() / 2
        x = (pos.x() - cx - self.pan_offset.x()) / self.zoom + cx
        y = (pos.y() - cy - self.pan_offset.y()) / self.zoom + cy
        rect = self.image_rect()
        return QPointF(
            (x - rect.left()) / max(rect.width(), 1),
            (y - rect.top()) / max(rect.height(), 1),
        )

    def normalized_to_screen(self, pos):
        """Map normalized image coordinates through fitting, zoom, and pan."""
        cx, cy = self.width() / 2, self.height() / 2
        rect = self.image_rect()
        x, y = (
            rect.left() + pos.x() * rect.width(),
            rect.top() + pos.y() * rect.height(),
        )
        return QPointF(
            (x - cx) * self.zoom + cx + self.pan_offset.x(),
            (y - cy) * self.zoom + cy + self.pan_offset.y(),
        )

    # -----------------------------------------------------------------------
    # MOUSE EVENTS & SIMULTANEOUS ACTIONS
    # -----------------------------------------------------------------------
    def wheelEvent(self, event):
        """Smooth mouse-wheel zoom centered on cursor position, synced to partner."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.20 if delta > 0 else (1.0 / 1.20)
        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.5, 40.0))
        if new_zoom == old_zoom:
            return

        pos = event.position()
        w, h = float(self.width()), float(self.height())
        cx, cy = w / 2.0, h / 2.0

        sx = pos.x() - cx - self.pan_offset.x()
        sy = pos.y() - cy - self.pan_offset.y()

        new_pan_x = self.pan_offset.x() - sx * (new_zoom / old_zoom - 1.0)
        new_pan_y = self.pan_offset.y() - sy * (new_zoom / old_zoom - 1.0)

        self.apply_view(new_zoom, QPointF(new_pan_x, new_pan_y))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            norm_pos = self.screen_to_normalized(event.position())

            if self.active_tool == "pan":
                self.is_dragging = True
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

            elif self.active_tool == "roi":
                self.is_drawing_roi = True
                self.roi_start = norm_pos
                self.roi_current = norm_pos
                self.roi_rect = QRectF(norm_pos, norm_pos)
                if self.partner_canvas and not self._is_syncing:
                    self._is_syncing = True
                    try:
                        self.partner_canvas.sync_roi(self.roi_rect, True)
                    finally:
                        self._is_syncing = False
                self.update()

            elif self.active_tool == "pointer":
                # Select pixel and query clinical information
                px = int(np.clip(norm_pos.x() * self.raster_w, 0, self.raster_w - 1))
                py = int(np.clip(norm_pos.y() * self.raster_h, 0, self.raster_h - 1))
                self.selected_pixel = (px, py)
                self.selected_norm_pos = norm_pos

                # Emit legacy signal
                self.pixel_selected.emit(norm_pos.x(), norm_pos.y())

                # Emit detailed clinical signal
                if self.patient_meta:
                    info = get_pixel_clinical_info(self.patient_meta, px, py)
                    self.pixel_selected_detailed.emit(info)

                # Sync pixel selection to partner canvas
                if self.partner_canvas and not self._is_syncing:
                    self._is_syncing = True
                    try:
                        self.partner_canvas.sync_selected_pixel(norm_pos, px, py)
                    finally:
                        self._is_syncing = False
                self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        norm_pos = self.screen_to_normalized(pos)
        self.is_hovering = True
        self.hover_norm_pos = norm_pos

        # Sync hover crosshair to partner canvas
        if self.partner_canvas and not self._is_syncing:
            self._is_syncing = True
            try:
                self.partner_canvas.sync_hover(norm_pos, True)
            finally:
                self._is_syncing = False

        if self.is_dragging and self.active_tool == "pan":
            delta = event.pos() - self.last_mouse_pos
            new_pan = self.pan_offset + QPointF(delta.x(), delta.y())
            self.apply_view(self.zoom, new_pan)
            self.last_mouse_pos = event.pos()

        elif self.is_drawing_roi and self.active_tool == "roi":
            self.roi_current = norm_pos
            r = QRectF(
                min(self.roi_start.x(), self.roi_current.x()),
                min(self.roi_start.y(), self.roi_current.y()),
                abs(self.roi_start.x() - self.roi_current.x()),
                abs(self.roi_start.y() - self.roi_current.y()),
            )
            self.roi_rect = r
            if self.partner_canvas and not self._is_syncing:
                self._is_syncing = True
                try:
                    self.partner_canvas.sync_roi(self.roi_rect, True)
                finally:
                    self._is_syncing = False

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_dragging:
                self.is_dragging = False
                self.update_cursor()
            if self.is_drawing_roi:
                self.is_drawing_roi = False
                if self.partner_canvas and not self._is_syncing:
                    self._is_syncing = True
                    try:
                        self.partner_canvas.sync_roi(self.roi_rect, False)
                    finally:
                        self._is_syncing = False
                self.update()

    def leaveEvent(self, event):
        self.is_hovering = False
        if self.partner_canvas and not self._is_syncing:
            self._is_syncing = True
            try:
                self.partner_canvas.sync_hover(self.hover_norm_pos, False)
            finally:
                self._is_syncing = False
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.toolbar.move(self.width() - self.toolbar.width() - 20, 20)

    # -----------------------------------------------------------------------
    # PAINTING & PIXEL-LEVEL DISTINGUISHABILITY
    # -----------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = float(self.width()), float(self.height())

        # 1. Fill base dark background
        bg_col = BG_CANVAS_GT if self.mode == "ground_truth" else BG_CANVAS_PRED
        painter.fillRect(self.rect(), bg_col)

        # 2. Draw standard spatial grid in background
        grid_cols, grid_rows = 13, 5
        line_color = GRID_LINE_GT if self.mode == "ground_truth" else GRID_LINE_PRED
        painter.setPen(QPen(line_color, 1.2))
        col_w = w / grid_cols
        row_h = h / grid_rows
        for c in range(1, grid_cols):
            x = int(c * col_w)
            painter.drawLine(x, 0, x, int(h))
        for r in range(1, grid_rows):
            y = int(r * row_h)
            painter.drawLine(0, y, int(w), y)

        # 3. Apply Zoom & Pan Transform
        painter.save()
        cx, cy = w / 2.0, h / 2.0
        painter.translate(cx + self.pan_offset.x(), cy + self.pan_offset.y())
        painter.scale(self.zoom, self.zoom)
        painter.translate(-cx, -cy)

        image_rect = self.image_rect()
        painter.translate(image_rect.left(), image_rect.top())
        painter.scale(image_rect.width() / max(w, 1), image_rect.height() / max(h, 1))
        dest_rect = QRectF(0, 0, w, h)
        src_rect = QRectF(0, 0, self.raster_w, self.raster_h)

        # Disable smoothing at all zooms so pixels remain crisp and true
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        # Draw underlying tissue (Blood vessels, Light glare, Normal cells)
        if self.show_tissue and self.tissue_pixmap:
            painter.drawPixmap(dest_rect, self.tissue_pixmap, src_rect)

        # Draw annotation/prediction overlay with user-controlled opacity
        overlay = self.gt_pixmap if self.mode == "ground_truth" else self.pred_pixmap
        if overlay and self.show_overlay and self.opacity > 0.005:
            painter.save()
            painter.setOpacity(self.opacity)
            painter.drawPixmap(dest_rect, overlay, src_rect)
            painter.restore()

        # Fallback vector shapes if pixmap not ready
        if not overlay and self.shapes:
            alpha = int(self.opacity * 255)
            for shape in self.shapes:
                hex_color = shape.get("color", "#2856d8")
                base_col = QColor(hex_color)
                fill_col = QColor(
                    base_col.red(), base_col.green(), base_col.blue(), alpha
                )
                painter.setBrush(QBrush(fill_col))
                painter.setPen(Qt.PenStyle.NoPen)
                stype = shape.get("type", "circle")
                scx = shape.get("cx", 0.5) * w
                scy = shape.get("cy", 0.5) * h
                if stype == "ellipse":
                    rx = shape.get("rx", 0.1) * w
                    ry = shape.get("ry", 0.1) * h
                    painter.drawEllipse(QPointF(scx, scy), rx, ry)
                else:
                    r = shape.get("r", 0.05) * min(w, h * 2)
                    painter.drawEllipse(QPointF(scx, scy), r, r)

        # 4. DISTINGUISHABLE PIXEL GRID (Active at high zoom)
        # When zoomed in, draw crisp boundary lines separating every single pixel!
        pixel_w = w / float(self.raster_w)
        pixel_h = h / float(self.raster_h)
        eff_size = pixel_w * self.zoom

        if self.show_pixel_grid and eff_size >= 6.0:
            inv_trans, ok = painter.combinedTransform().inverted()
            vis_poly = inv_trans.map(QRectF(self.rect()))
            vis_rect = vis_poly.boundingRect()

            min_ix = max(0, int(vis_rect.left() / pixel_w) - 1)
            max_ix = min(self.raster_w, int(vis_rect.right() / pixel_w) + 2)
            min_iy = max(0, int(vis_rect.top() / pixel_h) - 1)
            max_iy = min(self.raster_h, int(vis_rect.bottom() / pixel_h) + 2)

            grid_alpha = int(np.clip((eff_size - 6.0) * 9.0, 25.0, 85.0))
            painter.setPen(QPen(QColor(255, 255, 255, grid_alpha), 0.75))

            for ix in range(min_ix, max_ix + 1):
                xp = ix * pixel_w
                painter.drawLine(
                    QPointF(xp, min_iy * pixel_h), QPointF(xp, max_iy * pixel_h)
                )
            for iy in range(min_iy, max_iy + 1):
                yp = iy * pixel_h
                painter.drawLine(
                    QPointF(min_ix * pixel_w, yp), QPointF(max_ix * pixel_w, yp)
                )

        # 5. SELECTED PIXEL RETICLE (Synchronized Marker)
        if self.selected_pixel is not None:
            spx, spy = self.selected_pixel
            sx = spx * pixel_w
            sy = spy * pixel_h
            sel_rect = QRectF(sx, sy, pixel_w, pixel_h)

            # Glowing border and corner reticles
            painter.setBrush(QBrush(QColor(41, 182, 246, 50)))
            painter.setPen(QPen(QColor("#29b6f6"), max(1.5, 2.0 / self.zoom)))
            painter.drawRect(sel_rect)

            # Crosshair circle
            scx = sx + pixel_w / 2.0
            scy = sy + pixel_h / 2.0
            rad = max(pixel_w * 0.8, 5.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#f5a10c"), max(1.2, 1.8 / self.zoom)))
            painter.drawEllipse(QPointF(scx, scy), rad, rad)

        # 6. HOVERED PIXEL BOX & CROSSHAIR (Synchronized Marker)
        if (
            self.is_hovering
            and 0.0 <= self.hover_norm_pos.x() <= 1.0
            and 0.0 <= self.hover_norm_pos.y() <= 1.0
        ):
            hpx = int(self.hover_norm_pos.x() * self.raster_w)
            hpy = int(self.hover_norm_pos.y() * self.raster_h)
            hx = hpx * pixel_w
            hy = hpy * pixel_h
            hov_rect = QRectF(hx, hy, pixel_w, pixel_h)

            painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
            painter.setPen(QPen(QColor(255, 255, 255, 180), max(1.0, 1.2 / self.zoom)))
            painter.drawRect(hov_rect)

        # 7. ROI BOX (Region of Interest)
        if self.roi_rect and (self.roi_rect.width() > 0 or self.roi_rect.height() > 0):
            rx = self.roi_rect.x() * w
            ry = self.roi_rect.y() * h
            rw = self.roi_rect.width() * w
            rh = self.roi_rect.height() * h
            roi_box = QRectF(rx, ry, rw, rh)

            painter.setBrush(QBrush(QColor(41, 182, 246, 40)))
            pen = QPen(QColor("#29b6f6"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(roi_box)

        painter.restore()

        # 8. VIEWPORT HUD OVERLAYS (Screen Space)
        if (
            self.mode == "predictions"
            and self.patient_meta
            and self.patient_meta.get("prediction") is None
        ):
            painter.setPen(QColor("#a29ab7"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(
                QRectF(0, h / 2 - 20, w, 40),
                Qt.AlignmentFlag.AlignCenter,
                "No prediction mask loaded",
            )
        self.draw_viewport_hud(painter, w, h)
        painter.end()

    def draw_viewport_hud(self, painter, w, h):
        """Draws top status badges and bottom clinical feature inspector HUD."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Top-Left Tag
        area_num = "1" if self.mode == "ground_truth" else "2"
        area_title = "GROUND TRUTH" if self.mode == "ground_truth" else "PREDICTIONS"

        # Pill Container
        hud_rect = QRectF(16, 14, 300, 24)
        painter.setBrush(QBrush(QColor(20, 22, 26, 210)))
        painter.setPen(QPen(QColor(50, 54, 64, 180), 1))
        painter.drawRoundedRect(hud_rect, 4, 4)

        # Text inside top tag
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QPointF(24, 30), f"({area_num}) {area_title}")

        # Sync Badge
        sync_x = 150
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor("#29b6f6"))
        painter.drawText(QPointF(sync_x, 30), "SYNCED")

        # Zoom level
        zoom_pct = int(self.zoom * 100)
        grid_status = " • PIXEL GRID" if (w / self.raster_w * self.zoom) >= 6.0 else ""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#ababaf"))
        painter.drawText(QPointF(sync_x + 72, 30), f"{zoom_pct}%{grid_status}")

        # Bottom Clinical Inspector HUD (Displays feature behind prediction)
        if (
            self.is_hovering
            and 0.0 <= self.hover_norm_pos.x() <= 1.0
            and 0.0 <= self.hover_norm_pos.y() <= 1.0
        ):
            hpx = int(self.hover_norm_pos.x() * self.raster_w)
            hpy = int(self.hover_norm_pos.y() * self.raster_h)

            if self.patient_meta:
                info = get_pixel_clinical_info(self.patient_meta, hpx, hpy)
                behind = info["behind_prediction"]
                gt_txt = info["gt"]
                pred_txt = info["pred"]

                bar_w = min(w - 32, 620)
                bar_rect = QRectF(16, h - 34, bar_w, 24)
                painter.setBrush(QBrush(QColor(18, 20, 24, 230)))
                painter.setPen(QPen(QColor(50, 54, 64, 200), 1))
                painter.drawRoundedRect(bar_rect, 4, 4)

                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
                painter.setPen(QColor("#29b6f6"))
                painter.drawText(QPointF(24, h - 18), f"Pixel [{hpx}, {hpy}]")

                painter.setPen(QColor("#dcdde0"))
                painter.drawText(QPointF(110, h - 18), f"Behind Prediction: {behind}")

                painter.setPen(QColor("#8d909b"))
                painter.drawText(
                    QPointF(bar_w - 180, h - 18), f"GT: {gt_txt} | Pred: {pred_txt}"
                )


# ---------------------------------------------------------------------------
# SPECTRUM DIAGNOSTICS GRAPH WIDGET
# ---------------------------------------------------------------------------
class SpectrumDiagnosticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(275)

        self.axis_label = "WAVELENGTH (nm)"
        self.pixel1_data = []
        self.pixel2_data = []
        self.pixel1_label = "Pixel 1: select a pixel"
        self.pixel2_label = "Pixel 2: select another pixel"

        # Mini zoom buttons on top right
        self.btn_zoom_in = QPushButton("+", self)
        self.btn_zoom_in.setFixedSize(22, 22)
        self.btn_zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_in.setStyleSheet("""
            QPushButton {
                background: #25272d;
                color: #ababaf;
                border: 1px solid #363942;
                border-radius: 3px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #343740;
                color: #ffffff;
            }
        """)

        self.btn_zoom_out = QPushButton("—", self)
        self.btn_zoom_out.setFixedSize(22, 22)
        self.btn_zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_out.setStyleSheet("""
            QPushButton {
                background: #25272d;
                color: #ababaf;
                border: 1px solid #363942;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #343740;
                color: #ffffff;
            }
        """)

    def set_spectrum_data(self, p1, p2):
        if p1:
            self.pixel1_data = p1
        if p2:
            self.pixel2_data = p2
        self.update()

    def update_selected_pixel_spectrum(self, info):
        """Dynamically updates graph curves when a user clicks on either canvas."""
        spec = info.get("spectrum", [])
        px = info.get("px", 0)
        py = info.get("py", 0)
        tissue = info.get("gt", "Unannotated")

        # Shift Pixel 1 to Pixel 2, and new selection to Pixel 1
        self.pixel2_data = self.pixel1_data
        self.pixel2_label = self.pixel1_label

        self.pixel1_data = spec
        self.pixel1_label = f"Pixel 1: [{px}, {py}] {tissue}"
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_zoom_in.move(self.width() - 32, 42)
        self.btn_zoom_out.move(self.width() - 32, 68)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw card background & border
        painter.setBrush(QBrush(QColor("#181a1e")))
        painter.setPen(QPen(QColor("#282a32"), 1))
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 6, 6)

        # Title: "SPECTRUM DIAGNOSTICS"
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#dcdde0"))
        painter.drawText(16, 22, "SPECTRUM DIAGNOSTICS")

        # Legend: Pixel 1 (blue) and Pixel 2 (orange)
        legend_x = max(180, self.width() - 310)

        # Pixel 1 line
        painter.setPen(QPen(QColor("#1e88e5"), 2.2))
        painter.drawLine(legend_x, 18, legend_x + 14, 18)
        painter.setPen(QColor("#ababaf"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.drawText(legend_x + 18, 22, self.pixel1_label)

        # Pixel 2 line
        painter.setPen(QPen(QColor("#f5a10c"), 2.2))
        painter.drawLine(legend_x, 32, legend_x + 14, 32)
        painter.setPen(QColor("#ababaf"))
        painter.drawText(legend_x + 18, 36, self.pixel2_label)

        # Plot viewport box
        plot_left = 68
        plot_right = self.width() - 44
        plot_top = 56
        plot_bottom = self.height() - 50
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        # Draw Rotated Y-Axis Label: "MEASURED INTENSITY"
        painter.save()
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#6f727c"))
        painter.translate(16, plot_bottom - 20)
        painter.rotate(-90)
        painter.drawText(0, 0, "MEASURED INTENSITY")
        painter.restore()

        # Draw X-Axis Label: "WAVELENGTH (nm)"
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#6f727c"))
        x_label_rect = QRect(plot_left, self.height() - 20, plot_w, 16)
        painter.drawText(x_label_rect, Qt.AlignmentFlag.AlignCenter, self.axis_label)

        # Fit axes to the measured values without normalizing the acquired spectra.
        points = [
            (float(wl), float(value))
            for wl, value in self.pixel1_data + self.pixel2_data
            if np.isfinite(wl) and np.isfinite(value)
        ]
        if points:
            min_wl, max_wl = min(p[0] for p in points), max(p[0] for p in points)
            min_value, max_value = min(p[1] for p in points), max(p[1] for p in points)
        else:
            min_wl, max_wl, min_value, max_value = 0, 1, 0, 1
        max_wl = max(max_wl, min_wl + 1)
        span = max(max_value - min_value, 1e-6)
        min_value, max_value = min_value - span * 0.05, max_value + span * 0.05
        painter.setFont(QFont("Segoe UI", 7))
        for value in np.linspace(min_value, max_value, 5):
            y_pos = int(
                plot_bottom - (value - min_value) / (max_value - min_value) * plot_h
            )
            painter.setPen(QPen(QColor("#242730"), 1))
            painter.drawLine(plot_left, y_pos, plot_right, y_pos)
            painter.setPen(QColor("#6f727c"))
            painter.drawText(plot_left - 44, y_pos + 4, f"{value:.2g}")
        for wl in np.linspace(min_wl, max_wl, 5):
            x_pos = int(plot_left + (wl - min_wl) / (max_wl - min_wl) * plot_w)
            painter.drawText(x_pos - 12, plot_bottom + 16, f"{wl:.0f}")

        def to_point(wl, intensity):
            px = plot_left + (wl - min_wl) / (max_wl - min_wl) * plot_w
            py = (
                plot_bottom - (intensity - min_value) / (max_value - min_value) * plot_h
            )
            return QPointF(float(px), float(py))

        # Draw Pixel 1 Curve (Blue)
        curve1 = [
            (x, y) for x, y in self.pixel1_data if np.isfinite(x) and np.isfinite(y)
        ]
        if curve1:
            path1 = QPainterPath()
            p0 = to_point(curve1[0][0], curve1[0][1])
            path1.moveTo(p0)
            for wl, inten in curve1[1:]:
                path1.lineTo(to_point(wl, inten))
            painter.setPen(QPen(QColor("#1e88e5"), 2.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path1)

        # Draw Pixel 2 Curve (Orange)
        curve2 = [
            (x, y) for x, y in self.pixel2_data if np.isfinite(x) and np.isfinite(y)
        ]
        if curve2:
            path2 = QPainterPath()
            p0 = to_point(curve2[0][0], curve2[0][1])
            path2.moveTo(p0)
            for wl, inten in curve2[1:]:
                path2.lineTo(to_point(wl, inten))
            painter.setPen(QPen(QColor("#f5a10c"), 2.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path2)

        painter.end()


# ---------------------------------------------------------------------------
# SIMULTANEOUS PREDICTIONS GRID DIALOG
# ---------------------------------------------------------------------------
class SimultaneousPredictionsDialog(QDialog):
    file_selected = pyqtSignal(str)

    def __init__(self, loaded_patients, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PDI - Simultaneous Predictions Grid")
        self.setMinimumSize(960, 640)
        self.setStyleSheet("background: #111113; color: #ffffff;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        top_bar = QHBoxLayout()
        header_title = QLabel("SIMULTANEOUS PREDICTIONS & GROUND TRUTH OVERVIEW")
        header_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffffff; letter-spacing: 0.5px;"
        )
        top_bar.addWidget(header_title)
        top_bar.addStretch()

        close_btn = QPushButton("✕ Close", self)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #25272d;
                color: #ababaf;
                border: 1px solid #363942;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #e81123;
                color: #ffffff;
            }
        """)
        close_btn.clicked.connect(self.close)
        top_bar.addWidget(close_btn)
        layout.addLayout(top_bar)

        # Scrollable area for cards
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        grid_container = QWidget()
        grid_container.setStyleSheet("background: transparent;")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(18)

        self.patients = loaded_patients
        self.page = 0
        self.card_grid = grid_layout
        scroll.setWidget(grid_container)
        layout.addWidget(scroll)
        controls = QHBoxLayout()
        previous = QPushButton("Previous patients")
        following = QPushButton("Next patients")
        self.page_label = QLabel()
        previous.clicked.connect(lambda: self.change_page(-1))
        following.clicked.connect(lambda: self.change_page(1))
        controls.addWidget(previous)
        controls.addWidget(self.page_label)
        controls.addWidget(following)
        layout.addLayout(controls)
        self.render_page()

    def change_page(self, delta):
        self.page = max(
            0, min(max(0, (len(self.patients) - 1) // 6), self.page + delta)
        )
        self.render_page()

    def render_page(self):
        while self.card_grid.count():
            item = self.card_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        start = self.page * 6
        for index, record in enumerate(self.patients[start : start + 6]):
            try:
                card = self.create_patient_card(record)
            except Exception as error:
                card = QLabel(f"{record['id']}: {error}")
                card.setWordWrap(True)
            self.card_grid.addWidget(card, index // 2, index % 2)
        self.page_label.setText(
            f"Patients {start+1 if self.patients else 0}-{min(start+6,len(self.patients))} of {len(self.patients)}"
        )

    def create_patient_card(self, p):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #181a1e;
                border: 1px solid #282a32;
                border-radius: 8px;
            }
        """)
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        pid_label = QLabel(f"📄 {p['id']}")
        pid_label.setStyleSheet("color: #29b6f6; font-size: 13px; font-weight: bold;")
        header.addWidget(pid_label)
        header.addStretch()

        inspect_btn = QPushButton("Inspect Active View")
        inspect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        inspect_btn.setStyleSheet("""
            QPushButton {
                background: #2856d8;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1e40af;
            }
        """)
        inspect_btn.clicked.connect(
            lambda checked, pid=p["id"]: self.on_inspect_clicked(pid)
        )
        header.addWidget(inspect_btn)
        card_layout.addLayout(header)

        # Canvases row: Ground Truth (1) vs Predictions (2)
        views_layout = QHBoxLayout()
        views_layout.setSpacing(10)

        # Synthesize rasters for card
        q_tissue, q_gt, q_pred, meta = synthesize_patient_rasters(p, 240, 160)

        # GT Box
        gt_box = QVBoxLayout()
        gt_title = QLabel("GROUND TRUTH")
        gt_title.setStyleSheet("color: #8d909b; font-size: 10px; font-weight: bold;")
        gt_canvas = GridCanvasWidget(mode="ground_truth")
        gt_canvas.setFixedHeight(120)
        gt_canvas.set_shapes(p.get("gt_shapes", []))
        gt_canvas.set_raster_data(q_tissue, q_gt, q_pred, meta)
        gt_canvas.toolbar.hide()
        gt_box.addWidget(gt_title)
        gt_box.addWidget(gt_canvas)
        views_layout.addLayout(gt_box)

        # Pred Box
        pred_box = QVBoxLayout()
        pred_title = QLabel("(2) U-NET PREDICTIONS")
        pred_title.setStyleSheet("color: #8d909b; font-size: 10px; font-weight: bold;")
        pred_canvas = GridCanvasWidget(mode="predictions")
        pred_canvas.setFixedHeight(120)
        pred_canvas.set_shapes(p.get("pred_shapes", []))
        pred_canvas.set_raster_data(q_tissue, q_gt, q_pred, meta)
        pred_canvas.toolbar.hide()
        pred_box.addWidget(pred_title)
        pred_box.addWidget(pred_canvas)
        views_layout.addLayout(pred_box)

        # Link card canvases together
        gt_canvas.connect_partner(pred_canvas)
        pred_canvas.connect_partner(gt_canvas)

        card_layout.addLayout(views_layout)

        # Metrics
        m = p.get("metrics", {})
        metrics_line = QLabel(
            f"Accuracy: {m.get('Accuracy','-')}   |   Sensitivity: {m.get('Sensitivity','-')}   |   F1: {m.get('F1-Score','-')}   |   AUC: {m.get('AUC','-')}"
        )
        metrics_line.setStyleSheet(
            "color: #ababaf; font-size: 11px; font-family: monospace;"
        )
        card_layout.addWidget(metrics_line)

        return frame

    def on_inspect_clicked(self, pid):
        self.file_selected.emit(pid)
        self.close()


# ---------------------------------------------------------------------------
# MAIN PDI WINDOW
# ---------------------------------------------------------------------------
class PatientImportWorker(QThread):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, paths, destination, parent=None):
        super().__init__(parent)
        self.paths, self.destination = paths, destination

    def run(self):
        for index, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                break
            self.progress.emit(index, Path(path).name)
            try:
                if Path(path).suffix.lower() == ".zip":
                    record = import_patient_zip(path, self.destination)
                elif Path(path).name.lower() == "cube.dat":
                    record = patient_record(Path(path).parent)
                else:
                    raise ValueError("Select cube.dat or Patient_<number>.zip")
                self.result.emit(record)
            except Exception as error:
                self.error.emit(f"{Path(path).name}: {error}")
        self.progress.emit(len(self.paths), "Complete")


class PDIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.resize(1536, 960)
        self.setMinimumSize(1100, 750)

        # Data state
        self.loaded_patients = {}
        self.patient_rasters_cache = {}
        self.active_patient_id = None
        self.load_initial_data()

        # Build UI
        self.init_ui()
        if self.load_errors:
            QMessageBox.warning(
                self, "Patient loading errors", "\n".join(self.load_errors)
            )

    def load_initial_data(self):
        """Load real patient folders from a stable, configurable data directory."""
        self.data_root = Path(os.environ.get("PDI_DATA_ROOT", r"D:\PDI\data")).resolve()
        self.loaded_patients = {}
        self.load_errors = []
        for cube in sorted(self.data_root.rglob("cube.dat")):
            try:
                record = patient_record(cube.parent)
                if record["id"] in self.loaded_patients:
                    raise ValueError(f"Duplicate patient ID: {record['id']}")
                self.loaded_patients[record["id"]] = record
            except Exception as error:
                self.load_errors.append(str(error))
        self.active_patient_id = next(iter(self.loaded_patients), None)

    def init_ui(self):
        main_widget = QWidget(self)
        main_widget.setStyleSheet(f"background: {BG_WINDOW.name()};")
        self.setCentralWidget(main_widget)

        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Custom Title Bar
        self.title_bar = CustomTitleBar(self)
        root_layout.addWidget(self.title_bar)

        # 2. Body Area
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # --- LEFT PANEL (VIEWPORTS: GROUND TRUTH & PREDICTIONS) ---
        left_widget = QWidget()
        left_widget.setStyleSheet(f"background: {BG_WINDOW.name()};")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(34, 36, 28, 30)
        left_layout.setSpacing(14)

        # (1) Ground Truth Header with Sync Indicator
        gt_header_layout = QHBoxLayout()
        gt_header_layout.setSpacing(8)

        gt_badge = QLabel("GROUND TRUTH")
        gt_badge.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;"
        )

        gt_sub = QLabel("[Reference patient annotations]")
        gt_sub.setStyleSheet("color: #666870; font-size: 12px; font-weight: 500;")

        self.gt_sync_badge = QLabel("SYNCED")
        self.gt_sync_badge.setStyleSheet("""
            background: #142838;
            color: #29b6f6;
            border: 1px solid #1976d2;
            border-radius: 3px;
            padding: 2px 7px;
            font-size: 10px;
            font-weight: bold;
        """)

        gt_header_layout.addWidget(gt_badge)
        gt_header_layout.addWidget(gt_sub)
        gt_header_layout.addWidget(self.gt_sync_badge)
        gt_header_layout.addStretch()
        left_layout.addLayout(gt_header_layout)

        # (1) Ground Truth Canvas
        self.gt_canvas = GridCanvasWidget(mode="ground_truth")
        left_layout.addWidget(self.gt_canvas, stretch=10)

        # Opacity Slider Bar
        opacity_layout = QHBoxLayout()
        opacity_layout.setContentsMargins(0, 2, 0, 2)
        opacity_layout.setSpacing(16)

        opacity_label = QLabel("ANNOTATIONS OPACITY")
        opacity_label.setStyleSheet(
            "color: #777982; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;"
        )
        opacity_layout.addWidget(opacity_label)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(65)
        self.opacity_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #333640;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1976d2, stop:1 #29b6f6);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 14px;
                height: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #e3f2fd;
            }
        """)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider, stretch=1)

        self.opacity_val_label = QLabel("65%")
        self.opacity_val_label.setFixedWidth(42)
        self.opacity_val_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.opacity_val_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: 600;"
        )
        opacity_layout.addWidget(self.opacity_val_label)

        left_layout.addLayout(opacity_layout)
        left_layout.addSpacing(24)

        # (2) Predictions Header with Sync Indicator
        pred_header_layout = QHBoxLayout()
        pred_header_layout.setSpacing(8)

        pred_badge = QLabel("PREDICTIONS")
        pred_badge.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;"
        )

        pred_sub = QLabel("[Patient prediction mask]")
        pred_sub.setStyleSheet("color: #666870; font-size: 12px; font-weight: 500;")

        self.pred_sync_badge = QLabel("SYNCED")
        self.pred_sync_badge.setStyleSheet("""
            background: #142838;
            color: #29b6f6;
            border: 1px solid #1976d2;
            border-radius: 3px;
            padding: 2px 7px;
            font-size: 10px;
            font-weight: bold;
        """)

        pred_header_layout.addWidget(pred_badge)
        pred_header_layout.addWidget(pred_sub)
        pred_header_layout.addWidget(self.pred_sync_badge)
        pred_header_layout.addStretch()
        left_layout.addLayout(pred_header_layout)

        # (2) Predictions Canvas
        self.pred_canvas = GridCanvasWidget(mode="predictions")
        left_layout.addWidget(self.pred_canvas, stretch=10)

        # Connect both viewports for continuous two-way simultaneous updates!
        self.gt_canvas.connect_partner(self.pred_canvas)
        self.pred_canvas.connect_partner(self.gt_canvas)

        body_layout.addWidget(left_widget, stretch=1)

        # --- RIGHT SIDEBAR ---
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setFixedWidth(568)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setStyleSheet(
            "QScrollArea { border: none; background: #25272d; }"
        )

        sidebar_content = QWidget()
        sidebar_content.setStyleSheet("background: #25272d;")
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(46, 8, 32, 24)
        sidebar_layout.setSpacing(5)

        # 1. DATA SELECTION HEADER
        data_sel_header = QLabel("DATA SELECTION")
        data_sel_header.setStyleSheet(
            "color: #ffffff; font-size: 12px; font-weight: 800; letter-spacing: 1px;"
        )
        sidebar_layout.addWidget(data_sel_header)

        # SELECT MODEL
        sidebar_layout.addWidget(self.create_field_label("SELECT MODEL"))
        self.model_input = self.create_path_input(
            "Optional model file", self.browse_model
        )
        sidebar_layout.addWidget(self.model_input)

        # SELECT CHECKPOINT
        sidebar_layout.addWidget(self.create_field_label("SELECT CHECKPOINT"))
        self.checkpoint_input = self.create_path_input(
            "Optional checkpoint (viewer does not run inference)",
            self.browse_checkpoint,
        )
        sidebar_layout.addWidget(self.checkpoint_input)

        # SELECT .DAT FILE
        dat_header_layout = QHBoxLayout()
        dat_header_layout.addWidget(
            self.create_field_label("IMPORT PATIENT ZIP / CUBE")
        )
        dat_header_layout.addStretch()

        self.btn_multi_grid = QPushButton("Compare patients")
        self.btn_multi_grid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_multi_grid.setStyleSheet("""
            QPushButton {
                background: #25272d;
                color: #29b6f6;
                border: 1px solid #363942;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #343740;
                color: #ffffff;
            }
        """)
        self.btn_multi_grid.clicked.connect(self.open_simultaneous_dialog)
        dat_header_layout.addWidget(self.btn_multi_grid)
        sidebar_layout.addLayout(dat_header_layout)

        self.dat_input = self.create_path_input(
            "Select patient ZIPs or cube.dat", self.browse_dat_files
        )
        sidebar_layout.addWidget(self.dat_input)

        # Multi-DAT Badges
        self.badge_scroll = QScrollArea()
        self.badge_scroll.setFixedHeight(38)
        self.badge_scroll.setWidgetResizable(True)
        self.badge_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.badge_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.badge_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self.badge_container = QWidget()
        self.badge_container.setStyleSheet("background: transparent;")
        self.badge_layout = QHBoxLayout(self.badge_container)
        self.badge_layout.setContentsMargins(0, 0, 0, 0)
        self.badge_layout.setSpacing(6)
        self.badge_scroll.setWidget(self.badge_container)
        sidebar_layout.addWidget(self.create_field_label("SELECT PATIENT"))
        sidebar_layout.addWidget(self.badge_scroll)
        sidebar_layout.addWidget(self.create_field_label("DISPLAY"))
        self.display_selector = QComboBox()
        self.display_selector.addItems(["Mask only", "RGB + mask", "RGB only"])
        self.display_selector.setStyleSheet(
            "QComboBox { background: #181a1e; color: white; padding: 5px; border: 1px solid #343740; border-radius: 4px; }"
        )
        self.display_selector.currentTextChanged.connect(self.on_display_changed)
        sidebar_layout.addWidget(self.display_selector)

        self.update_patient_badges()

        # SELECT MASK
        sidebar_layout.addWidget(self.create_field_label("PATIENT ANNOTATIONS (.MAT)"))
        self.mask_input = self.create_path_input(
            "Select MATLAB annotation map", self.browse_mask
        )
        sidebar_layout.addWidget(self.mask_input)
        prediction_row = QHBoxLayout()
        prediction_button = QPushButton("Load prediction mask (.npy / .mat)")
        prediction_button.setStyleSheet(
            "background: #181a1e; color: #bfc7dc; padding: 6px; border: 1px solid #343740; border-radius: 4px;"
        )
        prediction_button.clicked.connect(self.browse_prediction)
        prediction_row.addWidget(prediction_button)
        sidebar_layout.addLayout(prediction_row)

        # REGION LABEL SWATCHES
        sidebar_layout.addWidget(self.create_field_label("REGION LABEL SWATCHES"))
        swatches_layout = QHBoxLayout()
        swatches_layout.setSpacing(10)
        swatches_layout.setContentsMargins(2, 2, 2, 2)
        for hex_code, desc in SWATCH_COLORS:
            swatch_btn = QPushButton()
            swatch_btn.setFixedSize(18, 18)
            swatch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch_btn.setToolTip(f"{desc}\nColor: {hex_code}")
            swatch_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_code};
                    border: none;
                    border-radius: 9px;
                }}
                QPushButton:hover {{
                    border: 2px solid #ffffff;
                }}
            """)
            swatches_layout.addWidget(swatch_btn)
        swatches_layout.addStretch()
        sidebar_layout.addLayout(swatches_layout)

        # 2. CLASSIFIER METRICS
        sidebar_layout.addSpacing(4)
        metrics_header = QLabel("CLASSIFIER METRICS")
        metrics_header.setStyleSheet(
            "color: #ffffff; font-size: 12px; font-weight: 800; letter-spacing: 1px;"
        )
        sidebar_layout.addWidget(metrics_header)

        self.metrics_labels = {}
        metrics_frame = QWidget()
        metrics_frame.setObjectName("metricsPanel")
        metrics_frame.setStyleSheet(
            "QWidget#metricsPanel { background: #181a1e; border: 1px solid #343740; border-radius: 5px; } QLabel { border: none; background: transparent; }"
        )
        metrics_grid = QGridLayout(metrics_frame)
        metrics_grid.setContentsMargins(14, 10, 14, 10)
        metrics_grid.setVerticalSpacing(4)

        metric_keys = [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "F1-Score",
            "AUC",
            "MCC",
        ]
        for row_idx, key in enumerate(metric_keys):
            lbl = QLabel(key)
            lbl.setStyleSheet("color: #8d909b; font-size: 12px; font-weight: 500;")

            val_lbl = QLabel("N/A")
            val_lbl.setStyleSheet(
                "color: #ffffff; font-size: 12px; font-weight: 700; font-family: 'Consolas', monospace;"
            )
            val_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            metrics_grid.addWidget(lbl, row_idx, 0)
            metrics_grid.addWidget(val_lbl, row_idx, 1)
            self.metrics_labels[key] = val_lbl

        sidebar_layout.addWidget(metrics_frame)

        # 3. SPECTRUM DIAGNOSTICS CARD
        sidebar_layout.addSpacing(2)
        self.spectrum_widget = SpectrumDiagnosticsWidget()
        sidebar_layout.addWidget(self.spectrum_widget)

        # Connect pixel selection to spectrum diagnostics
        self.gt_canvas.pixel_selected_detailed.connect(
            self.spectrum_widget.update_selected_pixel_spectrum
        )
        self.pred_canvas.pixel_selected_detailed.connect(
            self.spectrum_widget.update_selected_pixel_spectrum
        )

        sidebar_scroll.setWidget(sidebar_content)
        body_layout.addWidget(sidebar_scroll)

        root_layout.addLayout(body_layout)

        # Apply active patient
        self.apply_patient_data(self.active_patient_id)

    # -----------------------------------------------------------------------
    # WIDGET FACTORY HELPERS
    # -----------------------------------------------------------------------
    def create_field_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #8d909b; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;"
        )
        return lbl

    def create_path_input(self, initial_text, browse_callback):
        container = QFrame()
        container.setFixedHeight(34)
        container.setStyleSheet("""
            QFrame {
                background: #25272d;
                border: 1px solid #32353e;
                border-radius: 6px;
            }
        """)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(6)

        line_edit = QLineEdit(initial_text)
        line_edit.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                color: #ffffff;
                font-size: 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        layout.addWidget(line_edit, stretch=1)

        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(24, 20)
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8d909b;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ffffff;
            }
        """)
        browse_btn.clicked.connect(browse_callback)
        layout.addWidget(browse_btn)

        container.line_edit = line_edit
        return container

    # -----------------------------------------------------------------------
    # PATIENT SWITCHING & SIMULTANEOUS FILE MANAGEMENT
    # -----------------------------------------------------------------------
    def update_patient_badges(self):
        """Provide a searchable selector for a large patient collection."""
        while self.badge_layout.count():
            item = self.badge_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        selector = QComboBox()
        selector.setEditable(True)
        selector.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        selector.addItems(sorted(self.loaded_patients))
        if self.active_patient_id:
            selector.setCurrentText(self.active_patient_id)
        selector.setMinimumWidth(180)
        self.patient_selector = selector
        selector.setStyleSheet("background: #25272d; color: white; padding: 5px;")
        selector.textActivated.connect(self.switch_patient)
        self.badge_layout.addWidget(selector)
        self.badge_layout.addWidget(QLabel(f"{len(self.loaded_patients)} patients"))
        self.badge_layout.addStretch()

    def switch_patient(self, patient_id):
        if patient_id in self.loaded_patients:
            self.active_patient_id = patient_id
            self.apply_patient_data(patient_id)
            self.update_patient_badges()

    def apply_patient_data(self, patient_id):
        p = self.loaded_patients.get(patient_id)
        if not p:
            return

        self.dat_input.line_edit.setText(p["cube"])
        self.dat_input.line_edit.setToolTip(p["cube"])
        self.mask_input.line_edit.setText(p["gt"])
        self.mask_input.line_edit.setToolTip(p["gt"])
        if "mask_id" in p:
            self.mask_input.line_edit.setText(f"data/{p['mask_id']}")

        # Generate or fetch cached rasters (tissue, gt, pred)
        if patient_id not in self.patient_rasters_cache:
            try:
                rasters = synthesize_patient_rasters(p, 320, 240)
            except Exception as error:
                QMessageBox.warning(self, "Unable to display patient", str(error))
                return
            # Keep only the active patient's native-resolution arrays in memory.
            self.patient_rasters_cache.clear()
            self.patient_rasters_cache[patient_id] = rasters

        q_tissue, q_gt, q_pred, meta = self.patient_rasters_cache[patient_id]

        # Update Canvases simultaneously
        self.gt_canvas.set_shapes(p.get("gt_shapes", []))
        self.gt_canvas.set_raster_data(q_tissue, q_gt, q_pred, meta)

        self.pred_canvas.set_shapes(p.get("pred_shapes", []))
        self.pred_canvas.set_raster_data(q_tissue, q_gt, q_pred, meta)

        # Update Metrics
        metrics = meta["metrics"]
        for key, val in metrics.items():
            if key in self.metrics_labels:
                self.metrics_labels[key].setText(str(val))

        # Update Spectrum
        spectrum = p.get("spectrum", {})
        info = pixel_info(meta, meta["width"] // 2, meta["height"] // 2)
        p1 = info["spectrum"]
        p2 = info["spectrum"]
        self.spectrum_widget.pixel1_label = (
            f"Pixel 1: [{info['px']}, {info['py']}] {info['gt']}"
        )
        self.spectrum_widget.pixel2_label = "Pixel 2: select another pixel to compare"
        self.spectrum_widget.axis_label = meta["spectral_axis_label"]
        self.spectrum_widget.set_spectrum_data(p1, p2)
        self.pred_sync_badge.setText(
            "MASK LOADED" if meta["prediction"] is not None else "NO PREDICTION"
        )
        self.on_display_changed(self.display_selector.currentText())

    def on_display_changed(self, mode):
        """Apply the same display mode to both patient views."""
        for canvas in (self.gt_canvas, self.pred_canvas):
            canvas.show_tissue = mode != "Mask only"
            canvas.show_overlay = mode != "RGB only"
            canvas.update()

    def on_opacity_changed(self, value):
        self.opacity_val_label.setText(f"{value}%")
        opacity_float = value / 100.0
        self.gt_canvas.set_opacity(opacity_float)
        self.pred_canvas.set_opacity(opacity_float)

    # -----------------------------------------------------------------------
    # FILE BROWSE HANDLERS
    # -----------------------------------------------------------------------
    def browse_dat_files(self):
        """Import multiple ZIPs or register extracted ENVI patient cubes."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select patient ZIP archives or ENVI cubes",
            str(self.data_root),
            "Patient datasets (*.zip *.dat);;All Files (*)",
        )
        if not paths:
            return
        self.imported_ids, self.import_errors = [], []
        self.import_progress = QProgressDialog(
            "Importing patients", "Stop after current patient", 0, len(paths), self
        )
        self.import_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.import_progress.setMinimumDuration(0)
        self.import_worker = PatientImportWorker(paths, self.data_root, self)
        self.import_worker.progress.connect(self.on_import_progress)
        self.import_worker.result.connect(self.on_patient_imported)
        self.import_worker.error.connect(self.import_errors.append)
        self.import_worker.finished.connect(self.on_import_finished)
        self.import_progress.canceled.connect(self.import_worker.requestInterruption)
        self.import_worker.start()

    def on_import_progress(self, index, name):
        self.import_progress.setLabelText(f"Importing: {name}")
        self.import_progress.setValue(index)

    def on_patient_imported(self, record):
        current = self.loaded_patients.get(record["id"])
        if current and current["cube"] != record["cube"]:
            self.import_errors.append(
                f"Duplicate patient at another location: {record['id']}"
            )
            return
        self.loaded_patients[record["id"]] = record
        self.patient_rasters_cache.pop(record["id"], None)
        self.imported_ids.append(record["id"])

    def on_import_finished(self):
        self.import_progress.close()
        if self.imported_ids:
            self.switch_patient(self.imported_ids[0])
        if self.import_errors:
            QMessageBox.warning(
                self, "Patient import errors", "\n".join(self.import_errors)
            )

    def closeEvent(self, event):
        if hasattr(self, "import_worker") and self.import_worker.isRunning():
            self.import_worker.requestInterruption()
            event.ignore()
            return
        super().closeEvent(event)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Directory / File", "models", "All Files (*)"
        )
        if path:
            self.model_input.line_edit.setText(path)

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "models", "PyTorch (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.checkpoint_input.line_edit.setText(path)

    def browse_prediction(self):
        """Attach an externally computed label map to the active patient."""
        if not self.active_patient_id:
            QMessageBox.information(
                self, "No patient selected", "Load a patient first."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select prediction with original label IDs",
            str(self.data_root),
            "Prediction labels (*.npy *.mat)",
        )
        if not path:
            return
        record = dict(self.loaded_patients[self.active_patient_id])
        record["prediction_path"] = path
        try:
            rasters = load_patient_rasters(record)
        except Exception as error:
            QMessageBox.warning(self, "Invalid prediction", str(error))
            return
        self.loaded_patients[self.active_patient_id] = record
        self.patient_rasters_cache.clear()
        self.patient_rasters_cache[self.active_patient_id] = rasters
        self.apply_patient_data(self.active_patient_id)

    def browse_mask(self):
        """Replace the active patient's annotation map after alignment validation."""
        if not self.active_patient_id:
            QMessageBox.information(
                self, "No patient selected", "Load a patient first."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MATLAB annotations", str(self.data_root), "MATLAB (*.mat)"
        )
        if not path:
            return
        record = dict(self.loaded_patients[self.active_patient_id])
        record["gt"] = path
        # Generate colors from the new map, rather than using an unrelated PNG.
        record["gt_png"] = None
        try:
            rasters = load_patient_rasters(record)
        except Exception as error:
            QMessageBox.warning(self, "Invalid annotation map", str(error))
            return
        self.loaded_patients[self.active_patient_id] = record
        self.patient_rasters_cache.clear()
        self.patient_rasters_cache[self.active_patient_id] = rasters
        self.apply_patient_data(self.active_patient_id)

    def open_simultaneous_dialog(self):
        dialog = SimultaneousPredictionsDialog(
            list(self.loaded_patients.values()), self
        )
        dialog.file_selected.connect(self.switch_patient)
        dialog.exec()


# ---------------------------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QToolTip {
            background-color: #25272d;
            color: #ffffff;
            border: 1px solid #363942;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
        QScrollBar:vertical {
            background: #181a1e;
            width: 8px;
            margin: 0px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #333640;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4f5360;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #181a1e;
            height: 6px;
            margin: 0px;
            border-radius: 3px;
        }
        QScrollBar::handle:horizontal {
            background: #333640;
            min-width: 20px;
            border-radius: 3px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #4f5360;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)

    window = PDIMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
