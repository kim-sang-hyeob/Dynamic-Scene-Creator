/**
 * PathEditorPanel — UI for the PathEditor.
 * Manages the "Path" tab in the right sidebar, replicating the original
 * web_viewer_final sidebar (mode buttons, point list, sliders, action bar).
 * Also handles tab switching between Layers and Path tabs.
 */

export class PathEditorPanel {
  constructor(pathEditor) {
    this.pathEditor = pathEditor;

    // Tab elements
    this.sidebarPanel = document.getElementById('sidebar-panel');
    this.tabButtons = this.sidebarPanel?.querySelectorAll('.sidebar-tab');
    this.tabLayers = document.getElementById('tab-layers');
    this.tabPath = document.getElementById('tab-path');

    // Path tab elements
    this.modeButtons = document.querySelectorAll('#pe-mode-buttons .pe-mode-btn');
    this.pointListEl = document.getElementById('pe-point-list');
    this.frameInfoEl = document.getElementById('pe-frame-info');

    this._bindTabs();
    this._bindModeButtons();
    this._bindSliders();
    this._bindActionBar();
    this._bindPathEvents();
    this._updatePointList();
    this._updateFrameInfo();
  }

  // ── Tab Switching ───────────────────────────────────────────────────

  _bindTabs() {
    if (!this.tabButtons) return;
    this.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        this._switchTab(tab);
      });
    });
  }

  _switchTab(tabName) {
    // Update tab button states
    this.tabButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Toggle content visibility
    if (this.tabLayers) this.tabLayers.classList.toggle('active', tabName === 'layers');
    if (this.tabPath) this.tabPath.classList.toggle('active', tabName === 'path');

    // Activate/deactivate path editor
    this.pathEditor.active = (tabName === 'path');
  }

  // ── Mode Buttons (2x2 grid) ────────────────────────────────────────

  _bindModeButtons() {
    this.modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        this.pathEditor.setMode(btn.dataset.mode);
      });
    });
  }

  _updateModeButtons() {
    this.modeButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === this.pathEditor.mode);
    });
  }

  // ── Sliders ─────────────────────────────────────────────────────────

  _bindSliders() {
    // Duration
    this._bindSlider('pe-duration-slider', 'pe-duration-val', (v) => {
      this.pathEditor.duration = v;
      this._updateFrameInfo();
      return v.toFixed(1) + 's';
    });

    // Dome Radius
    this._bindSlider('pe-cam-dist-slider', 'pe-cam-dist-val', (v) => {
      this.pathEditor.camDistance = v;
      this.pathEditor.generateCameraFrustums();
      return v.toFixed(1);
    });

    // Azimuth
    this._bindSlider('pe-cam-az-slider', 'pe-cam-az-val', (v) => {
      this.pathEditor.camAzimuth = v;
      this.pathEditor.generateCameraFrustums();
      return v + '\u00B0';
    });

    // Elevation
    this._bindSlider('pe-cam-el-slider', 'pe-cam-el-val', (v) => {
      this.pathEditor.camElevation = v;
      this.pathEditor.generateCameraFrustums();
      return v + '\u00B0';
    });

    // Export FPS
    this._bindSlider('pe-export-fps-slider', 'pe-export-fps-val', (v) => {
      this.pathEditor.exportFps = Math.round(v);
      this._updateFrameInfo();
      return String(Math.round(v));
    });

    // Show cameras checkbox
    document.getElementById('pe-show-cams')?.addEventListener('change', (e) => {
      this.pathEditor.showCameras = e.target.checked;
      this.pathEditor.generateCameraFrustums();
    });
  }

  _bindSlider(sliderId, valId, onChange) {
    const slider = document.getElementById(sliderId);
    const valEl = document.getElementById(valId);
    if (!slider) return;
    slider.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      const display = onChange(v);
      if (valEl) valEl.textContent = display;
    });
  }

  // ── Action Bar ──────────────────────────────────────────────────────

  _bindActionBar() {
    document.getElementById('pe-btn-save')?.addEventListener('click', () => {
      this.pathEditor.savePath();
    });

    document.getElementById('pe-btn-load')?.addEventListener('click', () => {
      this._loadFile();
    });

    document.getElementById('pe-btn-play')?.addEventListener('click', () => {
      this.pathEditor.toggleAnimation();
    });

    document.getElementById('pe-btn-record')?.addEventListener('click', () => {
      this.pathEditor.toggleRecording();
    });

    document.getElementById('pe-btn-clear')?.addEventListener('click', () => {
      this.pathEditor.clearAll();
    });

    document.getElementById('pe-export-btn')?.addEventListener('click', () => {
      this.pathEditor.exportDataset();
    });
  }

  _loadFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = () => {
      if (input.files[0]) this.pathEditor.loadPath(input.files[0]);
    };
    input.click();
  }

  // ── PathEditor Event Listener ───────────────────────────────────────

  _bindPathEvents() {
    this.pathEditor.onChange = (eventType) => {
      this._updateModeButtons();
      this._updatePointList();
      this._updatePlayButton();
      this._updateRecordButton();
      this._syncSliders();
      this._updateFrameInfo();
      this._showEventToast(eventType);
    };
  }

  // ── Toast Messages ──────────────────────────────────────────────────

  _showEventToast(eventType) {
    const messages = {
      'place': `Point added (#${this.pathEditor.controlPoints.length})`,
      'delete': 'Point removed',
      'clear': 'All points cleared',
      'undo': 'Reverted change',
      'save': 'Path saved',
      'load': `Path loaded (${this.pathEditor.controlPoints.length} points)`,
      'load-error': ['Failed to load path file', 'error'],
      'export': 'Dataset exported (transforms_train.json)',
      'animation-start': 'Playing...',
      'animation-stop': 'Stopped',
      'recording-start': 'Recording started...',
      'recording-stop': 'Recording saved',
    };

    const entry = messages[eventType];
    if (!entry) return;

    if (Array.isArray(entry)) {
      this._toast(entry[0], entry[1]);
    } else {
      this._toast(entry);
    }
  }

  _toast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateX(-50%) translateY(0)';
    });
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(-50%) translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 2500);
  }

  // ── Point List ──────────────────────────────────────────────────────

  _updatePointList() {
    if (!this.pointListEl) return;
    const points = this.pathEditor.controlPoints;
    const selectedIdx = this.pathEditor.selectedIdx;

    this.pointListEl.innerHTML = '';

    if (points.length === 0) {
      this.pointListEl.innerHTML = '<div class="pe-point-empty">No control points yet.<br>Use PLACE mode to add.</div>';
      return;
    }

    points.forEach((pt, i) => {
      const div = document.createElement('div');
      div.className = 'pe-point-item' + (i === selectedIdx ? ' selected' : '');
      div.innerHTML = `
        <span class="pe-point-idx">${i + 1}</span>
        <span class="pe-point-coords">(${pt.position[0].toFixed(1)}, ${pt.position[1].toFixed(1)}, ${pt.position[2].toFixed(1)})</span>
        <button class="pe-point-del" title="Delete">&times;</button>
      `;
      div.addEventListener('click', (e) => {
        if (e.target.closest('.pe-point-del')) {
          this.pathEditor.selectedIdx = i;
          this.pathEditor.deleteSelected();
        } else {
          this.pathEditor.selectedIdx = i;
          this.pathEditor._updateOverlay();
          this.pathEditor._notify('select');
        }
      });
      this.pointListEl.appendChild(div);
    });
  }

  // ── Frame Info ──────────────────────────────────────────────────────

  _updateFrameInfo() {
    if (!this.frameInfoEl) return;
    const totalFrames = Math.round(this.pathEditor.duration * this.pathEditor.exportFps);
    this.frameInfoEl.textContent = `Frames: ${totalFrames} (${this.pathEditor.duration.toFixed(1)}s × ${this.pathEditor.exportFps}fps)`;
  }

  // ── Play / Record Button State ──────────────────────────────────────

  _updatePlayButton() {
    const btn = document.getElementById('pe-btn-play');
    if (!btn) return;
    btn.innerHTML = this.pathEditor.animating ? '&#9632;' : '&#9654;';
    btn.classList.toggle('pe-playing', this.pathEditor.animating);
  }

  _updateRecordButton() {
    const btn = document.getElementById('pe-btn-record');
    if (!btn) return;
    btn.classList.toggle('recording', this.pathEditor.recording);
  }

  // ── Sync Sliders ────────────────────────────────────────────────────

  _syncSliders() {
    const pe = this.pathEditor;
    this._setSlider('pe-duration-slider', 'pe-duration-val', pe.duration, pe.duration.toFixed(1) + 's');
    this._setSlider('pe-cam-dist-slider', 'pe-cam-dist-val', pe.camDistance, pe.camDistance.toFixed(1));
    this._setSlider('pe-cam-az-slider', 'pe-cam-az-val', pe.camAzimuth, pe.camAzimuth + '\u00B0');
    this._setSlider('pe-cam-el-slider', 'pe-cam-el-val', pe.camElevation, pe.camElevation + '\u00B0');
    this._setSlider('pe-export-fps-slider', 'pe-export-fps-val', pe.exportFps, String(pe.exportFps));
  }

  _setSlider(sliderId, valId, value, display) {
    const slider = document.getElementById(sliderId);
    const valEl = document.getElementById(valId);
    if (slider && document.activeElement !== slider) slider.value = value;
    if (valEl) valEl.textContent = display;
  }
}
