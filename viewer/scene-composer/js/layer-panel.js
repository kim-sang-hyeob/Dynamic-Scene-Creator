/**
 * Layer Panel — Right sidebar UI for layer management.
 * Shows Map section (top) and Objects section separately.
 * Shows transform inputs for the selected layer.
 */

import { exportSceneToSplatv, exportSceneToSplat, downloadBlob } from './utils/splat-exporter.js';

const LAYER_COLORS = ['#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#f97316'];

export class LayerPanel {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
    this.panelEl = document.getElementById('sidebar-panel');
    this.listEl = document.getElementById('layer-list');
    this.transformEl = document.getElementById('transform-section');
    this.onAddLayerClick = null;

    // External refs (set by main.js)
    this.controls = null;
    this.pathEditor = null;

    this._bindEvents();
    this.render();
  }

  _bindEvents() {
    const addBtn = this.panelEl.querySelector('.add-layer-btn');
    if (addBtn) {
      addBtn.addEventListener('click', () => {
        if (this.onAddLayerClick) this.onAddLayerClick();
      });
    }

    // Export button
    const exportBtn = this.panelEl.querySelector('.export-scene-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this._handleExport());
    }

    const origOnChange = this.sceneManager.onChange;
    this.sceneManager.onChange = (event, data) => {
      if (origOnChange) origOnChange(event, data);
      this.render();
    };
  }

  /**
   * Handle export button click — show format selection dialog.
   */
  _handleExport() {
    const layers = this.sceneManager.getLayers();
    if (layers.length === 0) {
      this._showToast('No layers to export', 'error');
      return;
    }

    // Show export dialog
    this._showExportDialog();
  }

  /**
   * Show export format selection dialog.
   */
  _showExportDialog() {
    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'export-modal-overlay';
    overlay.innerHTML = `
      <div class="export-modal glass">
        <div class="export-modal-header">
          <h3>Export Scene</h3>
          <button class="export-modal-close">&times;</button>
        </div>
        <div class="export-modal-body">
          <div class="export-info">
            <strong>${this.sceneManager.totalGaussians.toLocaleString()}</strong> gaussians from
            <strong>${this.sceneManager.layers.filter(l => l.visible).length}</strong> layers
          </div>
          <div class="export-options">
            <button class="export-option-btn" data-format="splatv">
              <span class="export-option-icon">&#128230;</span>
              <span class="export-option-title">.splatv (Recommended)</span>
              <span class="export-option-desc">Full metadata: layers, transforms, camera path</span>
            </button>
            <button class="export-option-btn" data-format="splat">
              <span class="export-option-icon">&#128196;</span>
              <span class="export-option-title">.splat (Standard)</span>
              <span class="export-option-desc">Compatible with other viewers, no metadata</span>
            </button>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Event handlers
    const closeModal = () => overlay.remove();

    overlay.querySelector('.export-modal-close').onclick = closeModal;
    overlay.onclick = (e) => {
      if (e.target === overlay) closeModal();
    };

    overlay.querySelectorAll('.export-option-btn').forEach(btn => {
      btn.onclick = () => {
        const format = btn.dataset.format;
        closeModal();
        this._doExport(format);
      };
    });
  }

  /**
   * Perform the actual export.
   */
  _doExport(format) {
    try {
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
      let blob, filename;

      if (format === 'splatv') {
        blob = exportSceneToSplatv(this.sceneManager, this.controls, this.pathEditor);
        filename = `scene_${timestamp}.splatv`;
      } else {
        blob = exportSceneToSplat(this.sceneManager);
        filename = `scene_${timestamp}.splat`;
      }

      downloadBlob(blob, filename);
      this._showToast(`Exported ${filename} (${(blob.size / 1024 / 1024).toFixed(1)} MB)`);
    } catch (err) {
      console.error('Export failed:', err);
      this._showToast('Export failed: ' + err.message, 'error');
    }
  }

  /**
   * Show toast notification.
   */
  _showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  render() {
    this._renderList();
    this._renderTransform();
  }

  _renderList() {
    const layers = this.sceneManager.getLayers();
    const selectedId = this.sceneManager.selectedLayerId;
    const mapLayers = layers.filter(l => l.type === 'map');
    const objLayers = layers.filter(l => l.type !== 'map');

    this.listEl.innerHTML = '';

    if (layers.length === 0) {
      this.listEl.innerHTML = '<div style="text-align:center;padding:24px 0;color:var(--text-muted);font-size:12px;">No layers yet<br>Drop files to add</div>';
      return;
    }

    // Map section
    if (mapLayers.length > 0) {
      const header = document.createElement('div');
      header.className = 'section-header';
      header.textContent = 'Map';
      this.listEl.appendChild(header);

      for (const layer of mapLayers) {
        this.listEl.appendChild(this._createLayerItem(layer, selectedId, '#6b7280'));
      }
    }

    // Objects section
    if (objLayers.length > 0 || mapLayers.length > 0) {
      const header = document.createElement('div');
      header.className = 'section-header';
      header.textContent = 'Objects';
      this.listEl.appendChild(header);
    }

    if (objLayers.length === 0 && mapLayers.length > 0) {
      const hint = document.createElement('div');
      hint.style.cssText = 'text-align:center;padding:12px 0;color:var(--text-muted);font-size:11px;';
      hint.textContent = 'Drop files or use prompt bar';
      this.listEl.appendChild(hint);
    }

    objLayers.forEach((layer, idx) => {
      const color = LAYER_COLORS[idx % LAYER_COLORS.length];
      this.listEl.appendChild(this._createLayerItem(layer, selectedId, color));
    });
  }

  _createLayerItem(layer, selectedId, color) {
    const el = document.createElement('div');
    el.className = 'layer-item' + (layer.id === selectedId ? ' selected' : '');
    el.dataset.layerId = layer.id;

    const icon = layer.type === 'map' ? '&#x1f5fa;' : '&#x1f4e6;';
    const canDelete = layer.type !== 'map';

    el.innerHTML = `
      <div class="layer-color" style="background:${color};opacity:${layer.visible ? 1 : 0.3}"></div>
      <div class="layer-info">
        <div class="layer-name">${icon} ${this._escapeHtml(layer.name)}</div>
        <div class="layer-stats">${layer.count.toLocaleString()} gs</div>
      </div>
      <div class="layer-actions">
        <button class="layer-btn ${layer.visible ? 'active' : ''}" data-action="visibility" title="Toggle visibility">
          ${layer.visible ? '&#128065;' : '&#128064;'}
        </button>
        ${layer.type !== 'map' ? `
          <button class="layer-btn ${layer.locked ? 'active' : ''}" data-action="lock" title="Toggle lock">
            ${layer.locked ? '&#128274;' : '&#128275;'}
          </button>
          <button class="layer-btn danger" data-action="delete" title="Delete layer">&#10005;</button>
        ` : ''}
      </div>
    `;

    // Click to select (objects only)
    el.addEventListener('click', (e) => {
      if (e.target.closest('.layer-btn')) return;
      if (layer.type !== 'map') {
        this.sceneManager.selectLayer(layer.id);
      }
    });

    // Action buttons
    el.querySelectorAll('.layer-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const action = btn.dataset.action;
        if (action === 'visibility') {
          this.sceneManager.setVisible(layer.id, !layer.visible);
        } else if (action === 'lock') {
          this.sceneManager.setLocked(layer.id, !layer.locked);
        } else if (action === 'delete') {
          this.sceneManager.removeLayer(layer.id);
        }
      });
    });

    return el;
  }

  _renderTransform() {
    const layer = this.sceneManager.getSelectedLayer();

    if (!layer || !this.transformEl) {
      if (this.transformEl) this.transformEl.classList.add('hidden');
      return;
    }

    this.transformEl.classList.remove('hidden');

    this._setInputValue('pos-x', layer.position[0]);
    this._setInputValue('pos-y', layer.position[1]);
    this._setInputValue('pos-z', layer.position[2]);
    this._setInputValue('rot-x', layer.rotation[0]);
    this._setInputValue('rot-y', layer.rotation[1]);
    this._setInputValue('rot-z', layer.rotation[2]);
    this._setInputValue('scale', layer.scale);
  }

  _setInputValue(id, value) {
    const el = document.getElementById(id);
    if (el && document.activeElement !== el) {
      el.value = typeof value === 'number' ? value.toFixed(2) : value;
    }
  }

  bindTransformInputs() {
    const ids = ['pos-x', 'pos-y', 'pos-z', 'rot-x', 'rot-y', 'rot-z', 'scale'];

    ids.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;

      const apply = () => {
        const layer = this.sceneManager.getSelectedLayer();
        if (!layer) return;

        const val = parseFloat(el.value);
        if (isNaN(val)) return;

        if (id.startsWith('pos-')) {
          const axis = { 'pos-x': 0, 'pos-y': 1, 'pos-z': 2 }[id];
          const pos = [...layer.position];
          pos[axis] = val;
          this.sceneManager.setPosition(layer.id, pos);
        } else if (id.startsWith('rot-')) {
          const axis = { 'rot-x': 0, 'rot-y': 1, 'rot-z': 2 }[id];
          const rot = [...layer.rotation];
          rot[axis] = val;
          this.sceneManager.setRotation(layer.id, rot);
        } else if (id === 'scale') {
          this.sceneManager.setScale(layer.id, val);
        }
      };

      el.addEventListener('change', apply);
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { apply(); el.blur(); }
      });
    });
  }

  _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
