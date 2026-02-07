/**
 * Layer Panel — Right sidebar UI for layer management.
 * Shows Map section (top) and Objects section separately.
 * Shows transform inputs for the selected layer.
 */

const LAYER_COLORS = ['#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#f97316'];

export class LayerPanel {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
    this.panelEl = document.getElementById('sidebar-panel');
    this.listEl = document.getElementById('layer-list');
    this.transformEl = document.getElementById('transform-section');
    this.onAddLayerClick = null;

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

    const origOnChange = this.sceneManager.onChange;
    this.sceneManager.onChange = (event, data) => {
      if (origOnChange) origOnChange(event, data);
      this.render();
    };
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
