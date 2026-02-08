/**
 * World Labs Modal — Full-screen modal for text/image → 3D Map generation
 * via the World Labs Marble API (proxied through /api/generate-map).
 */

import { parseSplatBuffer } from './utils/splat-loader.js';

export class WorldLabsModal {
  constructor(sceneManager, controls) {
    this.sceneManager = sceneManager;
    this.controls = controls;
    this.serverUrl = '/api';
    this.isGenerating = false;
    this.currentImage = null;

    // DOM refs
    this.overlay = null;
    this.card = null;
    this.textInput = null;
    this.generateBtn = null;
    this.closeBtn = null;
    this.imageDropzone = null;
    this.imageInput = null;
    this.seedInput = null;
    this.statusText = null;
    this.progressBar = null;

    this.initUI();
  }

  initUI() {
    // Create overlay
    this.overlay = document.getElementById('worldlabs-modal');
    if (!this.overlay) return;

    this.overlay.innerHTML = `
      <div class="wl-card glass">
        <div class="wl-header">
          <h3>Generate 3D Map</h3>
          <button class="wl-close">&times;</button>
        </div>

        <div class="wl-body">
          <!-- Text prompt -->
          <div class="wl-field">
            <label class="wl-label">Prompt</label>
            <input type="text" class="wl-text-input" placeholder="Describe a scene... e.g. a cozy cafe interior" />
          </div>

          <!-- Image upload -->
          <div class="wl-field">
            <label class="wl-label">Reference Image <span class="wl-optional">(optional)</span></label>
            <div class="wl-dropzone">
              <input type="file" class="wl-file-input" accept="image/*" hidden />
              <div class="wl-dropzone-content">
                <span class="wl-dropzone-icon">&#x1f5bc;</span>
                <span class="wl-dropzone-text">Drop image here or click to upload</span>
              </div>
              <div class="wl-dropzone-preview" style="display:none">
                <img class="wl-preview-img" />
                <button class="wl-preview-remove">&times;</button>
              </div>
            </div>
          </div>

          <!-- Seed -->
          <div class="wl-field wl-field-inline">
            <label class="wl-label">Seed</label>
            <input type="number" class="wl-seed-input" value="-1" />
            <span class="wl-hint">-1 = random</span>
          </div>
        </div>

        <!-- Progress / Status -->
        <div class="wl-status" style="display:none">
          <div class="wl-progress-bar"><div class="wl-progress-fill"></div></div>
          <span class="wl-status-text">Generating...</span>
        </div>

        <div class="wl-footer">
          <button class="wl-cancel-btn">Cancel</button>
          <button class="wl-generate-btn">Generate Map</button>
        </div>
      </div>
    `;

    // Refs
    this.card = this.overlay.querySelector('.wl-card');
    this.closeBtn = this.overlay.querySelector('.wl-close');
    this.textInput = this.overlay.querySelector('.wl-text-input');
    this.imageDropzone = this.overlay.querySelector('.wl-dropzone');
    this.imageInput = this.overlay.querySelector('.wl-file-input');
    this.seedInput = this.overlay.querySelector('.wl-seed-input');
    this.generateBtn = this.overlay.querySelector('.wl-generate-btn');
    this.statusText = this.overlay.querySelector('.wl-status-text');
    this.progressBar = this.overlay.querySelector('.wl-status');

    const cancelBtn = this.overlay.querySelector('.wl-cancel-btn');

    // Events
    this.closeBtn.onclick = () => this.close();
    cancelBtn.onclick = () => this.close();
    this.overlay.onclick = (e) => { if (e.target === this.overlay) this.close(); };
    this.generateBtn.onclick = () => this.generate();
    this.textInput.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.generate();
      }
    };

    // Prevent gizmo shortcuts when typing
    this.textInput.onfocus = () => { window._promptBarFocused = true; };
    this.textInput.onblur = () => { window._promptBarFocused = false; };

    // Image upload
    this.imageDropzone.querySelector('.wl-dropzone-content').onclick = () => this.imageInput.click();
    this.imageInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) this._setImage(file);
    };
    this.overlay.querySelector('.wl-preview-remove').onclick = () => this._clearImage();

    // Drag-and-drop on dropzone
    this.imageDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.imageDropzone.classList.add('drag-over');
    });
    this.imageDropzone.addEventListener('dragleave', () => {
      this.imageDropzone.classList.remove('drag-over');
    });
    this.imageDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.imageDropzone.classList.remove('drag-over');
      const file = Array.from(e.dataTransfer.files).find(f => f.type.startsWith('image/'));
      if (file) this._setImage(file);
    });
  }

  // ── Image handling ────────────────────────────────────────────────

  _setImage(file) {
    this.currentImage = file;
    const url = URL.createObjectURL(file);
    const preview = this.imageDropzone.querySelector('.wl-dropzone-preview');
    const content = this.imageDropzone.querySelector('.wl-dropzone-content');
    preview.querySelector('.wl-preview-img').src = url;
    preview.style.display = 'flex';
    content.style.display = 'none';
  }

  _clearImage() {
    if (this.currentImage) {
      URL.revokeObjectURL(this.imageDropzone.querySelector('.wl-preview-img').src);
    }
    this.currentImage = null;
    const preview = this.imageDropzone.querySelector('.wl-dropzone-preview');
    const content = this.imageDropzone.querySelector('.wl-dropzone-content');
    preview.style.display = 'none';
    content.style.display = '';
    this.imageInput.value = '';
  }

  // ── Open / Close ──────────────────────────────────────────────────

  open() {
    if (this.sceneManager.hasMap()) {
      this._showToast('Map already exists. Remove it first to generate a new one.', 'error');
      return;
    }
    // Hide drop zone while modal is open
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) dropZone.classList.remove('empty');

    this.overlay.style.display = 'flex';
    this.textInput.focus();
  }

  close() {
    if (this.isGenerating) return;
    this.overlay.style.display = 'none';

    // Restore drop zone if no layers exist
    if (this.sceneManager.layers.length === 0) {
      const dropZone = document.getElementById('drop-zone');
      if (dropZone) dropZone.classList.add('empty');
    }
  }

  // ── Generation ────────────────────────────────────────────────────

  async generate() {
    if (this.isGenerating) return;

    const text = this.textInput.value.trim();
    const hasImage = !!this.currentImage;

    if (!text && !hasImage) {
      this._showToast('Enter a text prompt or upload an image', 'error');
      return;
    }

    this.isGenerating = true;
    this._setLoading(true);

    try {
      const body = {
        prompt_type: hasImage ? 'image' : 'text',
        seed: parseInt(this.seedInput.value) || -1,
      };

      if (hasImage) {
        body.prompt_image = await this._fileToBase64(this.currentImage);
      }
      if (text) {
        body.prompt_text = text;
      }

      const response = await fetch(`${this.serverUrl}/generate-map`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const result = await response.json();

      if (result.status === 'success') {
        const name = text ? text.slice(0, 24) : 'World Labs Map';
        await this._addGeneratedMap(result, name);
        this._showToast(`Map generated: ${result.gaussian_count.toLocaleString()} gaussians (${result.generation_time.toFixed(1)}s)`);
        this.close();
        this.textInput.value = '';
        this._clearImage();
      } else {
        this._showToast(result.error || 'Generation failed', 'error');
      }
    } catch (error) {
      this._showToast(`Connection failed: ${error.message}`, 'error');
    } finally {
      this.isGenerating = false;
      this._setLoading(false);
    }
  }

  async _addGeneratedMap(result, name) {
    // base64 → ArrayBuffer
    const binary = atob(result.splat_data);
    const buffer = new ArrayBuffer(binary.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);

    // Parse .splat → texdata/positions
    const parsed = parseSplatBuffer(buffer);

    // Add as map layer
    const layerId = this.sceneManager.addLayer(
      name, parsed.texdata, parsed.positions, parsed.count,
      parsed.texwidth, parsed.texheight, 'map'
    );

    // Hide drop zone
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) dropZone.classList.remove('empty');

    this.sceneManager.selectLayer(layerId);
  }

  // ── UI helpers ────────────────────────────────────────────────────

  _setLoading(loading) {
    this.generateBtn.disabled = loading;
    this.textInput.disabled = loading;
    this.progressBar.style.display = loading ? 'flex' : 'none';

    if (loading) {
      this.generateBtn.innerHTML = '<span class="wl-btn-spinner"></span>Generating...';
      this._startTimer();
    } else {
      this.generateBtn.textContent = 'Generate Map';
      this._stopTimer();
    }
  }

  _startTimer() {
    this._timerStart = Date.now();
    this._timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - this._timerStart) / 1000).toFixed(0);
      this.statusText.textContent = `Generating... (${elapsed}s)`;
    }, 1000);
  }

  _stopTimer() {
    if (this._timerInterval) {
      clearInterval(this._timerInterval);
      this._timerInterval = null;
    }
  }

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

  _fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
}
