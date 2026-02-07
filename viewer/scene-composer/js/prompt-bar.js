/**
 * Prompt Bar — Bottom-center UI for image/text → 3D generation via TRELLIS API.
 * Handles image upload, text input, settings popup, loading state, and toast notifications.
 */

import { parseSplatBuffer, parsePlyBuffer, detectFormat } from './utils/splat-loader.js';
import { invert4 } from './utils/matrix-math.js';

export class PromptBar {
  constructor(sceneManager, controls, config = {}) {
    this.sceneManager = sceneManager;
    this.controls = controls;
    // Connect to TRELLIS server directly (same host, port 8000)
    this.serverUrl = config.trellisServerUrl || `http://${window.location.hostname}:8000`;
    this.isGenerating = false;
    this.currentImage = null;

    this.settings = {
      sparseSteps: 12,
      slatSteps: 12,
      seed: -1,
    };

    // DOM refs (set in initUI)
    this.container = null;
    this.textInput = null;
    this.createBtn = null;
    this.cameraBtn = null;
    this.imageInput = null;
    this.settingsBtn = null;
    this.progressBar = null;
    this.previewArea = null;
    this.settingsPopup = null;

    this.initUI();
  }

  initUI() {
    this.container = document.getElementById('prompt-bar');
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="image-preview" style="display:none">
        <img class="preview-thumb" />
        <span class="preview-name"></span>
        <button class="preview-close" title="Remove image">&times;</button>
      </div>
      <div class="prompt-main">
        <button class="prompt-btn camera-btn" title="Upload image">&#x1f4f7;</button>
        <input type="file" class="image-input" accept="image/*" hidden />
        <input type="text" class="prompt-input" placeholder="Imagine an object..." />
        <button class="prompt-btn settings-btn" title="Settings">&#x2699;&#xfe0f;</button>
        <button class="prompt-btn create-btn" title="Generate 3D">&#x2728; Create</button>
      </div>
      <div class="generation-progress" style="display:none">
        <div class="bar"></div>
      </div>
      <div class="settings-popup" style="display:none">
        <label>Structure Steps: <input type="range" class="set-sparse" min="4" max="24" value="12" /><span class="set-sparse-val">12</span></label>
        <label>Detail Steps: <input type="range" class="set-slat" min="4" max="24" value="12" /><span class="set-slat-val">12</span></label>
        <label>Seed: <input type="number" class="set-seed" value="-1" /> <small>(-1 = random)</small></label>
        <label>Server: <input type="text" class="set-server" value="${this.serverUrl}" /></label>
      </div>
    `;

    // Grab refs
    this.textInput = this.container.querySelector('.prompt-input');
    this.createBtn = this.container.querySelector('.create-btn');
    this.cameraBtn = this.container.querySelector('.camera-btn');
    this.imageInput = this.container.querySelector('.image-input');
    this.settingsBtn = this.container.querySelector('.settings-btn');
    this.progressBar = this.container.querySelector('.generation-progress');
    this.previewArea = this.container.querySelector('.image-preview');
    this.settingsPopup = this.container.querySelector('.settings-popup');

    // Events
    this.cameraBtn.onclick = () => this.imageInput.click();

    this.imageInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) this.setImage(file);
    };

    this.textInput.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.generate();
      }
    };

    this.createBtn.onclick = () => this.generate();
    this.settingsBtn.onclick = () => this.toggleSettings();

    this.container.querySelector('.preview-close').onclick = () => this.clearImage();

    // Settings inputs
    const sparseRange = this.container.querySelector('.set-sparse');
    sparseRange.oninput = () => {
      this.settings.sparseSteps = parseInt(sparseRange.value);
      this.container.querySelector('.set-sparse-val').textContent = sparseRange.value;
    };
    const slatRange = this.container.querySelector('.set-slat');
    slatRange.oninput = () => {
      this.settings.slatSteps = parseInt(slatRange.value);
      this.container.querySelector('.set-slat-val').textContent = slatRange.value;
    };
    this.container.querySelector('.set-seed').onchange = (e) => {
      this.settings.seed = parseInt(e.target.value) || -1;
    };
    this.container.querySelector('.set-server').onchange = (e) => {
      this.serverUrl = e.target.value.trim();
    };

    // Prevent gizmo shortcuts when typing
    this.textInput.onfocus = () => { window._promptBarFocused = true; };
    this.textInput.onblur = () => { window._promptBarFocused = false; };
  }

  // ── Image handling ──────────────────────────────────────────────────

  setImage(file) {
    this.currentImage = file;
    const url = URL.createObjectURL(file);
    this.previewArea.querySelector('.preview-thumb').src = url;
    this.previewArea.querySelector('.preview-name').textContent = file.name;
    this.previewArea.style.display = 'flex';
    this.textInput.placeholder = 'Image mode — click Create to generate';
  }

  clearImage() {
    if (this.currentImage) {
      URL.revokeObjectURL(this.previewArea.querySelector('.preview-thumb').src);
    }
    this.currentImage = null;
    this.previewArea.style.display = 'none';
    this.textInput.placeholder = 'Imagine an object...';
    this.imageInput.value = '';
  }

  toggleSettings() {
    const vis = this.settingsPopup.style.display === 'none';
    this.settingsPopup.style.display = vis ? 'flex' : 'none';
  }

  // ── Generation ──────────────────────────────────────────────────────

  async generate() {
    if (this.isGenerating) return;

    if (this.currentImage) {
      await this.generateFromImage(this.currentImage);
    } else {
      const text = this.textInput.value.trim();
      if (!text) {
        this.showToast('Upload an image or type a prompt first', 'error');
        return;
      }
      await this.generateFromText(text);
    }
  }

  async generateFromImage(imageFile) {
    this.setLoading(true, `Generating from ${imageFile.name}...`);

    try {
      const base64 = await this.fileToBase64(imageFile);

      const response = await fetch(`${this.serverUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt_type: 'image',
          prompt_image: base64,
          seed: this.settings.seed,
          params: {
            sparse_structure_steps: this.settings.sparseSteps,
            slat_steps: this.settings.slatSteps,
            output_format: 'splat',
          },
        }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        await this.addGeneratedObject(result, imageFile.name.replace(/\.[^.]+$/, ''));
        this.showToast(`Generated "${imageFile.name}" (${result.gaussian_count} gaussians, ${result.generation_time.toFixed(1)}s)`);
        this.clearImage();
        this.textInput.value = '';
      } else {
        this.showToast(result.error || 'Generation failed', 'error');
      }
    } catch (error) {
      this.showToast(`Connection failed: ${error.message}`, 'error');
    } finally {
      this.setLoading(false);
    }
  }

  async generateFromText(promptText) {
    this.setLoading(true, `Generating "${promptText}"...`);

    try {
      const response = await fetch(`${this.serverUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt_type: 'text',
          prompt_text: promptText,
          seed: this.settings.seed,
          params: {
            sparse_structure_steps: this.settings.sparseSteps,
            slat_steps: this.settings.slatSteps,
            output_format: 'splat',
          },
        }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        await this.addGeneratedObject(result, promptText.slice(0, 20));
        this.showToast(`Generated "${promptText}"`);
        this.textInput.value = '';
      } else {
        this.showToast(result.error || 'Generation failed', 'error');
      }
    } catch (error) {
      this.showToast(`Connection failed: ${error.message}`, 'error');
    } finally {
      this.setLoading(false);
    }
  }

  async addGeneratedObject(result, name) {
    // Decode base64 → ArrayBuffer
    const binary = atob(result.ply_data);
    const buffer = new ArrayBuffer(binary.length);
    const view = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);

    // Parse based on format
    const format = result.format || detectFormat(buffer, name + '.splat');
    let parsed;
    if (format === 'ply') {
      parsed = parsePlyBuffer(buffer);
    } else {
      parsed = parseSplatBuffer(buffer);
    }

    // Place at camera forward 3m
    const viewMatrix = this.controls.getViewMatrix();
    const inv = invert4(viewMatrix);
    const camPos = inv ? [inv[12], inv[13], inv[14]] : [0, 0, 0];
    const camFwd = inv ? [-inv[8], -inv[9], -inv[10]] : [0, 0, -1];
    const placementPos = [
      camPos[0] + camFwd[0] * 3,
      camPos[1] + camFwd[1] * 3,
      camPos[2] + camFwd[2] * 3,
    ];

    // Add as object layer
    const layerId = this.sceneManager.addLayer(
      name, parsed.texdata, parsed.positions, parsed.count,
      parsed.texwidth, parsed.texheight, 'object'
    );

    this.sceneManager.setPosition(layerId, placementPos);
    this.sceneManager.selectLayer(layerId);
  }

  // ── UI State ────────────────────────────────────────────────────────

  setLoading(isLoading, message = '') {
    this.isGenerating = isLoading;
    this.createBtn.disabled = isLoading;
    this.textInput.disabled = isLoading;
    this.progressBar.style.display = isLoading ? 'block' : 'none';
    if (message) this.textInput.placeholder = message;
    else this.textInput.placeholder = this.currentImage ? 'Image mode — click Create' : 'Imagine an object...';
  }

  showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; toast.style.transform = 'translateY(0)'; });
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
}
