/**
 * Scene Composer — App entry point.
 * Wires together: renderer, camera-controls, scene-manager, splat-loader,
 * gizmo, undo-manager. Handles drag-and-drop and keyboard shortcuts.
 */

import { SplatRenderer } from './renderer.js';
import { CameraControls } from './camera-controls.js';
import { SceneManager } from './scene-manager.js';
import { LayerPanel } from './layer-panel.js';
import { Gizmo } from './gizmo.js';
import { UndoManager } from './undo-manager.js';
import { getViewMatrix } from './utils/matrix-math.js';
import { parseSplatBuffer, parsePlyBuffer, parseSplatvBuffer, detectFormat } from './utils/splat-loader.js';
import { PromptBar } from './prompt-bar.js';
import { PathEditor } from './path-editor/path-editor.js';
import { PathEditorPanel } from './path-editor-panel.js';

// ── Globals ──────────────────────────────────────────────────────────
let renderer;
let controls;
let sceneManager;
let layerPanel;
let gizmo;
let undoManager;
let promptBar;
let pathEditor;
let pathEditorPanel;

// ── Init ─────────────────────────────────────────────────────────────
function init() {
  const canvas = document.getElementById('canvas');
  const fpsEl = document.getElementById('fps');
  const gaussianEl = document.getElementById('gaussian-count');
  const layerCountEl = document.getElementById('layer-count');
  const spinnerEl = document.getElementById('spinner');
  const progressEl = document.getElementById('progress');
  const dropZone = document.getElementById('drop-zone');

  // Renderer
  renderer = new SplatRenderer(canvas);
  renderer.init();

  // Camera
  controls = new CameraControls(canvas);

  // Scene Manager
  sceneManager = new SceneManager(renderer);

  // Undo Manager
  undoManager = new UndoManager(sceneManager);

  // Gizmo
  gizmo = new Gizmo(renderer, sceneManager, controls);
  gizmo.init();
  gizmo.undoManager = undoManager;

  // Path Editor
  pathEditor = new PathEditor(renderer, sceneManager, controls);
  pathEditor.init();
  pathEditorPanel = new PathEditorPanel(pathEditor);

  // Wire overlay rendering (gizmo + path editor, chained)
  renderer.onAfterDraw = (viewMatrix, projMatrix) => {
    gizmo.render(viewMatrix, projMatrix);
    pathEditor.render(viewMatrix, projMatrix);
  };

  // Wire click handling (gizmo first, then path editor)
  controls.onLeftClick = (clientX, clientY, event) => {
    if (gizmo.handleClick(clientX, clientY, event)) return true;
    if (pathEditor.handleClick(clientX, clientY, event)) return true;
    return false;
  };

  // Wire mousemove and mouseup (both gizmo and path editor)
  canvas.addEventListener('mousemove', (e) => {
    gizmo.handleMouseMove(e.clientX, e.clientY, e);
    pathEditor.handleMouseMove(e.clientX, e.clientY, e);
  });
  canvas.addEventListener('mouseup', (e) => {
    gizmo.handleMouseUp(e.clientX, e.clientY, e);
    pathEditor.handleMouseUp(e.clientX, e.clientY, e);
  });

  // Layer Panel
  layerPanel = new LayerPanel(sceneManager);
  layerPanel.bindTransformInputs();
  layerPanel.onAddLayerClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.splat,.ply,.splatv';
    input.multiple = true;
    input.onchange = async () => {
      for (const file of input.files) {
        await loadFileAsLayer(file);
      }
    };
    input.click();
  };

  // Prompt Bar
  promptBar = new PromptBar(sceneManager, controls);

  // Render loop
  renderer.startLoop(() => {
    const actualView = controls.update();

    // Update HUD
    if (fpsEl) fpsEl.textContent = renderer.fps + ' fps';
    if (gaussianEl) gaussianEl.textContent = sceneManager.totalGaussians.toLocaleString() + ' gaussians';
    if (layerCountEl) layerCountEl.textContent = sceneManager.layers.length + ' layers';

    return actualView;
  });

  // Drag and drop
  setupDragDrop(spinnerEl, progressEl, dropZone);

  // Keyboard shortcuts
  setupKeyboard();

  // Ctrl+V clipboard image paste
  setupClipboardPaste();

  // Expose for console debugging
  window.sceneComposer = { renderer, controls, sceneManager, layerPanel, gizmo, undoManager, promptBar, pathEditor, pathEditorPanel, loadFileAsLayer };

  console.log('Scene Composer initialized');
  console.log('Tip: drop .splat/.ply/.splatv files to add layers');
  console.log('Shortcuts: G=Translate, R=Rotate, S=Scale, Esc=Deselect, Del=Delete, Ctrl+Z/Y=Undo/Redo');
  console.log('Path Editor: Switch to "Path" tab in sidebar. 1-4=Modes, Space=Play');
}

// ── File Loading as Layer ────────────────────────────────────────────

/**
 * Load a file and add it as a new layer.
 * @param {File} file
 * @param {string} [layerName] - Optional layer name (defaults to filename)
 * @returns {Promise<number>} layer id
 */
async function loadFileAsLayer(file, layerName) {
  const buffer = await file.arrayBuffer();
  const format = detectFormat(buffer, file.name);

  let result;
  if (format === 'splat') {
    result = parseSplatBuffer(buffer);
  } else if (format === 'ply') {
    result = parsePlyBuffer(buffer);
  } else if (format === 'splatv') {
    result = parseSplatvBuffer(buffer);
    // Use camera from first splatv file loaded
    if (result.cameras && result.cameras.length > 0 && sceneManager.layers.length === 0) {
      const cam = result.cameras[0];
      if (cam.fx) renderer.camera.fx = cam.fx;
      if (cam.fy) renderer.camera.fy = cam.fy;
      renderer.resize();
      if (cam.rotation && cam.position) {
        controls.setViewMatrix(getViewMatrix(cam));
      }
    }
  } else {
    throw new Error('Unsupported format: ' + file.name);
  }

  const name = layerName || file.name.replace(/\.[^.]+$/, '');
  // First file becomes the map layer; subsequent files are objects
  const type = sceneManager.hasMap() ? 'object' : 'map';
  const layerId = sceneManager.addLayer(
    name, result.texdata, result.positions, result.count, result.texwidth, result.texheight, type
  );

  // Hide drop zone after first load
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) dropZone.classList.remove('empty');

  console.log(`Layer "${name}" (id=${layerId}): ${result.count.toLocaleString()} gaussians`);
  return layerId;
}

// ── Keyboard Shortcuts ──────────────────────────────────────────────
function setupKeyboard() {
  const keyDownTime = {};

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (!keyDownTime[e.code]) keyDownTime[e.code] = Date.now();

    // Space — toggle path animation (if path editor active)
    if (e.code === 'Space' && pathEditor.active) {
      e.preventDefault();
      pathEditor.toggleAnimation();
      return;
    }

    // Ctrl/Cmd shortcuts
    if (e.ctrlKey || e.metaKey) {
      if (e.code === 'KeyZ') {
        e.preventDefault();
        if (e.shiftKey) {
          undoManager.redo();
        } else {
          // Route undo to path editor when active, otherwise to undo manager
          if (pathEditor.active) {
            pathEditor.undo();
          } else {
            undoManager.undo();
          }
        }
      } else if (e.code === 'KeyY') {
        e.preventDefault();
        undoManager.redo();
      }
    }

    // Esc — deselect (immediate, no keyup delay)
    if (e.code === 'Escape') {
      gizmo.deselect();
    }

    // Delete — path editor point or layer
    if (e.code === 'Delete' || e.code === 'Backspace') {
      if (pathEditor.active && pathEditor.mode === 'select' && pathEditor.selectedIdx >= 0) {
        e.preventDefault();
        pathEditor.deleteSelected();
      } else {
        const selected = sceneManager.getSelectedLayer();
        if (selected) {
          e.preventDefault();
          sceneManager.removeLayer(selected.id);
        }
      }
    }
  });

  // Mode switching on keyup (short press < 300ms to avoid WASD conflict)
  window.addEventListener('keyup', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const dt = Date.now() - (keyDownTime[e.code] || 0);
    delete keyDownTime[e.code];

    if (dt < 300) {
      // Path editor mode shortcuts (1-4)
      if (pathEditor.active) {
        switch (e.code) {
          case 'Digit1': pathEditor.setMode('view'); return;
          case 'Digit2': pathEditor.setMode('place'); return;
          case 'Digit3': pathEditor.setMode('select'); return;
          case 'Digit4': pathEditor.setMode('animate'); return;
        }
      }
      // Gizmo mode shortcuts
      switch (e.code) {
        case 'KeyG': gizmo.setMode('translate'); break;
        case 'KeyR': gizmo.setMode('rotate'); break;
        case 'KeyS': gizmo.setMode('scale'); break;
      }
    }
  });
}

// ── Drag & Drop ──────────────────────────────────────────────────────
function setupDragDrop(spinnerEl, progressEl, dropZone) {
  const preventDefault = (e) => { e.preventDefault(); e.stopPropagation(); };

  document.addEventListener("dragenter", preventDefault);
  document.addEventListener("dragover", (e) => {
    preventDefault(e);
    if (dropZone) dropZone.classList.add('active');
  });
  document.addEventListener("dragleave", (e) => {
    preventDefault(e);
    if (dropZone) dropZone.classList.remove('active');
  });

  document.addEventListener("drop", async (e) => {
    preventDefault(e);
    if (dropZone) dropZone.classList.remove('active');

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    if (spinnerEl) spinnerEl.style.display = '';
    if (progressEl) {
      progressEl.style.display = '';
      progressEl.style.width = '0%';
    }

    try {
      for (let i = 0; i < files.length; i++) {
        if (progressEl) progressEl.style.width = ((i + 0.5) / files.length * 100) + '%';
        await loadFileAsLayer(files[i]);
        if (progressEl) progressEl.style.width = ((i + 1) / files.length * 100) + '%';
      }
      if (progressEl) {
        setTimeout(() => { progressEl.style.display = 'none'; }, 500);
      }
    } catch (err) {
      console.error('Failed to load file:', err);
      const msgEl = document.getElementById('message');
      if (msgEl) {
        msgEl.textContent = 'Error: ' + err.message;
        setTimeout(() => { msgEl.textContent = ''; }, 5000);
      }
    }

    if (spinnerEl) spinnerEl.style.display = 'none';
  });
}

// ── Clipboard Paste (Ctrl+V image) ──────────────────────────────────
function setupClipboardPaste() {
  document.addEventListener('paste', (e) => {
    if (!promptBar) return;
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) promptBar.setImage(file);
        break;
      }
    }
  });
}

// ── Start ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
