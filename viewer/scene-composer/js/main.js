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
import { WorldLabsModal } from './worldlabs-modal.js';
import { PathEditor } from './path-editor/path-editor.js';
import { PathEditorPanel } from './path-editor-panel.js';
import { DirectManip } from './direct-manip.js';
import { SelectionBox } from './selection-box.js';
import { ObjectAnimator } from './object-animator.js';

// ── Globals ──────────────────────────────────────────────────────────
let renderer;
let controls;
let sceneManager;
let layerPanel;
let gizmo;
let undoManager;
let promptBar;
let worldLabsModal;
let pathEditor;
let pathEditorPanel;
let directManip;
let selectionBox;
let objectAnimator;

// Pending object path assignment (when path is drawn before object is dropped)
let pendingObjectPath = null;

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

  // Re-show empty state when all layers removed + toggle Map button visibility
  // (must be before LayerPanel onChange chain)
  sceneManager.onChange = (event, data) => {
    if (event === 'removeLayer' && sceneManager.layers.length === 0) {
      document.getElementById('drop-zone')?.classList.add('empty');
    }
    // Show/hide Map button based on whether a map layer exists
    const mapBtn = document.querySelector('#prompt-bar .map-btn');
    if (mapBtn) {
      mapBtn.style.display = sceneManager.hasMap() ? 'none' : '';
    }
  };

  // Undo Manager
  undoManager = new UndoManager(sceneManager);

  // Gizmo (auxiliary — hidden by default, shown via G/R/S keys or buttons)
  gizmo = new Gizmo(renderer, sceneManager, controls);
  gizmo.init();
  gizmo.undoManager = undoManager;

  // Direct Manipulation (primary interaction)
  directManip = new DirectManip(renderer, sceneManager, controls);
  directManip.undoManager = undoManager;

  // Selection Box (AABB wireframe around selected object)
  selectionBox = new SelectionBox(renderer, sceneManager);
  selectionBox.init();

  // Gizmo mode buttons (top HUD) — clicking shows gizmo
  document.getElementById('gizmo-modes')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.gizmo-mode-btn');
    if (btn && btn.dataset.mode) {
      gizmo.setMode(btn.dataset.mode); // setMode sets visible=true
    }
  });

  // Path Editor
  pathEditor = new PathEditor(renderer, sceneManager, controls);
  pathEditor.init();
  pathEditorPanel = new PathEditorPanel(pathEditor);

  // Wire overlay rendering (selection box → gizmo → path editor)
  renderer.onAfterDraw = (viewMatrix, projMatrix) => {
    selectionBox.render(viewMatrix, projMatrix);
    gizmo.render(viewMatrix, projMatrix);
    pathEditor.render(viewMatrix, projMatrix);
  };

  // ── Mouse Event Wiring ──────────────────────────────────────────────
  // Priority: gizmo (when visible) → directManip → path editor → camera
  let bgClickStart = null;

  controls.onLeftClick = (clientX, clientY, event) => {
    bgClickStart = { x: clientX, y: clientY };
    if (pathEditor.active) {
      if (pathEditor.handleClick(clientX, clientY, event)) { bgClickStart = null; return true; }
      return false;
    }
    if (gizmo.visible && gizmo.handleClick(clientX, clientY, event)) { bgClickStart = null; return true; }
    if (directManip.handleMouseDown(clientX, clientY, event)) { bgClickStart = null; return true; }
    return false;
  };

  controls.onRightClick = (clientX, clientY, event) => {
    if (pathEditor.active) return false;
    return directManip.handleMouseDown(clientX, clientY, event);
  };

  controls.onWheel = (clientX, clientY, deltaY, shiftKey, event) => {
    if (pathEditor.active) return false;
    return directManip.handleWheel(clientX, clientY, deltaY, shiftKey);
  };

  canvas.addEventListener('mousemove', (e) => {
    gizmo.handleMouseMove(e.clientX, e.clientY, e);
    directManip.handleMouseMove(e.clientX, e.clientY, e);
    pathEditor.handleMouseMove(e.clientX, e.clientY, e);
  });
  canvas.addEventListener('mouseup', (e) => {
    gizmo.handleMouseUp(e.clientX, e.clientY, e);
    directManip.handleMouseUp(e.clientX, e.clientY, e);
    pathEditor.handleMouseUp(e.clientX, e.clientY, e);
    // Empty space short click → deselect
    if (bgClickStart && e.button === 0) {
      const dx = e.clientX - bgClickStart.x, dy = e.clientY - bgClickStart.y;
      if (dx * dx + dy * dy < 25) {
        sceneManager.selectLayer(null);
        gizmo.hide();
      }
      bgClickStart = null;
    }
  });

  // Layer Panel
  layerPanel = new LayerPanel(sceneManager);
  layerPanel.controls = controls;  // For export
  layerPanel.pathEditor = pathEditor;  // For export
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

  // World Labs Modal (Map generation)
  worldLabsModal = new WorldLabsModal(sceneManager, controls);

  // Wire Map button in prompt bar to open World Labs modal
  const mapBtn = document.querySelector('#prompt-bar .map-btn');
  if (mapBtn) mapBtn.onclick = () => worldLabsModal.open();

  // Object Animator (path animation for objects)
  objectAnimator = new ObjectAnimator(renderer, sceneManager);
  layerPanel.objectAnimator = objectAnimator;  // For animation controls UI

  // Render loop
  renderer.startLoop(() => {
    const actualView = controls.update();

    // Update object path animations
    objectAnimator.update();

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
  window.sceneComposer = { renderer, controls, sceneManager, layerPanel, gizmo, directManip, selectionBox, undoManager, promptBar, worldLabsModal, pathEditor, pathEditorPanel, objectAnimator, loadFileAsLayer, assignPathToLayer };

  // Help overlay — close on backdrop click or close button
  const helpOverlay = document.getElementById('help-overlay');
  if (helpOverlay) {
    helpOverlay.addEventListener('click', (e) => {
      if (e.target === helpOverlay) helpOverlay.style.display = 'none';
    });
    document.getElementById('help-close-btn')?.addEventListener('click', () => {
      helpOverlay.style.display = 'none';
    });
  }

  // Hint bubble — auto-hide after 5 seconds
  setTimeout(() => {
    const h = document.getElementById('hint-bubble');
    if (h) { h.classList.add('hidden'); setTimeout(() => h.remove(), 600); }
  }, 5000);

  console.log('Scene Composer initialized');
  console.log('Tip: drop .splat/.ply/.splatv files to add layers');
  console.log('Direct: Drag=Move, Alt+Drag=Height, RightDrag=Rotate, Scroll=Scale');
  console.log('Gizmo: G=Translate, R=Rotate, Esc=Deselect, Del=Delete, Ctrl+Z/Y=Undo/Redo');
  console.log('Path Editor: Switch to "Path" tab in sidebar. 1-4=Modes, Space=Play');
}

// ── File Loading as Layer ────────────────────────────────────────────

/**
 * Load a file and add it as a new layer (or multiple layers if splatv with layer metadata).
 * @param {File} file
 * @param {string} [layerName] - Optional layer name (defaults to filename)
 * @returns {Promise<number|number[]>} layer id(s)
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

    // If splatv has layer metadata, restore layers separately
    if (result.layers && result.layers.length > 0) {
      const layerIds = await loadSplatvWithLayers(result, file.name);
      return layerIds;
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

  // If this is an object layer and there's a pending path, assign it
  if (type === 'object' && pendingObjectPath) {
    assignPathToLayer(layerId, pendingObjectPath);
    showToast(`Path assigned to "${name}"!`);
    pendingObjectPath = null;
  }
  // Or if there's a current path in Path Editor and this is an object, offer to assign
  else if (type === 'object' && pathEditor && pathEditor.controlPoints && pathEditor.controlPoints.length >= 2) {
    // Show assignment dialog
    showPathAssignmentDialog(layerId, name);
  }

  return layerId;
}

/**
 * Load splatv file with layer metadata, restoring separate layers.
 * @param {Object} result - Parsed splatv result with layers metadata
 * @param {string} filename - Original filename for logging
 * @returns {Promise<number[]>} Array of layer ids
 */
async function loadSplatvWithLayers(result, filename) {
  const layerIds = [];
  const texdata = result.texdata;
  const texdata_f = new Float32Array(texdata.buffer);

  console.log(`Restoring ${result.layers.length} layers from ${filename}...`);

  for (const layerMeta of result.layers) {
    const startIdx = layerMeta.start_index;
    const endIdx = layerMeta.end_index;
    const count = layerMeta.count || (endIdx - startIdx);

    // Extract this layer's texdata (16 uint32 per gaussian)
    const layerTexdata = new Uint32Array(count * 16);
    layerTexdata.set(texdata.subarray(startIdx * 16, endIdx * 16));

    // Extract positions from texdata
    const layerTexdata_f = new Float32Array(layerTexdata.buffer);
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[3 * i + 0] = layerTexdata_f[16 * i + 0];
      positions[3 * i + 1] = layerTexdata_f[16 * i + 1];
      positions[3 * i + 2] = layerTexdata_f[16 * i + 2];
    }

    // Calculate texwidth/texheight for this layer
    const texwidth = 1024 * 4;
    const texheight = Math.ceil((4 * count) / texwidth);

    // Add layer with original type
    const layerId = sceneManager.addLayer(
      layerMeta.name,
      layerTexdata,
      positions,
      count,
      texwidth,
      texheight,
      layerMeta.type || 'object'
    );

    // Note: Transforms are already baked into the texdata during export,
    // so we don't need to apply them again. But we store the original
    // transform values for reference/UI display.
    const layer = sceneManager.getLayer(layerId);
    if (layer && layerMeta.position) {
      // Store original transform for UI display (already baked in data)
      layer._originalPosition = layerMeta.position;
      layer._originalRotation = layerMeta.rotation;
      layer._originalScale = layerMeta.scale;
    }

    // Restore path animation if present
    if (layerMeta.path && layerMeta.path.controlPoints && layerMeta.path.controlPoints.length >= 2) {
      objectAnimator.setLayerPath(layerId, layerMeta.path);
      console.log(`    Path restored: ${layerMeta.path.controlPoints.length} control points`);
    }

    layerIds.push(layerId);
    console.log(`  Layer "${layerMeta.name}" (${layerMeta.type}): ${count.toLocaleString()} gaussians`);
  }

  // Hide drop zone after load
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) dropZone.classList.remove('empty');

  return layerIds;
}

// ── Object Path Assignment ───────────────────────────────────────────

/**
 * Assign a path to an object layer.
 * @param {number} layerId - Layer ID
 * @param {Object} pathData - Path data with controlPoints and settings
 */
function assignPathToLayer(layerId, pathData) {
  if (!objectAnimator) return;

  objectAnimator.setLayerPath(layerId, pathData);

  // Update layer panel to show animation controls
  if (layerPanel) {
    layerPanel.render();
  }

  console.log(`Path assigned to layer ${layerId}`);
}

/**
 * Set a pending path that will be assigned to the next dropped object.
 * @param {Object} pathData - Path data with controlPoints and settings
 */
function setPendingObjectPath(pathData) {
  pendingObjectPath = pathData;
  console.log(`Pending object path set with ${pathData.controlPoints?.length || 0} points`);

  // Show notification
  showToast('Path ready! Drop a .splatv file to animate it along this path.');
}

/**
 * Get path data from the current Path Editor state.
 * @returns {Object|null} Path data or null if no points
 */
function getCurrentPathData() {
  if (!pathEditor || !pathEditor.controlPoints || pathEditor.controlPoints.length < 2) {
    return null;
  }

  return {
    controlPoints: pathEditor.controlPoints.map(p => ({
      position: [...p.position],
      target: p.target ? [...p.target] : null
    })),
    settings: {
      duration: pathEditor.duration || 5,
      pathSpeed: 0.5,
      walkSpeed: 1.0
    }
  };
}

/**
 * Show toast notification.
 */
function showToast(message, type = 'success') {
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

/**
 * Show dialog asking if user wants to assign current path to dropped object.
 */
function showPathAssignmentDialog(layerId, layerName) {
  const overlay = document.createElement('div');
  overlay.className = 'export-modal-overlay';
  overlay.innerHTML = `
    <div class="export-modal glass" style="width: 320px;">
      <div class="export-modal-header">
        <h3>Assign Path?</h3>
        <button class="export-modal-close">&times;</button>
      </div>
      <div class="export-modal-body">
        <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 16px;">
          A path with <strong>${pathEditor.controlPoints.length}</strong> control points is available.
          Assign it to "<strong>${layerName}</strong>"?
        </p>
        <div style="display: flex; gap: 8px; justify-content: flex-end;">
          <button class="assign-no-btn" style="padding: 8px 16px; border: 1px solid var(--border); border-radius: 8px; background: transparent; color: var(--text-secondary); cursor: pointer;">No</button>
          <button class="assign-yes-btn" style="padding: 8px 16px; border: none; border-radius: 8px; background: var(--accent); color: white; cursor: pointer; font-weight: 500;">Yes, Animate!</button>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const closeModal = () => overlay.remove();

  overlay.querySelector('.export-modal-close').onclick = closeModal;
  overlay.querySelector('.assign-no-btn').onclick = closeModal;
  overlay.onclick = (e) => {
    if (e.target === overlay) closeModal();
  };

  overlay.querySelector('.assign-yes-btn').onclick = () => {
    const pathData = getCurrentPathData();
    if (pathData) {
      assignPathToLayer(layerId, pathData);
      // Clear path points from viewer after assignment
      pathEditor.clearAll();
      showToast(`Path assigned to "${layerName}"!`);
    }
    closeModal();
  };
}

// ── Keyboard Shortcuts ──────────────────────────────────────────────
function setupKeyboard() {
  const keyDownTime = {};

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (!keyDownTime[e.code]) keyDownTime[e.code] = Date.now();

    // ? key — toggle help overlay
    if (e.key === '?' || (e.shiftKey && e.code === 'Slash')) {
      e.preventDefault();
      const helpEl = document.getElementById('help-overlay');
      if (helpEl) helpEl.style.display = helpEl.style.display === 'none' ? 'flex' : 'none';
      return;
    }

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

    // Esc — close help overlay first, then hide gizmo + deselect
    if (e.code === 'Escape') {
      const helpEl = document.getElementById('help-overlay');
      if (helpEl && helpEl.style.display !== 'none') {
        helpEl.style.display = 'none';
        return;
      }
      gizmo.hide();
      sceneManager.selectLayer(null);
    }

    // Arrow key nudge — move selected object instead of camera
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown'].includes(e.code)) {
      const selected = sceneManager.getSelectedLayer();
      if (selected && !selected.locked) {
        e.preventDefault();
        const step = e.shiftKey ? 0.5 : 0.1;
        const pos = [...selected.position];
        switch (e.code) {
          case 'ArrowUp':    pos[2] -= step; break;  // Z-
          case 'ArrowDown':  pos[2] += step; break;  // Z+
          case 'ArrowLeft':  pos[0] -= step; break;  // X-
          case 'ArrowRight': pos[0] += step; break;  // X+
          case 'PageUp':     pos[1] += step; break;  // Y+
          case 'PageDown':   pos[1] -= step; break;  // Y-
        }
        sceneManager.setPosition(selected.id, pos);
        // Remove from camera activeKeys so camera doesn't also move
        controls.activeKeys = controls.activeKeys.filter(
          k => !['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)
        );
        return;
      }
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

  // Mode switching on keyup (short press < 300ms)
  // NOTE: S key removed — it conflicts with WASD backward.
  // Use the mode buttons in the top HUD or G/R shortcuts instead.
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
      // Gizmo mode shortcuts (show gizmo only when a layer is selected)
      if (sceneManager.getSelectedLayer()) {
        switch (e.code) {
          case 'KeyG': gizmo.show('translate'); break;
          case 'KeyR': gizmo.show('rotate'); break;
        }
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

    // Don't handle drops that landed on the prompt bar (it handles images itself)
    if (e.target.closest('#prompt-bar')) return;

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
