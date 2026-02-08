/**
 * DirectManip — Direct object manipulation via mouse.
 * Click+drag objects to move/rotate/scale without gizmo handles.
 *
 * Left drag on object   → XZ plane (floor) movement
 * Alt+left drag         → Y axis (height) movement
 * Right drag on object  → Y axis rotation (turntable)
 * Scroll on object      → Uniform scale
 * Short click (≤5px)    → Select only
 */

import { screenToRay, rayVsLayerBounds, rayPlaneIntersect } from './utils/ray-cast.js';

export class DirectManip {
  constructor(renderer, sceneManager, controls) {
    this.renderer = renderer;
    this.sceneManager = sceneManager;
    this.controls = controls;
    this.undoManager = null;  // set from main.js
    this.enabled = true;

    // Drag state
    this.dragType = null;        // 'xz-move' | 'y-move' | 'y-rotate' | null
    this.dragging = false;
    this.dragLayerId = null;
    this.dragStartMouse = null;  // {x, y}
    this.dragStartTransform = null; // {position, rotation, scale}
    this.dragStartRayHit = null; // [x,y,z] XZ plane intersection at drag start

    // Hover state
    this.hoveredLayerId = null;

    // Scale undo debounce
    this._scaleUndoState = null;
    this._scaleUndoTimer = null;
  }

  // ── Mouse Down ──────────────────────────────────────────────────────

  /**
   * Handle mousedown on a layer. Called from onLeftClick / onRightClick.
   * @returns {boolean} true if consumed (object hit), false for camera fallthrough
   */
  handleMouseDown(clientX, clientY, event) {
    if (!this.enabled) return false;
    if (this.dragging) return true; // already dragging, consume

    const ray = this._makeRay(clientX, clientY);
    const hit = rayVsLayerBounds(ray, this.sceneManager);
    if (!hit) return false; // empty space → camera

    const layer = this.sceneManager.getLayer(hit.layerId);
    if (!layer || layer.locked) return false;

    // Select the layer
    this.sceneManager.selectLayer(hit.layerId);

    // Record start state
    this.dragLayerId = hit.layerId;
    this.dragStartMouse = { x: clientX, y: clientY };
    this.dragStartTransform = {
      position: [...layer.position],
      rotation: [...layer.rotation],
      scale: layer.scale,
    };

    // Determine drag type
    if (event.button === 2) {
      this.dragType = 'y-rotate';
      this.renderer.gl.canvas.style.cursor = 'ew-resize';
    } else if (event.altKey) {
      this.dragType = 'y-move';
      this.renderer.gl.canvas.style.cursor = 'ns-resize';
    } else {
      this.dragType = 'xz-move';
      this.dragStartRayHit = rayPlaneIntersect(ray, layer.position, [0, 1, 0]);
      this.renderer.gl.canvas.style.cursor = 'move';
    }

    this.dragging = true;
    return true;
  }

  // ── Mouse Move ──────────────────────────────────────────────────────

  handleMouseMove(clientX, clientY, event) {
    if (this.dragging) {
      this._updateDrag(clientX, clientY, event.shiftKey);
      return;
    }

    // Hover detection (cursor feedback)
    if (!this.enabled) return;
    const ray = this._makeRay(clientX, clientY);
    const hit = rayVsLayerBounds(ray, this.sceneManager);
    const newHover = hit ? hit.layerId : null;
    if (newHover !== this.hoveredLayerId) {
      this.hoveredLayerId = newHover;
      // Only update cursor if we're in direct-manip mode (not gizmo hover)
      const canvas = this.renderer.gl.canvas;
      if (!canvas.style.cursor || canvas.style.cursor === 'pointer' || canvas.style.cursor === '') {
        canvas.style.cursor = newHover ? 'pointer' : '';
      }
    }
  }

  // ── Mouse Up ────────────────────────────────────────────────────────

  handleMouseUp(clientX, clientY, event) {
    if (!this.dragging) return;

    const dx = clientX - this.dragStartMouse.x;
    const dy = clientY - this.dragStartMouse.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 5) {
      // Short click: revert any micro-movement, keep selection
      const layer = this.sceneManager.getLayer(this.dragLayerId);
      if (layer && this.dragStartTransform) {
        this.sceneManager.setTransform(this.dragLayerId, this.dragStartTransform);
      }
    } else {
      // Real drag: push undo
      const layer = this.sceneManager.getLayer(this.dragLayerId);
      if (layer && this.undoManager) {
        this.undoManager.push({
          type: 'transform',
          layerId: layer.id,
          before: this.dragStartTransform,
          after: {
            position: [...layer.position],
            rotation: [...layer.rotation],
            scale: layer.scale,
          },
        });
      }
    }

    this._endDrag();
  }

  // ── Wheel (Scale) ──────────────────────────────────────────────────

  /**
   * Handle scroll on selected object → uniform scale.
   * @returns {boolean} true if consumed
   */
  handleWheel(clientX, clientY, deltaY, shiftKey) {
    if (!this.enabled) return false;

    const selected = this.sceneManager.getSelectedLayer();
    if (!selected || selected.locked || selected.type !== 'object') return false;

    // Check if cursor is over the selected object
    const ray = this._makeRay(clientX, clientY);
    const hit = rayVsLayerBounds(ray, this.sceneManager);
    if (!hit || hit.layerId !== selected.id) return false;

    // Compute new scale
    const scaleDelta = -deltaY * 0.001;
    let newScale = selected.scale * (1 + scaleDelta);
    newScale = Math.max(0.01, Math.min(100, newScale));
    if (shiftKey) newScale = Math.round(newScale * 4) / 4;

    // Debounced undo for scroll-to-scale
    if (!this._scaleUndoState) {
      this._scaleUndoState = {
        layerId: selected.id,
        before: {
          position: [...selected.position],
          rotation: [...selected.rotation],
          scale: selected.scale,
        },
      };
    }

    this.sceneManager.setScale(selected.id, newScale);

    clearTimeout(this._scaleUndoTimer);
    this._scaleUndoTimer = setTimeout(() => {
      if (this._scaleUndoState && this.undoManager) {
        const current = this.sceneManager.getLayer(this._scaleUndoState.layerId);
        if (current) {
          this.undoManager.push({
            type: 'transform',
            layerId: current.id,
            before: this._scaleUndoState.before,
            after: {
              position: [...current.position],
              rotation: [...current.rotation],
              scale: current.scale,
            },
          });
        }
      }
      this._scaleUndoState = null;
    }, 300);

    return true;
  }

  // ── Internal: Drag Update ──────────────────────────────────────────

  _updateDrag(clientX, clientY, shiftKey) {
    const layer = this.sceneManager.getLayer(this.dragLayerId);
    if (!layer) { this._endDrag(); return; }

    switch (this.dragType) {
      case 'xz-move': {
        const ray = this._makeRay(clientX, clientY);
        const currentHit = rayPlaneIntersect(ray, this.dragStartTransform.position, [0, 1, 0]);
        if (!this.dragStartRayHit || !currentHit) break;

        let dx = currentHit[0] - this.dragStartRayHit[0];
        let dz = currentHit[2] - this.dragStartRayHit[2];
        if (shiftKey) {
          dx = Math.round(dx / 0.5) * 0.5;
          dz = Math.round(dz / 0.5) * 0.5;
        }

        this.sceneManager.setPosition(layer.id, [
          this.dragStartTransform.position[0] + dx,
          this.dragStartTransform.position[1],
          this.dragStartTransform.position[2] + dz,
        ]);
        break;
      }

      case 'y-move': {
        const dyPx = this.dragStartMouse.y - clientY; // up = positive
        const dist = this.controls._getCameraDistance();
        const sensitivity = Math.max(0.001, dist * 0.003);
        let deltaY = dyPx * sensitivity;
        if (shiftKey) deltaY = Math.round(deltaY / 0.5) * 0.5;

        this.sceneManager.setPosition(layer.id, [
          this.dragStartTransform.position[0],
          this.dragStartTransform.position[1] + deltaY,
          this.dragStartTransform.position[2],
        ]);
        break;
      }

      case 'y-rotate': {
        const dxPx = clientX - this.dragStartMouse.x;
        let deltaDeg = dxPx * 0.5; // 0.5 deg per pixel
        if (shiftKey) deltaDeg = Math.round(deltaDeg / 15) * 15;

        this.sceneManager.setRotation(layer.id, [
          this.dragStartTransform.rotation[0],
          this.dragStartTransform.rotation[1] + deltaDeg,
          this.dragStartTransform.rotation[2],
        ]);
        break;
      }
    }
  }

  // ── Internal: End Drag ──────────────────────────────────────────────

  _endDrag() {
    this.dragging = false;
    this.dragType = null;
    this.dragLayerId = null;
    this.dragStartMouse = null;
    this.dragStartTransform = null;
    this.dragStartRayHit = null;
    const canvas = this.renderer.gl.canvas;
    canvas.style.cursor = this.hoveredLayerId ? 'pointer' : '';
  }

  // ── Internal: Ray Construction ──────────────────────────────────────

  _makeRay(clientX, clientY) {
    const gl = this.renderer.gl;
    return screenToRay(
      clientX, clientY,
      this.controls.getViewMatrix(),
      this.renderer.projectionMatrix,
      gl.canvas.width, gl.canvas.height
    );
  }
}
