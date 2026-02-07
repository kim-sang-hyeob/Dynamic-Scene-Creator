/**
 * Gizmo — 3D Transform Gizmo controller.
 * Handles WebGL overlay rendering, mouse interaction (drag state machine),
 * and mode switching (translate/rotate/scale).
 */

import { multiply4, invert4, projectToScreen } from './utils/matrix-math.js';
import {
  buildTranslateGizmo, buildRotateGizmo, buildScaleGizmo,
  AXIS_COLORS, AXIS_COLORS_HIGHLIGHT,
} from './gizmo-geometry.js';
import {
  screenToRay, rayVsLayerBounds, rayVsAxis, rayVsRing, rayVsCube,
  rayAxisClosestT, invalidateBounds,
} from './utils/ray-cast.js';

const AXIS_DIRS = { x: [1,0,0], y: [0,1,0], z: [0,0,1] };

export class Gizmo {
  constructor(renderer, sceneManager, controls) {
    this.renderer = renderer;
    this.gl = renderer.gl;
    this.sceneManager = sceneManager;
    this.controls = controls;
    this.undoManager = null;

    // State
    this.mode = 'translate'; // 'translate' | 'rotate' | 'scale'
    this.activeAxis = null;
    this.hoveredAxis = null;
    this.dragging = false;
    this.dragStart = null;

    // GL resources
    this.program = null;
    this.vao = null;
    this.posBuffer = null;
    this.colorBuffer = null;
    this.u_viewProj = null;

    // Cached per-frame
    this._currentHitAreas = null;
    this._currentAxisLength = 1;
    this._currentCenter = null;
  }

  // ── GL Setup ────────────────────────────────────────────────────────

  init() {
    const gl = this.gl;

    const vsSource = `#version 300 es
      precision highp float;
      uniform mat4 u_viewProj;
      in vec3 a_position;
      in vec4 a_color;
      out vec4 vColor;
      void main() {
        gl_Position = u_viewProj * vec4(a_position, 1.0);
        vColor = a_color;
      }
    `;

    const fsSource = `#version 300 es
      precision highp float;
      in vec4 vColor;
      out vec4 fragColor;
      void main() {
        fragColor = vColor;
      }
    `;

    // Compile shaders
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('Gizmo VS:', gl.getShaderInfoLog(vs));
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('Gizmo FS:', gl.getShaderInfoLog(fs));
    }

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Gizmo link:', gl.getProgramInfoLog(this.program));
    }

    this.u_viewProj = gl.getUniformLocation(this.program, 'u_viewProj');

    // Create VAO
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    this.colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    const colLoc = gl.getAttribLocation(this.program, 'a_color');
    gl.enableVertexAttribArray(colLoc);
    gl.vertexAttribPointer(colLoc, 4, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
  }

  // ── Render ──────────────────────────────────────────────────────────

  render(viewMatrix, projMatrix) {
    const selectedLayer = this.sceneManager.getSelectedLayer();
    if (!selectedLayer) {
      this._currentHitAreas = null;
      return;
    }

    const gl = this.gl;
    const center = selectedLayer.position;

    // Compute gizmo size: constant screen size
    const invView = invert4(viewMatrix);
    const camPos = invView ? [invView[12], invView[13], invView[14]] : [0,0,5];
    const dx = camPos[0]-center[0], dy = camPos[1]-center[1], dz = camPos[2]-center[2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const axisLength = Math.max(0.3, dist * 0.15);

    // Build geometry for current mode
    let geom;
    if (this.mode === 'translate') {
      geom = buildTranslateGizmo(center, axisLength);
    } else if (this.mode === 'rotate') {
      geom = buildRotateGizmo(center, axisLength * 0.8);
    } else {
      geom = buildScaleGizmo(center, axisLength);
    }

    // Apply hover highlight
    if (this.hoveredAxis && geom.hitAreas[this.hoveredAxis]) {
      this._applyHighlight(geom, this.hoveredAxis);
    }

    // Cache for mouse interaction
    this._currentHitAreas = geom.hitAreas;
    this._currentAxisLength = axisLength;
    this._currentCenter = [...center];

    // Upload geometry
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, geom.positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, geom.colors, gl.DYNAMIC_DRAW);

    // Switch to overlay blend mode
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.blendEquation(gl.FUNC_ADD);

    // Compute viewProj
    const vp = multiply4(projMatrix, viewMatrix);

    // Draw
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.u_viewProj, false, vp);
    gl.bindVertexArray(this.vao);

    if (geom.lineVertexCount > 0) {
      gl.drawArrays(gl.LINES, 0, geom.lineVertexCount);
    }
    if (geom.triangleVertexCount > 0) {
      gl.drawArrays(gl.TRIANGLES, geom.triangleVertexStart, geom.triangleVertexCount);
    }

    gl.bindVertexArray(null);

    // Restore splat blend mode
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
  }

  _applyHighlight(geom, axis) {
    const highlight = AXIS_COLORS_HIGHLIGHT[axis];
    if (!highlight) return;
    const normal = AXIS_COLORS[axis];
    if (!normal) return;

    // Replace matching colors in the color array
    const colors = geom.colors;
    const totalVerts = colors.length / 4;
    for (let i = 0; i < totalVerts; i++) {
      const off = i * 4;
      if (Math.abs(colors[off]-normal[0]) < 0.01 &&
          Math.abs(colors[off+1]-normal[1]) < 0.01 &&
          Math.abs(colors[off+2]-normal[2]) < 0.01) {
        colors[off] = highlight[0];
        colors[off+1] = highlight[1];
        colors[off+2] = highlight[2];
        colors[off+3] = highlight[3];
      }
    }
  }

  // ── Mouse Interaction ─────────────────────────────────────────────

  /**
   * Handle mousedown on canvas.
   * Only consumes click if a gizmo handle is hit (starts drag).
   * Layer selection is deferred to mouseup (click = down+up with minimal movement).
   * @returns {boolean} true if consumed (no camera orbit), false to allow orbit
   */
  handleClick(clientX, clientY, event) {
    // Record click start for click-vs-drag detection in mouseup
    this._clickStart = { x: clientX, y: clientY };

    // Only consume if gizmo handle hit (start drag immediately)
    if (this.sceneManager.getSelectedLayer() && this._currentHitAreas) {
      const ray = this._makeRay(clientX, clientY);
      const hitAxis = this._testGizmoHit(ray);
      if (hitAxis) {
        this._startDrag(hitAxis, clientX, clientY, ray);
        return true; // consume — no orbit
      }
    }

    // Everything else: allow camera orbit, selection happens on mouseup
    return false;
  }

  handleMouseMove(clientX, clientY, event) {
    if (this.dragging) {
      this._updateDrag(clientX, clientY, event.shiftKey);
      return;
    }

    // Hover detection
    if (this.sceneManager.getSelectedLayer() && this._currentHitAreas) {
      const ray = this._makeRay(clientX, clientY);
      const prev = this.hoveredAxis;
      this.hoveredAxis = this._testGizmoHit(ray);
      if (this.hoveredAxis !== prev) {
        this.gl.canvas.style.cursor = this.hoveredAxis ? 'grab' : '';
      }
    } else {
      if (this.hoveredAxis) {
        this.hoveredAxis = null;
        this.gl.canvas.style.cursor = '';
      }
    }
  }

  handleMouseUp(clientX, clientY, event) {
    if (this.dragging) {
      this._endDrag();
      return;
    }

    // Click detection: mouseup close to mousedown = click (not drag)
    if (this._clickStart) {
      const dx = clientX - this._clickStart.x;
      const dy = clientY - this._clickStart.y;
      if (dx*dx + dy*dy < 25) { // < 5px movement = click
        this._handleSelection(clientX, clientY);
      }
      this._clickStart = null;
    }
  }

  /** Try to select a layer by ray cast. Called on short click only. */
  _handleSelection(clientX, clientY) {
    const ray = this._makeRay(clientX, clientY);
    const hit = rayVsLayerBounds(ray, this.sceneManager);
    if (hit) {
      this.sceneManager.selectLayer(hit.layerId);
    } else {
      this.sceneManager.selectLayer(null);
    }
  }

  // ── Mode Switching ────────────────────────────────────────────────

  setMode(mode) {
    this.mode = mode;
    this.hoveredAxis = null;
    this._updateModeHUD();
  }

  deselect() {
    this.sceneManager.selectLayer(null);
    this.hoveredAxis = null;
    this.activeAxis = null;
    if (this.dragging) {
      this.dragging = false;
      this.gl.canvas.style.cursor = '';
    }
    this._updateModeHUD();
  }

  _updateModeHUD() {
    const el = document.getElementById('gizmo-mode');
    if (!el) return;
    const labels = { translate: 'G: Translate', rotate: 'R: Rotate', scale: 'S: Scale' };
    el.textContent = labels[this.mode] || '';
    el.style.display = this.sceneManager.getSelectedLayer() ? '' : 'none';
  }

  // ── Internal: Ray building ────────────────────────────────────────

  _makeRay(clientX, clientY) {
    return screenToRay(
      clientX, clientY,
      this.controls.getViewMatrix(),
      this.renderer.projectionMatrix,
      this.gl.canvas.width, this.gl.canvas.height
    );
  }

  // ── Internal: Gizmo hit testing ───────────────────────────────────

  _testGizmoHit(ray) {
    const ha = this._currentHitAreas;
    const threshold = this._currentAxisLength * 0.12;
    let bestAxis = null;
    let bestT = Infinity;

    if (this.mode === 'translate') {
      for (const axis of ['x', 'y', 'z']) {
        if (!ha[axis]) continue;
        const result = rayVsAxis(ray, ha[axis].start, ha[axis].end, threshold);
        if (result && result.t < bestT) {
          bestT = result.t;
          bestAxis = axis;
        }
      }
    } else if (this.mode === 'rotate') {
      for (const axis of ['x', 'y', 'z']) {
        if (!ha[axis]) continue;
        const result = rayVsRing(ray, ha[axis].center, ha[axis].normal, ha[axis].radius, threshold);
        if (result && result.t < bestT) {
          bestT = result.t;
          bestAxis = axis;
        }
      }
    } else if (this.mode === 'scale') {
      // Test center cube first (higher priority if overlapping)
      if (ha.center) {
        const result = rayVsCube(ray, ha.center.cubeCenter, ha.center.cubeSize);
        if (result && result.t < bestT) {
          bestT = result.t;
          bestAxis = 'center';
        }
      }
      for (const axis of ['x', 'y', 'z']) {
        if (!ha[axis]) continue;
        const cubeResult = rayVsCube(ray, ha[axis].cubeCenter, ha[axis].cubeSize);
        if (cubeResult && cubeResult.t < bestT) {
          bestT = cubeResult.t;
          bestAxis = axis;
        }
        // Also test axis line
        const lineResult = rayVsAxis(ray,
          this._currentCenter, ha[axis].cubeCenter, threshold);
        if (lineResult && lineResult.t < bestT) {
          bestT = lineResult.t;
          bestAxis = axis;
        }
      }
    }

    return bestAxis;
  }

  // ── Internal: Drag ────────────────────────────────────────────────

  _startDrag(axis, clientX, clientY, ray) {
    const layer = this.sceneManager.getSelectedLayer();
    if (!layer || layer.locked) return;

    this.dragging = true;
    this.activeAxis = axis;
    this.dragStart = {
      clientX, clientY, ray,
      position: [...layer.position],
      rotation: [...layer.rotation],
      scale: layer.scale,
    };
    this.gl.canvas.style.cursor = 'grabbing';
  }

  _updateDrag(clientX, clientY, shiftKey) {
    const layer = this.sceneManager.getSelectedLayer();
    if (!layer) { this._endDrag(); return; }

    if (this.mode === 'translate') {
      this._updateTranslate(clientX, clientY, layer, shiftKey);
    } else if (this.mode === 'rotate') {
      this._updateRotate(clientX, clientY, layer, shiftKey);
    } else if (this.mode === 'scale') {
      this._updateScale(clientX, clientY, layer, shiftKey);
    }
  }

  _updateTranslate(clientX, clientY, layer, shiftKey) {
    const ray = this._makeRay(clientX, clientY);
    const axisDir = AXIS_DIRS[this.activeAxis];
    if (!axisDir) return;

    const startT = rayAxisClosestT(this.dragStart.ray, this.dragStart.position, axisDir);
    const currentT = rayAxisClosestT(ray, this.dragStart.position, axisDir);
    let delta = currentT - startT;

    if (shiftKey) delta = Math.round(delta / 0.5) * 0.5;

    const newPos = [...this.dragStart.position];
    const axisIndex = { x: 0, y: 1, z: 2 }[this.activeAxis];
    newPos[axisIndex] += delta;

    this.sceneManager.setPosition(layer.id, newPos);
  }

  _updateRotate(clientX, clientY, layer, shiftKey) {
    const vp = multiply4(this.renderer.projectionMatrix, this.controls.getViewMatrix());
    const center2D = projectToScreen(this._currentCenter || layer.position, vp,
      this.gl.canvas.width, this.gl.canvas.height);
    if (!center2D) return;

    const startAngle = Math.atan2(
      this.dragStart.clientY - center2D[1],
      this.dragStart.clientX - center2D[0]
    );
    const currentAngle = Math.atan2(
      clientY - center2D[1],
      clientX - center2D[0]
    );
    let deltaDeg = (currentAngle - startAngle) * (180 / Math.PI);

    if (shiftKey) deltaDeg = Math.round(deltaDeg / 15) * 15;

    const newRot = [...this.dragStart.rotation];
    const axisIndex = { x: 0, y: 1, z: 2 }[this.activeAxis];
    if (axisIndex !== undefined) {
      newRot[axisIndex] = this.dragStart.rotation[axisIndex] + deltaDeg;
    }

    this.sceneManager.setRotation(layer.id, newRot);
  }

  _updateScale(clientX, clientY, layer, shiftKey) {
    const deltaY = this.dragStart.clientY - clientY;
    let scaleFactor = 1.0 + deltaY * 0.005;

    let newScale = this.dragStart.scale * scaleFactor;
    newScale = Math.max(0.01, Math.min(100.0, newScale));

    if (shiftKey) newScale = Math.round(newScale * 4) / 4;

    this.sceneManager.setScale(layer.id, newScale);
  }

  _endDrag() {
    if (!this.dragging) return;

    const layer = this.sceneManager.getSelectedLayer();
    if (layer && this.undoManager && this.dragStart) {
      this.undoManager.push({
        type: 'transform',
        layerId: layer.id,
        before: {
          position: this.dragStart.position,
          rotation: this.dragStart.rotation,
          scale: this.dragStart.scale,
        },
        after: {
          position: [...layer.position],
          rotation: [...layer.rotation],
          scale: layer.scale,
        },
      });
    }

    this.dragging = false;
    this.activeAxis = null;
    this.gl.canvas.style.cursor = this.hoveredAxis ? 'grab' : '';
  }
}
