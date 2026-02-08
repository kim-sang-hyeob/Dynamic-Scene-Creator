/**
 * PathEditor — Bezier path editor + dome camera animation + recording.
 * Ported from web_viewer_final/index.html Editor object to ES module class.
 * Follows the same integration pattern as gizmo.js.
 */

import { BezierMath } from './bezier-math.js';
import { PathOverlayRenderer } from './overlay-renderer.js';
import { multiply4, invert4 } from '../utils/matrix-math.js';

export class PathEditor {
  constructor(renderer, sceneManager, controls) {
    this.renderer = renderer;
    this.gl = renderer.gl;
    this.sceneManager = sceneManager;
    this.controls = controls;

    this.overlay = new PathOverlayRenderer();

    // Active state
    this.active = false;

    // Mode: 'view' | 'place' | 'select' | 'animate'
    this.mode = 'view';

    // Control points
    this.controlPoints = []; // [{id, position: [x,y,z]}]
    this.selectedIdx = -1;
    this.nextId = 0;
    this.segments = []; // Bezier segments from BezierMath

    // Drag state
    this.draggingPoint = false;
    this._lastDragTime = 0;

    // Animation
    this.animating = false;
    this.animStartTime = 0;
    this.duration = 5.0;
    this.savedViewMatrix = null;

    // Dome camera
    this.camDistance = 3.0;
    this.camAzimuth = 0;
    this.camElevation = 15;
    this.showCameras = true;

    // Ground plane axes
    this.mapUp = [0, -1, 0];
    this.mapRight = [1, 0, 0];
    this.mapBack = [0, 0, 1];

    // Recording
    this._recorder = null;
    this._recordedChunks = [];
    this.recording = false;
    this.exportFps = 30;

    // Undo
    this.undoStack = [];

    // Callback for UI updates
    this.onChange = null; // (eventType) => {}
  }

  init() {
    this.overlay.init(this.gl);
    this._extractMapUp(this.controls.getViewMatrix());
  }

  // ── Rendering ──────────────────────────────────────────────────────

  render(viewMatrix, projMatrix) {
    if (!this.active) return;
    this.overlay.render(viewMatrix, projMatrix);
  }

  // ── Mouse Event Handlers ───────────────────────────────────────────

  handleClick(clientX, clientY, event) {
    if (!this.active) return false;
    if (this.mode === 'place') {
      this.placePoint(clientX, clientY);
      return true;
    }
    if (this.mode === 'select') {
      this.selectPoint(clientX, clientY);
      return true;
    }
    return false;
  }

  handleMouseMove(clientX, clientY, event) {
    if (!this.active) return;
    if (this.mode === 'select' && this.selectedIdx >= 0 && this.draggingPoint) {
      // Throttle to ~30fps
      if (performance.now() - this._lastDragTime < 33) return;
      this._lastDragTime = performance.now();
      this.dragPoint(clientX, clientY);
    }
  }

  handleMouseUp(clientX, clientY, event) {
    if (!this.active) return;
    this.draggingPoint = false;
  }

  // ── Mode ───────────────────────────────────────────────────────────

  setMode(mode) {
    this.mode = mode;
    if (mode !== 'animate') this.stopAnimation();
    this._notify('mode');
  }

  // ── Gaussian Picking ───────────────────────────────────────────────

  pickGaussian(mouseX, mouseY) {
    const viewMatrix = this.controls.getViewMatrix();
    const projMatrix = this.renderer.projectionMatrix;
    if (!viewMatrix || !projMatrix) return null;

    const vp = multiply4(projMatrix, viewMatrix);
    const w = this.gl.canvas.width;
    const h = this.gl.canvas.height;
    let minDist = Infinity;
    let bestPos = null;

    for (const layer of this.sceneManager.getLayers()) {
      if (!layer.visible) continue;
      const pos = layer.bakedPositions;
      for (let i = 0; i < layer.count; i++) {
        const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
        const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
        const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        if (cw <= 0) continue;
        const sx = (cx / cw + 1) * 0.5 * w;
        const sy = (1 - cy / cw) * 0.5 * h;
        const d = (sx - mouseX) ** 2 + (sy - mouseY) ** 2;
        if (d < minDist) { minDist = d; bestPos = [x, y, z]; }
      }
    }
    return minDist < 50 * 50 ? bestPos : null;
  }

  // ── Point Operations ───────────────────────────────────────────────

  placePoint(mx, my) {
    const pos = this.pickGaussian(mx, my);
    if (!pos) { this._notify('place-miss'); return; }
    this.pushUndo();
    this.controlPoints.push({ id: this.nextId++, position: [...pos] });
    this.selectedIdx = this.controlPoints.length - 1;
    this.rebuildPath();
    this._notify('place');
  }

  selectPoint(mx, my) {
    if (this.controlPoints.length === 0) return;
    const viewMatrix = this.controls.getViewMatrix();
    const projMatrix = this.renderer.projectionMatrix;
    const vp = multiply4(projMatrix, viewMatrix);
    const w = this.gl.canvas.width, h = this.gl.canvas.height;
    let minDist = Infinity, bestIdx = -1;

    for (let i = 0; i < this.controlPoints.length; i++) {
      const p = this.controlPoints[i].position;
      const cx = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
      const cy = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
      const cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
      if (cw <= 0) continue;
      const sx = (cx / cw + 1) * 0.5 * w;
      const sy = (1 - cy / cw) * 0.5 * h;
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < minDist) { minDist = d; bestIdx = i; }
    }

    if (minDist < 30 * 30) {
      this.selectedIdx = bestIdx;
      this.draggingPoint = true;
      this.pushUndo();
    } else {
      this.selectedIdx = -1;
    }
    this._updateOverlay();
    this._notify('select');
  }

  dragPoint(mx, my) {
    if (this.selectedIdx < 0) return;
    const pos = this.pickGaussian(mx, my);
    if (pos) {
      this.controlPoints[this.selectedIdx].position = [...pos];
      this.rebuildPath();
    }
  }

  deleteSelected() {
    if (this.selectedIdx < 0) return;
    this.pushUndo();
    this.controlPoints.splice(this.selectedIdx, 1);
    this.selectedIdx = Math.min(this.selectedIdx, this.controlPoints.length - 1);
    this.rebuildPath();
    this._notify('delete');
  }

  clearAll() {
    if (this.controlPoints.length === 0) return;
    this.pushUndo();
    this.controlPoints = [];
    this.selectedIdx = -1;
    this.rebuildPath();
    this._notify('clear');
  }

  // ── Path Rebuild ───────────────────────────────────────────────────

  rebuildPath() {
    if (this.controlPoints.length >= 3) this._updateGroundAxes();
    this.segments = BezierMath.buildNaturalSplinePath(
      this.controlPoints.map(p => p.position)
    );
    this._updateOverlay();
    this.generateCameraFrustums();
    this._notify('rebuild');
  }

  _updateOverlay() {
    if (!this.overlay.gl) return;

    // Lines
    if (this.segments.length > 0) {
      const { positions, count } = BezierMath.sampleBezierPath(this.segments, 48);
      this.overlay.updateLines(positions, count);
    } else {
      this.overlay.updateLines(new Float32Array(0), 0);
    }

    // Points
    const n = this.controlPoints.length;
    if (n > 0) {
      const pos = new Float32Array(n * 3);
      const col = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        pos[3 * i] = this.controlPoints[i].position[0];
        pos[3 * i + 1] = this.controlPoints[i].position[1];
        pos[3 * i + 2] = this.controlPoints[i].position[2];
        if (i === this.selectedIdx) {
          col[4 * i] = 0.23; col[4 * i + 1] = 0.51; col[4 * i + 2] = 0.96; col[4 * i + 3] = 1;
        } else {
          col[4 * i] = 1; col[4 * i + 1] = 1; col[4 * i + 2] = 1; col[4 * i + 3] = 0.8;
        }
      }
      this.overlay.updatePoints(pos, col, n);
    } else {
      this.overlay.updatePoints(new Float32Array(0), new Float32Array(0), 0);
    }
  }

  // ── Camera Frustums ────────────────────────────────────────────────

  generateCameraFrustums(num = 12) {
    if (!this.overlay.gl || !this.showCameras || this.segments.length === 0) {
      this.overlay.updateCameraLines(new Float32Array(0), 0);
      return;
    }
    const v = [];
    const d = 0.2, hw = 0.12, hh = 0.08;
    for (let c = 0; c < num; c++) {
      const t = num > 1 ? c / (num - 1) : 0;
      const pos = BezierMath.evaluatePathAt(this.segments, t);
      if (!pos) continue;
      const { eye } = this.getDomeCameraEye(pos, t);
      const vm = this._lookAt(eye, pos, this.mapUp);
      if (!vm) continue;
      const c2w = invert4(vm);
      const r = [c2w[0], c2w[1], c2w[2]];
      const dw = [c2w[4], c2w[5], c2w[6]];
      const f = [c2w[8], c2w[9], c2w[10]];
      const e = [c2w[12], c2w[13], c2w[14]];
      const cx = e[0] + f[0] * d, cy = e[1] + f[1] * d, cz = e[2] + f[2] * d;
      const tl = [cx - r[0] * hw - dw[0] * hh, cy - r[1] * hw - dw[1] * hh, cz - r[2] * hw - dw[2] * hh];
      const tr = [cx + r[0] * hw - dw[0] * hh, cy + r[1] * hw - dw[1] * hh, cz + r[2] * hw - dw[2] * hh];
      const bl = [cx - r[0] * hw + dw[0] * hh, cy - r[1] * hw + dw[1] * hh, cz - r[2] * hw + dw[2] * hh];
      const br = [cx + r[0] * hw + dw[0] * hh, cy + r[1] * hw + dw[1] * hh, cz + r[2] * hw + dw[2] * hh];
      v.push(...e, ...tl, ...e, ...tr, ...e, ...bl, ...e, ...br, ...tl, ...tr, ...tr, ...br, ...br, ...bl, ...bl, ...tl);
    }
    this.overlay.updateCameraLines(new Float32Array(v), v.length / 3);
  }

  // ── Dome Camera ────────────────────────────────────────────────────

  getDomeCameraEye(pos, t) {
    const az = this.camAzimuth * Math.PI / 180;
    const el = this.camElevation * Math.PI / 180;
    const dist = this.camDistance;
    const hFwd = this._getHorizontalTangent(t);
    const rx = hFwd[1] * this.mapUp[2] - hFwd[2] * this.mapUp[1];
    const ry = hFwd[2] * this.mapUp[0] - hFwd[0] * this.mapUp[2];
    const rz = hFwd[0] * this.mapUp[1] - hFwd[1] * this.mapUp[0];
    const rl = Math.sqrt(rx ** 2 + ry ** 2 + rz ** 2);
    const hR = rl > 1e-8 ? [rx / rl, ry / rl, rz / rl] : [this.mapRight[0], this.mapRight[1], this.mapRight[2]];
    const ce = Math.cos(el), se = Math.sin(el), ca = Math.cos(az), sa = Math.sin(az);
    return {
      eye: [
        pos[0] + dist * (ce * sa * hR[0] + se * this.mapUp[0] + ce * ca * hFwd[0]),
        pos[1] + dist * (ce * sa * hR[1] + se * this.mapUp[1] + ce * ca * hFwd[1]),
        pos[2] + dist * (ce * sa * hR[2] + se * this.mapUp[2] + ce * ca * hFwd[2]),
      ]
    };
  }

  _getHorizontalTangent(t) {
    const dt = 0.005;
    const a = BezierMath.evaluatePathAt(this.segments, Math.max(0, t - dt));
    const b = BezierMath.evaluatePathAt(this.segments, Math.min(1, t + dt));
    if (!a || !b) return [-this.mapBack[0], -this.mapBack[1], -this.mapBack[2]];
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const dot = dx * this.mapUp[0] + dy * this.mapUp[1] + dz * this.mapUp[2];
    const h = [dx - dot * this.mapUp[0], dy - dot * this.mapUp[1], dz - dot * this.mapUp[2]];
    const len = Math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2);
    return len < 1e-8 ? [-this.mapBack[0], -this.mapBack[1], -this.mapBack[2]] : [h[0] / len, h[1] / len, h[2] / len];
  }

  // ── Animation ──────────────────────────────────────────────────────

  toggleAnimation() { this.animating ? this.stopAnimation() : this.startAnimation(); }

  startAnimation() {
    if (this.segments.length === 0) return;
    this.savedViewMatrix = this.controls.getViewMatrix();
    this.animating = true;
    this.controls.enabled = false;
    this.animStartTime = performance.now();
    this.setMode('animate');
    this._notify('animation-start');
    this._animLoop();
  }

  stopAnimation() {
    if (!this.animating) return;
    this.animating = false;
    this.controls.enabled = true;
    this.overlay.animIndicatorPos = null;
    if (this.savedViewMatrix) {
      this.controls.setViewMatrix(this.savedViewMatrix);
    }
    this.savedViewMatrix = null;
    this._notify('animation-stop');
  }

  _animLoop() {
    if (!this.animating) return;
    const t = Math.min((performance.now() - this.animStartTime) / 1000 / this.duration, 1);
    const pos = BezierMath.evaluatePathAt(this.segments, t);
    if (pos) {
      this.overlay.animIndicatorPos = pos;
      const { eye } = this.getDomeCameraEye(pos, t);
      const vm = this._lookAt(eye, pos, this.mapUp);
      if (vm) this.controls.setViewMatrix(vm);
    }
    if (t >= 1) {
      if (this.recording) { this.stopRecording(); this.stopAnimation(); return; }
      this.animStartTime = performance.now(); // loop
    }
    requestAnimationFrame(() => this._animLoop());
  }

  // ── Recording ──────────────────────────────────────────────────────

  toggleRecording() {
    this.recording ? (this.stopRecording(), this.stopAnimation()) : this.startRecording();
  }

  startRecording() {
    if (this.segments.length === 0) return;
    const canvas = this.renderer.canvas;

    // Save and resize for pro recording (1024px target)
    this._origWidth = window.innerWidth;
    this._origHeight = window.innerHeight;
    const targetSize = 1024;
    const ratio = this._origWidth / this._origHeight;
    let newW, newH;
    if (ratio > 1) { newW = targetSize; newH = Math.round(targetSize / ratio); }
    else { newH = targetSize; newW = Math.round(targetSize * ratio); }

    canvas.style.width = newW + 'px';
    canvas.style.height = newH + 'px';
    const origIW = window.innerWidth, origIH = window.innerHeight;
    Object.defineProperty(window, 'innerWidth', { value: newW, configurable: true });
    Object.defineProperty(window, 'innerHeight', { value: newH, configurable: true });
    window.dispatchEvent(new Event('resize'));

    this.overlay.hideOverlay = true;
    const stream = canvas.captureStream(this.exportFps);
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9' : 'video/webm;codecs=vp8';
    this._recordedChunks = [];
    this._recorder = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: 50_000_000
    });

    this._recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._recordedChunks.push(e.data);
    };
    this._recorder.onstop = () => {
      const blob = new Blob(this._recordedChunks, { type: mime });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `4dgs_scene_${Date.now()}.webm`;
      a.click();
      this._recordedChunks = [];
      // Restore size
      Object.defineProperty(window, 'innerWidth', { value: origIW, configurable: true });
      Object.defineProperty(window, 'innerHeight', { value: origIH, configurable: true });
      canvas.style.width = '';
      canvas.style.height = '';
      window.dispatchEvent(new Event('resize'));
    };

    this._recorder.start();
    this.recording = true;
    this._notify('recording-start');

    if (!this.animating) this.startAnimation();
    else this.animStartTime = performance.now();
  }

  stopRecording() {
    if (!this._recorder || this._recorder.state === 'inactive') return;
    this._recorder.stop();
    this.recording = false;
    this.overlay.hideOverlay = false;
    this._notify('recording-stop');
  }

  // ── Save / Load ────────────────────────────────────────────────────

  savePath() {
    const data = {
      version: '1.0',
      type: 'bezier_object_path',
      settings: {
        duration: this.duration,
        camDistance: this.camDistance,
        camAzimuth: this.camAzimuth,
        camElevation: this.camElevation,
      },
      controlPoints: this.controlPoints,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'scene_path.json';
    a.click();
    this._notify('save');
  }

  loadPath(file) {
    const fr = new FileReader();
    fr.onload = () => {
      try {
        const data = JSON.parse(fr.result);
        if (data.type !== 'bezier_object_path') throw new Error('Format mismatch');
        this.pushUndo();
        this.controlPoints = data.controlPoints;
        this.nextId = Math.max(...this.controlPoints.map(p => p.id), 0) + 1;
        if (data.settings) {
          this.duration = data.settings.duration ?? 5.0;
          this.camDistance = data.settings.camDistance ?? 3.0;
          this.camAzimuth = data.settings.camAzimuth ?? 0;
          this.camElevation = data.settings.camElevation ?? 15;
        }
        this.rebuildPath();
        this._notify('load');
      } catch (e) {
        console.error('Path load error:', e);
        this._notify('load-error');
      }
    };
    fr.readAsText(file);
  }

  // ── Export Dataset ─────────────────────────────────────────────────

  exportDataset() {
    if (this.segments.length === 0) return;
    const fps = this.exportFps;
    const total = Math.round(this.duration * fps);
    const canvas = this.renderer.canvas;
    const w = canvas.width || 1920, h = canvas.height || 1080;
    const pm = this.renderer.projectionMatrix;
    const fl_x = pm ? (pm[0] * w) / 2 : w;
    const fl_y = pm ? (-pm[5] * h) / 2 : h;

    const frames = [];
    for (let i = 0; i < total; i++) {
      const t = total > 1 ? i / (total - 1) : 0;
      const pos = BezierMath.evaluatePathAt(this.segments, t);
      if (!pos) continue;
      const { eye } = this.getDomeCameraEye(pos, t);
      const vm = this._lookAt(eye, pos, this.mapUp);
      if (vm) {
        const c2w = invert4(vm);
        frames.push({
          file_path: `./images/frame_${String(i).padStart(5, '0')}`,
          time: i / fps,
          time_normalized: t,
          transform_matrix: [
            [c2w[0], c2w[4], c2w[8], c2w[12]],
            [c2w[1], c2w[5], c2w[9], c2w[13]],
            [c2w[2], c2w[6], c2w[10], c2w[14]],
            [c2w[3], c2w[7], c2w[11], c2w[15]],
          ],
        });
      }
    }
    const output = {
      camera_angle_x: pm ? 2 * Math.atan(w / (2 * fl_x)) : 0.7,
      camera_angle_y: pm ? 2 * Math.atan(h / (2 * fl_y)) : 0.7,
      fl_x, fl_y, cx: w / 2, cy: h / 2, w, h,
      duration: this.duration, fps, frames,
    };
    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'transforms_train.json';
    a.click();
    this._notify('export');
  }

  // ── Undo ───────────────────────────────────────────────────────────

  pushUndo() {
    this.undoStack.push(JSON.stringify(this.controlPoints));
    if (this.undoStack.length > 50) this.undoStack.shift();
  }

  undo() {
    if (this.undoStack.length === 0) return;
    this.controlPoints = JSON.parse(this.undoStack.pop());
    this.rebuildPath();
    this._notify('undo');
  }

  // ── Private Helpers ────────────────────────────────────────────────

  /** LookAt matrix (forward=+Z convention matching original Editor) */
  _lookAt(eye, target, up) {
    const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    const fl = Math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2);
    if (fl < 1e-8) return null;
    f[0] /= fl; f[1] /= fl; f[2] /= fl;
    const r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
    const rl = Math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2);
    if (rl < 1e-8) return null;
    r[0] /= rl; r[1] /= rl; r[2] /= rl;
    const dw = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];
    return [
      r[0], dw[0], f[0], 0,
      r[1], dw[1], f[1], 0,
      r[2], dw[2], f[2], 0,
      -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]),
      -(dw[0] * eye[0] + dw[1] * eye[1] + dw[2] * eye[2]),
      -(f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]),
      1,
    ];
  }

  /** Extract map axes from a default view matrix */
  _extractMapUp(dvm) {
    if (!dvm || dvm.length < 16) return;
    const right = [dvm[0], dvm[4], dvm[8]];
    const up = [dvm[1], dvm[5], dvm[9]];
    const back = [dvm[2], dvm[6], dvm[10]];
    const uLen = Math.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2);
    if (uLen < 1e-8) return;
    const u = [up[0] / uLen, up[1] / uLen, up[2] / uLen];
    this.mapUp = [-u[0], -u[1], -u[2]]; // camera down → world up
    // Recompute mapRight from right vector, orthogonal to mapUp
    const rDotU = right[0] * u[0] + right[1] * u[1] + right[2] * u[2];
    const r = [right[0] - rDotU * u[0], right[1] - rDotU * u[1], right[2] - rDotU * u[2]];
    const rLen = Math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2);
    if (rLen > 1e-8) this.mapRight = [r[0] / rLen, r[1] / rLen, r[2] / rLen];
    // mapBack = cross(mapRight, mapUp)
    this.mapBack = [
      this.mapRight[1] * this.mapUp[2] - this.mapRight[2] * this.mapUp[1],
      this.mapRight[2] * this.mapUp[0] - this.mapRight[0] * this.mapUp[2],
      this.mapRight[0] * this.mapUp[1] - this.mapRight[1] * this.mapUp[0],
    ];
  }

  /** Update ground axes from 3+ control points (auto-derive ground plane) */
  _updateGroundAxes() {
    const points = this.controlPoints;
    if (points.length < 3) return;
    const p0 = points[0].position, p1 = points[1].position, p2 = points[2].position;
    const v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    const v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    const n = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]];
    const len = Math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2);
    if (len < 1e-8) return;
    const norm = [n[0] / len, n[1] / len, n[2] / len];

    // Orient so mapUp points away from camera
    const dvm = this.controls.getViewMatrix();
    const R = [[dvm[0], dvm[4], dvm[8]], [dvm[1], dvm[5], dvm[9]], [dvm[2], dvm[6], dvm[10]]];
    const t = [dvm[12], dvm[13], dvm[14]];
    const camPos = [
      -(R[0][0] * t[0] + R[0][1] * t[1] + R[0][2] * t[2]),
      -(R[1][0] * t[0] + R[1][1] * t[1] + R[1][2] * t[2]),
      -(R[2][0] * t[0] + R[2][1] * t[1] + R[2][2] * t[2]),
    ];
    const mid = [
      points.reduce((s, p) => s + p.position[0], 0) / points.length,
      points.reduce((s, p) => s + p.position[1], 0) / points.length,
      points.reduce((s, p) => s + p.position[2], 0) / points.length,
    ];
    const toCam = [camPos[0] - mid[0], camPos[1] - mid[1], camPos[2] - mid[2]];
    const dot = norm[0] * toCam[0] + norm[1] * toCam[1] + norm[2] * toCam[2];
    const up = dot > 0 ? norm : [-norm[0], -norm[1], -norm[2]];

    this.mapUp = up;
    // Compute forward from first two points
    const fwd = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    const fwdDotUp = fwd[0] * up[0] + fwd[1] * up[1] + fwd[2] * up[2];
    const hFwd = [fwd[0] - fwdDotUp * up[0], fwd[1] - fwdDotUp * up[1], fwd[2] - fwdDotUp * up[2]];
    const hLen = Math.sqrt(hFwd[0] ** 2 + hFwd[1] ** 2 + hFwd[2] ** 2);
    if (hLen > 1e-8) {
      this.mapBack = [-hFwd[0] / hLen, -hFwd[1] / hLen, -hFwd[2] / hLen];
      this.mapRight = [
        up[1] * this.mapBack[2] - up[2] * this.mapBack[1],
        up[2] * this.mapBack[0] - up[0] * this.mapBack[2],
        up[0] * this.mapBack[1] - up[1] * this.mapBack[0],
      ];
    }
  }

  _notify(eventType) {
    if (this.onChange) this.onChange(eventType);
  }
}
