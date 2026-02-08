/**
 * Camera controls for 3DGS viewer.
 * Extracted from hybrid.js: orbit, pan, zoom, WASD, touch gestures.
 * Enhanced: damping/inertia, elevation clamp, distance-based speed.
 */

import { invert4, rotate4, translate4 } from './utils/matrix-math.js';

export class CameraControls {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.viewMatrix = options.viewMatrix || [
      -0.97, 0.13, 0.22, 0,
       0.04, 0.91, -0.41, 0,
      -0.25, -0.39, -0.89, 0,
      -1.32, 1.59, 2.84, 1,
    ];
    this.activeKeys = [];
    this.down = false;
    this.startX = 0;
    this.startY = 0;
    this.altX = 0;
    this.altY = 0;
    this.jumpDelta = 0;
    this.enabled = true;

    // Intercept callbacks
    this.onLeftClick = null;   // (clientX, clientY, event) => boolean
    this.onRightClick = null;  // (clientX, clientY, event) => boolean
    this.onWheel = null;       // (clientX, clientY, deltaY, shiftKey, event) => boolean

    // Velocity / damping system
    this.orbitVelX = 0;   // yaw delta per frame
    this.orbitVelY = 0;   // pitch delta per frame
    this.panVelX = 0;
    this.panVelY = 0;
    this.zoomVel = 0;
    this.damping = 0.85;  // velocity decay (lower = faster stop)

    // Elevation clamp
    this.maxPitch = 30 * (Math.PI / 180); // ±30 degrees

    this._bindEvents();
  }

  getViewMatrix() {
    return [...this.viewMatrix];
  }

  setViewMatrix(m) {
    this.viewMatrix = [...m];
  }

  /** Extract camera pitch angle from current view matrix */
  _extractPitch() {
    const inv = invert4(this.viewMatrix);
    if (!inv) return 0;
    // Forward vector in world space = -Z column of inverse view
    const fy = -inv[9]; // forward.y (inv[8]=fwd.x, inv[9]=fwd.y, inv[10]=fwd.z, negated)
    return Math.asin(Math.max(-1, Math.min(1, fy)));
  }

  /** Approximate camera distance from world origin */
  _getCameraDistance() {
    const inv = invert4(this.viewMatrix);
    if (!inv) return 4;
    return Math.sqrt(inv[12] * inv[12] + inv[13] * inv[13] + inv[14] * inv[14]);
  }

  _bindEvents() {
    // ── Keyboard ─────────────────────────────────────────────
    window.addEventListener("keydown", (e) => {
      if (!this.enabled) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (!this.activeKeys.includes(e.code)) this.activeKeys.push(e.code);
    });
    window.addEventListener("keyup", (e) => {
      this.activeKeys = this.activeKeys.filter(k => k !== e.code);
    });
    // Clear all keys when an input gains focus (prevents stuck keys)
    window.addEventListener("focusin", (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        this.activeKeys = [];
      }
    });
    window.addEventListener("blur", () => { this.activeKeys = []; });

    // ── Wheel ────────────────────────────────────────────────
    this.canvas.addEventListener("wheel", (e) => {
      if (!this.enabled) return;
      if (e.target.closest('.panel')) return;
      e.preventDefault();
      if (this.onWheel && this.onWheel(e.clientX, e.clientY, e.deltaY, e.shiftKey, e)) return;
      const scale = e.deltaMode == 1 ? 10 : e.deltaMode == 2 ? innerHeight : 1;
      if (e.shiftKey) {
        // Shift+scroll → pan
        this.panVelX += (e.deltaX * scale) / innerWidth;
        this.panVelY += (e.deltaY * scale) / innerHeight;
      } else {
        // Distance-based zoom speed
        const dist = this._getCameraDistance();
        const zoomAmount = (-10 * (e.deltaY * scale)) / innerHeight;
        this.zoomVel += zoomAmount * Math.max(0.3, dist * 0.25);
        this.zoomVel = Math.max(-3, Math.min(3, this.zoomVel));
      }
    }, { passive: false });

    // ── Mouse ────────────────────────────────────────────────
    this.canvas.addEventListener("mousedown", (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      this.startX = e.clientX;
      this.startY = e.clientY;

      if (e.button === 0 && this.onLeftClick) {
        const consumed = this.onLeftClick(e.clientX, e.clientY, e);
        if (consumed) {
          this.down = 0; // No camera drag
          return;
        }
        // Fall through to normal orbit
      }

      if (e.button === 2 && this.onRightClick) {
        if (this.onRightClick(e.clientX, e.clientY, e)) {
          this.down = 0;
          return;
        }
      }

      if (e.button === 2 || e.shiftKey) {
        this.down = 2; // Pan
      } else {
        this.down = 1; // Orbit
      }
    });

    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.canvas.addEventListener("mousemove", (e) => {
      if (!this.enabled || !this.down) return;
      e.preventDefault();
      if (this.down == 1) {
        // Store orbit velocity (applied in update())
        this.orbitVelX = (8 * (e.clientX - this.startX)) / innerWidth;
        this.orbitVelY = (8 * (e.clientY - this.startY)) / innerHeight;
        this.startX = e.clientX;
        this.startY = e.clientY;
      } else if (this.down == 2) {
        // Store pan velocity
        const dist = this._getCameraDistance();
        const panScale = Math.max(1, dist * 0.5);
        this.panVelX = (-10 * (e.clientX - this.startX)) / innerWidth * panScale;
        this.panVelY = (-10 * (e.clientY - this.startY)) / innerHeight * panScale;
        this.startX = e.clientX;
        this.startY = e.clientY;
      }
    });

    this.canvas.addEventListener("mouseup", (e) => {
      e.preventDefault();
      this.down = false;
      this.startX = 0;
      this.startY = 0;
    });

    // ── Touch ────────────────────────────────────────────────
    this.canvas.addEventListener("touchstart", (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      if (e.touches.length === 1) {
        this.startX = e.touches[0].clientX;
        this.startY = e.touches[0].clientY;
        this.down = 1;
      } else if (e.touches.length === 2) {
        this.startX = e.touches[0].clientX;
        this.altX = e.touches[1].clientX;
        this.startY = e.touches[0].clientY;
        this.altY = e.touches[1].clientY;
        this.down = 1;
      }
    }, { passive: false });

    this.canvas.addEventListener("touchmove", (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      if (e.touches.length === 1 && this.down) {
        // Single-touch orbit → velocity
        this.orbitVelX = (4 * (e.touches[0].clientX - this.startX)) / innerWidth;
        this.orbitVelY = (4 * (e.touches[0].clientY - this.startY)) / innerHeight;
        this.startX = e.touches[0].clientX;
        this.startY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        const dtheta =
          Math.atan2(this.startY - this.altY, this.startX - this.altX) -
          Math.atan2(e.touches[0].clientY - e.touches[1].clientY, e.touches[0].clientX - e.touches[1].clientX);
        const dscale =
          Math.hypot(this.startX - this.altX, this.startY - this.altY) /
          Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        const dx = (e.touches[0].clientX + e.touches[1].clientX - (this.startX + this.altX)) / 2;
        const dy = (e.touches[0].clientY + e.touches[1].clientY - (this.startY + this.altY)) / 2;
        let inv = invert4(this.viewMatrix);
        inv = rotate4(inv, dtheta, 0, 0, 1);
        inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);
        inv = translate4(inv, 0, 0, 3 * (1 - dscale));
        this.viewMatrix = invert4(inv);
        this.startX = e.touches[0].clientX;
        this.altX = e.touches[1].clientX;
        this.startY = e.touches[0].clientY;
        this.altY = e.touches[1].clientY;
      }
    }, { passive: false });

    this.canvas.addEventListener("touchend", (e) => {
      e.preventDefault();
      this.down = false;
      this.startX = 0;
      this.startY = 0;
    }, { passive: false });
  }

  /**
   * Per-frame update: apply velocities with damping, process WASD + jump.
   * Call this every frame before rendering.
   * @returns {number[]} actualViewMatrix with jump offset applied
   */
  update() {
    if (!this.enabled) return [...this.viewMatrix];

    const EPS = 0.0001;

    // ── Apply orbit velocity (with elevation clamp) ──────────
    if (Math.abs(this.orbitVelX) > EPS || Math.abs(this.orbitVelY) > EPS) {
      let inv = invert4(this.viewMatrix);
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, this.orbitVelX, 0, 1, 0);

      // Clamp pitch to ±maxPitch
      const currentPitch = this._extractPitch();
      let pitchDelta = -this.orbitVelY;
      const newPitch = currentPitch + pitchDelta;
      if (newPitch > this.maxPitch) {
        pitchDelta = this.maxPitch - currentPitch;
      } else if (newPitch < -this.maxPitch) {
        pitchDelta = -this.maxPitch - currentPitch;
      }
      if (Math.abs(pitchDelta) > EPS) {
        inv = rotate4(inv, pitchDelta, 1, 0, 0);
      }

      inv = translate4(inv, 0, 0, -d);
      this.viewMatrix = invert4(inv);
    }

    // ── Apply pan velocity ───────────────────────────────────
    if (Math.abs(this.panVelX) > EPS || Math.abs(this.panVelY) > EPS) {
      let inv = invert4(this.viewMatrix);
      inv = translate4(inv, this.panVelX, this.panVelY, 0);
      this.viewMatrix = invert4(inv);
    }

    // ── Apply zoom velocity ──────────────────────────────────
    if (Math.abs(this.zoomVel) > 0.001) {
      let inv = invert4(this.viewMatrix);
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = translate4(inv, 0, 0, this.zoomVel);
      inv = translate4(inv, 0, 0, -d);
      this.viewMatrix = invert4(inv);
    }

    // ── Damping ──────────────────────────────────────────────
    if (this.down === 1) {
      // Dragging orbit → zero velocity after applying (fresh each frame from mousemove)
      this.orbitVelX = 0;
      this.orbitVelY = 0;
    } else if (this.down === 2) {
      // Dragging pan
      this.panVelX = 0;
      this.panVelY = 0;
    } else {
      // Not dragging → decay all velocities (inertia)
      this.orbitVelX *= this.damping;
      this.orbitVelY *= this.damping;
      this.panVelX *= this.damping;
      this.panVelY *= this.damping;
    }
    this.zoomVel *= this.damping;

    // ── WASD / Arrow keys (distance-based speed) ─────────────
    let inv = invert4(this.viewMatrix);
    const shiftKey = this.activeKeys.includes("ShiftLeft") || this.activeKeys.includes("ShiftRight");
    const dist = this._getCameraDistance();
    const moveSpeed = Math.max(0.02, dist * 0.025);
    const strafeSpeed = Math.max(0.01, dist * 0.015);
    const vertSpeed = Math.max(0.01, dist * 0.02);

    if (this.activeKeys.includes("ArrowUp")) {
      inv = shiftKey ? translate4(inv, 0, -vertSpeed, 0) : translate4(inv, 0, 0, moveSpeed);
    }
    if (this.activeKeys.includes("ArrowDown")) {
      inv = shiftKey ? translate4(inv, 0, vertSpeed, 0) : translate4(inv, 0, 0, -moveSpeed);
    }
    if (this.activeKeys.includes("ArrowLeft")) inv = translate4(inv, -strafeSpeed, 0, 0);
    if (this.activeKeys.includes("ArrowRight")) inv = translate4(inv, strafeSpeed, 0, 0);

    // WASD movement
    if (this.activeKeys.includes("KeyW")) inv = translate4(inv, 0, 0, moveSpeed);
    if (this.activeKeys.includes("KeyS")) inv = translate4(inv, 0, 0, -moveSpeed);
    if (this.activeKeys.includes("KeyA")) inv = translate4(inv, -strafeSpeed, 0, 0);
    if (this.activeKeys.includes("KeyD")) inv = translate4(inv, strafeSpeed, 0, 0);
    if (this.activeKeys.includes("KeyE")) inv = translate4(inv, 0, -vertSpeed, 0);
    if (this.activeKeys.includes("KeyQ")) inv = translate4(inv, 0, vertSpeed, 0);

    // IJKL rotation
    if (["KeyJ", "KeyK", "KeyL", "KeyI"].some(k => this.activeKeys.includes(k))) {
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, this.activeKeys.includes("KeyJ") ? -0.05 : this.activeKeys.includes("KeyL") ? 0.05 : 0, 0, 1, 0);
      inv = rotate4(inv, this.activeKeys.includes("KeyI") ? 0.05 : this.activeKeys.includes("KeyK") ? -0.05 : 0, 1, 0, 0);
      inv = translate4(inv, 0, 0, -d);
    }

    this.viewMatrix = invert4(inv);

    // ── Universal pitch clamp (catches IJKL, touch bypass) ──
    {
      const pitch = this._extractPitch();
      if (Math.abs(pitch) > this.maxPitch) {
        const excess = pitch > 0 ? pitch - this.maxPitch : pitch + this.maxPitch;
        let ci = invert4(this.viewMatrix);
        ci = translate4(ci, 0, 0, 4);
        ci = rotate4(ci, -excess, 1, 0, 0);
        ci = translate4(ci, 0, 0, -4);
        this.viewMatrix = invert4(ci);
      }
    }

    // Jump
    const isJumping = this.activeKeys.includes("Space");
    this.jumpDelta = isJumping ? Math.min(1, this.jumpDelta + 0.05) : Math.max(0, this.jumpDelta - 0.05);

    let inv2 = invert4(this.viewMatrix);
    inv2 = translate4(inv2, 0, -this.jumpDelta, 0);
    inv2 = rotate4(inv2, -0.1 * this.jumpDelta, 1, 0, 0);
    return invert4(inv2);
  }
}
