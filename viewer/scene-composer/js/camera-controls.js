/**
 * Camera controls for 3DGS viewer.
 * Extracted from hybrid.js: orbit, pan, zoom, WASD, touch gestures.
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

    // Intercept callback: if set, left-click calls this instead of orbit
    // (used by Gizmo/scene-manager for object selection)
    this.onLeftClick = null;

    this._bindEvents();
  }

  getViewMatrix() {
    return [...this.viewMatrix];
  }

  setViewMatrix(m) {
    this.viewMatrix = [...m];
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
      const scale = e.deltaMode == 1 ? 10 : e.deltaMode == 2 ? innerHeight : 1;
      let inv = invert4(this.viewMatrix);
      if (e.shiftKey) {
        inv = translate4(inv, (e.deltaX * scale) / innerWidth, (e.deltaY * scale) / innerHeight, 0);
      } else {
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = translate4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight);
        inv = translate4(inv, 0, 0, -d);
      }
      this.viewMatrix = invert4(inv);
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
        let inv = invert4(this.viewMatrix);
        let dx = (8 * (e.clientX - this.startX)) / innerWidth;
        let dy = (8 * (e.clientY - this.startY)) / innerHeight;
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(inv, dx, 0, 1, 0);
        inv = rotate4(inv, -dy, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);
        this.viewMatrix = invert4(inv);
        this.startX = e.clientX;
        this.startY = e.clientY;
      } else if (this.down == 2) {
        let inv = invert4(this.viewMatrix);
        inv = translate4(inv, (-10 * (e.clientX - this.startX)) / innerWidth, (-10 * (e.clientY - this.startY)) / innerHeight, 0);
        this.viewMatrix = invert4(inv);
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
        let inv = invert4(this.viewMatrix);
        let dx = (4 * (e.touches[0].clientX - this.startX)) / innerWidth;
        let dy = (4 * (e.touches[0].clientY - this.startY)) / innerHeight;
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(inv, dx, 0, 1, 0);
        inv = rotate4(inv, -dy, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);
        this.viewMatrix = invert4(inv);
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
   * Per-frame update: process WASD/Arrow keys + jump.
   * Call this every frame before rendering.
   * @returns {number[]} actualViewMatrix with jump offset applied
   */
  update() {
    if (!this.enabled) return [...this.viewMatrix];

    let inv = invert4(this.viewMatrix);
    const shiftKey = this.activeKeys.includes("ShiftLeft") || this.activeKeys.includes("ShiftRight");

    if (this.activeKeys.includes("ArrowUp")) {
      inv = shiftKey ? translate4(inv, 0, -0.03, 0) : translate4(inv, 0, 0, 0.1);
    }
    if (this.activeKeys.includes("ArrowDown")) {
      inv = shiftKey ? translate4(inv, 0, 0.03, 0) : translate4(inv, 0, 0, -0.1);
    }
    if (this.activeKeys.includes("ArrowLeft")) inv = translate4(inv, -0.03, 0, 0);
    if (this.activeKeys.includes("ArrowRight")) inv = translate4(inv, 0.03, 0, 0);

    // WASD movement
    if (this.activeKeys.includes("KeyW")) inv = translate4(inv, 0, 0, 0.1);
    if (this.activeKeys.includes("KeyS")) inv = translate4(inv, 0, 0, -0.1);
    if (this.activeKeys.includes("KeyA")) inv = translate4(inv, -0.05, 0, 0);
    if (this.activeKeys.includes("KeyD")) inv = translate4(inv, 0.05, 0, 0);
    if (this.activeKeys.includes("KeyE")) inv = translate4(inv, 0, -0.05, 0);
    if (this.activeKeys.includes("KeyQ")) inv = translate4(inv, 0, 0.05, 0);

    // IJKL rotation
    if (["KeyJ", "KeyK", "KeyL", "KeyI"].some(k => this.activeKeys.includes(k))) {
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, this.activeKeys.includes("KeyJ") ? -0.05 : this.activeKeys.includes("KeyL") ? 0.05 : 0, 0, 1, 0);
      inv = rotate4(inv, this.activeKeys.includes("KeyI") ? 0.05 : this.activeKeys.includes("KeyK") ? -0.05 : 0, 1, 0, 0);
      inv = translate4(inv, 0, 0, -d);
    }

    this.viewMatrix = invert4(inv);

    // Jump
    const isJumping = this.activeKeys.includes("Space");
    this.jumpDelta = isJumping ? Math.min(1, this.jumpDelta + 0.05) : Math.max(0, this.jumpDelta - 0.05);

    let inv2 = invert4(this.viewMatrix);
    inv2 = translate4(inv2, 0, -this.jumpDelta, 0);
    inv2 = rotate4(inv2, -0.1 * this.jumpDelta, 1, 0, 0);
    return invert4(inv2);
  }
}
