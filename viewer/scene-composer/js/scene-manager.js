/**
 * Scene Manager — Multi-layer CRUD, transform, and global merge.
 * Each layer stores original data + current transform.
 * On rebuild, all visible layers are baked and merged into one GPU upload.
 */

import { bakeTransform } from './utils/splat-transform.js';

let nextId = 1;

class Layer {
  constructor(name, texdata, positions, count, texwidth, texheight, type = 'object') {
    this.id = nextId++;
    this.name = name;
    this.type = type; // 'map' | 'object'
    this.originalTexdata = texdata;
    this.originalPositions = positions;
    this.count = count;
    this.texwidth = texwidth;
    this.texheight = texheight;

    // Transform
    this.position = [0, 0, 0];
    this.rotation = [0, 0, 0]; // Euler degrees (ZYX)
    this.scale = 1.0;

    // State
    this.visible = true;
    this.locked = type === 'map'; // map is auto-locked

    // Baked data (initially = original, no transform)
    this.bakedTexdata = new Uint32Array(texdata);
    this.bakedPositions = new Float32Array(positions);
  }

  bake() {
    const result = bakeTransform(
      this.originalTexdata,
      this.originalPositions,
      this.count,
      { position: this.position, rotation: this.rotation, scale: this.scale }
    );
    this.bakedTexdata = result.texdata;
    this.bakedPositions = result.positions;
    this._boundsCached = false; // invalidate for ray-cast picking
  }

  getTransform() {
    return {
      position: [...this.position],
      rotation: [...this.rotation],
      scale: this.scale,
    };
  }
}

export class SceneManager {
  constructor(renderer) {
    this.renderer = renderer;
    this.layers = [];
    this.selectedLayerId = null;
    this.gaussianToLayer = null; // Uint32Array mapping gaussian index → layer id
    this.totalGaussians = 0;

    // Callbacks
    this.onChange = null; // (eventType, data) => {}
  }

  /**
   * Add a new layer from parsed splat data.
   * @returns {number} layer id
   */
  addLayer(name, texdata, positions, count, texwidth, texheight, type = 'object') {
    const layer = new Layer(name, texdata, positions, count, texwidth, texheight, type);
    this.layers.push(layer);
    this.rebuild();
    if (this.onChange) this.onChange('addLayer', layer);
    return layer.id;
  }

  removeLayer(id) {
    const idx = this.layers.findIndex(l => l.id === id);
    if (idx === -1) return;
    const layer = this.layers[idx];
    this.layers.splice(idx, 1);
    if (this.selectedLayerId === id) this.selectedLayerId = null;
    this.rebuild();
    if (this.onChange) this.onChange('removeLayer', layer);
  }

  getLayer(id) {
    return this.layers.find(l => l.id === id) || null;
  }

  getLayers() {
    return this.layers;
  }

  getSelectableLayers() {
    return this.layers.filter(l => l.type === 'object' && l.visible && !l.locked);
  }

  hasMap() {
    return this.layers.some(l => l.type === 'map');
  }

  getSelectedLayer() {
    return this.selectedLayerId ? this.getLayer(this.selectedLayerId) : null;
  }

  selectLayer(id) {
    this.selectedLayerId = id;
    if (this.onChange) this.onChange('select', id);
  }

  // ── Transform setters ──────────────────────────────────────────────

  setPosition(id, pos) {
    const layer = this.getLayer(id);
    if (!layer || layer.locked) return;
    layer.position = [pos[0], pos[1], pos[2]];
    layer.bake();
    this.rebuild();
    if (this.onChange) this.onChange('transform', layer);
  }

  setRotation(id, rot) {
    const layer = this.getLayer(id);
    if (!layer || layer.locked) return;
    layer.rotation = [rot[0], rot[1], rot[2]];
    layer.bake();
    this.rebuild();
    if (this.onChange) this.onChange('transform', layer);
  }

  setScale(id, s) {
    const layer = this.getLayer(id);
    if (!layer || layer.locked) return;
    layer.scale = Math.max(0.01, Math.min(100.0, s));
    layer.bake();
    this.rebuild();
    if (this.onChange) this.onChange('transform', layer);
  }

  setTransform(id, transform) {
    const layer = this.getLayer(id);
    if (!layer || layer.locked) return;
    if (transform.position) layer.position = [...transform.position];
    if (transform.rotation) layer.rotation = [...transform.rotation];
    if (transform.scale != null) layer.scale = Math.max(0.01, Math.min(100.0, transform.scale));
    layer.bake();
    this.rebuild();
    if (this.onChange) this.onChange('transform', layer);
  }

  setVisible(id, visible) {
    const layer = this.getLayer(id);
    if (!layer) return;
    layer.visible = visible;
    this.rebuild();
    if (this.onChange) this.onChange('visibility', layer);
  }

  setLocked(id, locked) {
    const layer = this.getLayer(id);
    if (!layer) return;
    layer.locked = locked;
    if (this.onChange) this.onChange('lock', layer);
  }

  // ── Gaussian → Layer lookup ────────────────────────────────────────

  /**
   * Find which layer a gaussian belongs to.
   * @param {number} gaussianIndex - Index in the merged buffer
   * @returns {number|null} layer id, or null
   */
  getLayerByGaussian(gaussianIndex) {
    if (!this.gaussianToLayer || gaussianIndex >= this.gaussianToLayer.length) return null;
    return this.gaussianToLayer[gaussianIndex];
  }

  // ── Merge & Upload ─────────────────────────────────────────────────

  /**
   * Merge all visible layers into a single texdata and upload to renderer.
   */
  rebuild() {
    const visibleLayers = this.layers.filter(l => l.visible);

    if (visibleLayers.length === 0) {
      this.totalGaussians = 0;
      this.gaussianToLayer = null;
      const texwidth = 1024 * 4;
      const texdata = new Uint32Array(texwidth * 4);
      this.renderer.uploadScene(texdata, texwidth, 1, 0);
      return;
    }

    // Total count
    let totalCount = 0;
    for (const layer of visibleLayers) {
      totalCount += layer.count;
    }

    const texwidth = 1024 * 4;
    const texheight = Math.ceil((4 * totalCount) / texwidth);
    const merged = new Uint32Array(texwidth * texheight * 4);
    this.gaussianToLayer = new Uint32Array(totalCount);

    let offset = 0;
    for (const layer of visibleLayers) {
      // Copy baked texdata (16 uint32 per gaussian)
      const srcLen = layer.count * 16;
      const dstStart = offset * 16;
      merged.set(layer.bakedTexdata.subarray(0, srcLen), dstStart);

      // Map gaussians to layer id
      for (let j = 0; j < layer.count; j++) {
        this.gaussianToLayer[offset + j] = layer.id;
      }

      offset += layer.count;
    }

    this.totalGaussians = totalCount;
    this.renderer.uploadScene(merged, texwidth, texheight, totalCount);
  }
}
