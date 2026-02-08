/**
 * Object Animator — Manages path animations for multiple objects.
 * Calculates path positions and updates renderer uniforms each frame.
 */

// Maximum number of animated objects (must match shader)
const MAX_OBJECTS = 12;

export class ObjectAnimator {
  constructor(renderer, sceneManager) {
    this.renderer = renderer;
    this.sceneManager = sceneManager;
    this.animatedObjects = [];
    this.isPlaying = true;
    this.globalSpeedMultiplier = 1.0;

    // Bind update method for use in render loop
    this.update = this.update.bind(this);
  }

  /**
   * Initialize or update animation for a layer with path data.
   * @param {number} layerId - Layer ID
   * @param {Object} pathData - Path control points and settings
   */
  setLayerPath(layerId, pathData) {
    const layer = this.sceneManager.getLayer(layerId);
    if (!layer) return;

    // Find existing animation or create new one
    let animObj = this.animatedObjects.find(o => o.layerId === layerId);
    if (!animObj) {
      animObj = this._createAnimatedObject(layer, pathData);
      this.animatedObjects.push(animObj);
    } else {
      // Update existing
      animObj.path = pathData;
      this._initializePath(animObj);
    }

    // Store path reference in layer for export
    layer.pathData = pathData;

    console.log(`[ObjectAnimator] Layer "${layer.name}" path set with ${pathData.controlPoints?.length || 0} points`);
  }

  /**
   * Remove animation from a layer.
   */
  removeLayerPath(layerId) {
    const idx = this.animatedObjects.findIndex(o => o.layerId === layerId);
    if (idx >= 0) {
      this.animatedObjects.splice(idx, 1);
    }
    const layer = this.sceneManager.getLayer(layerId);
    if (layer) {
      layer.pathData = null;
    }
  }

  /**
   * Get animation object for a layer.
   */
  getLayerAnimation(layerId) {
    return this.animatedObjects.find(o => o.layerId === layerId);
  }

  /**
   * Set path speed for a specific layer.
   */
  setPathSpeed(layerId, speed) {
    const animObj = this.animatedObjects.find(o => o.layerId === layerId);
    if (animObj) {
      animObj.pathSpeed = Math.max(0.01, Math.min(5.0, speed));
    }
  }

  /**
   * Set walk animation speed for a specific layer.
   */
  setWalkSpeed(layerId, speed) {
    const animObj = this.animatedObjects.find(o => o.layerId === layerId);
    if (animObj) {
      animObj.walkSpeed = Math.max(0.0, Math.min(3.0, speed));
    }
  }

  /**
   * Set height offset for a specific layer.
   */
  setHeightOffset(layerId, offset) {
    const animObj = this.animatedObjects.find(o => o.layerId === layerId);
    if (animObj) {
      animObj.heightOffset = Math.max(-10.0, Math.min(10.0, offset));
    }
  }

  /**
   * Toggle play/pause.
   */
  togglePlay() {
    this.isPlaying = !this.isPlaying;
    const now = performance.now();
    if (this.isPlaying) {
      // Resume: reset lastUpdateTime to now
      for (const obj of this.animatedObjects) {
        obj.lastUpdateTime = now;
      }
    }
    // When pausing, currentT is already stored, no action needed
    return this.isPlaying;
  }

  /**
   * Reset all animations to start.
   */
  reset() {
    for (const obj of this.animatedObjects) {
      obj.currentT = 0;
      obj.direction = 1;
      obj.lastUpdateTime = undefined;
      obj.currentOffset = [0, 0, 0];
      obj.currentYaw = 0;
    }
  }

  /**
   * Update all animations and send to renderer.
   * Called each frame from main render loop.
   */
  update() {
    if (this.animatedObjects.length === 0) {
      this.renderer.updateObjectAnimations([]);
      return;
    }

    const now = performance.now();
    const animData = [];

    for (const obj of this.animatedObjects) {
      // Get layer's current merged index range
      const layer = this.sceneManager.getLayer(obj.layerId);
      if (!layer || !layer.visible) continue;

      // Calculate start/end indices in merged buffer
      const layerInfo = this._getLayerIndexRange(layer);
      if (!layerInfo) continue;

      // Calculate animation state
      if (this.isPlaying && obj.samples && obj.samples.length > 0) {
        if (obj.lastUpdateTime === undefined) {
          obj.lastUpdateTime = now;
        }

        // Calculate delta time and update t
        const deltaTime = (now - obj.lastUpdateTime) * obj.pathSpeed;
        obj.lastUpdateTime = now;

        // Update t based on direction
        const deltaT = (deltaTime / obj.duration) * obj.direction;
        obj.currentT = (obj.currentT || 0) + deltaT;

        // Handle ping-pong or loop
        if (obj.pingPong) {
          // Ping-pong: reverse direction at endpoints
          if (obj.currentT >= 1.0) {
            obj.currentT = 1.0;
            obj.direction = -1;
          } else if (obj.currentT <= 0.0) {
            obj.currentT = 0.0;
            obj.direction = 1;
          }
        } else if (obj.loop) {
          // Regular loop: wrap around
          obj.currentT = obj.currentT % 1.0;
          if (obj.currentT < 0) obj.currentT += 1.0;
        } else {
          // No loop: clamp
          obj.currentT = Math.max(0, Math.min(1, obj.currentT));
        }

        const pathState = this._interpolatePath(obj, obj.currentT, obj.direction);
        obj.currentOffset = pathState.offset;
        obj.currentYaw = pathState.yaw;
      }

      animData.push({
        startIndex: layerInfo.startIndex,
        endIndex: layerInfo.endIndex,
        center: obj.center,
        offset: obj.currentOffset,
        yaw: obj.currentYaw
      });
    }

    this.renderer.updateObjectAnimations(animData);
  }

  // ── Private Methods ────────────────────────────────────────────────

  _createAnimatedObject(layer, pathData) {
    const obj = {
      layerId: layer.id,
      name: layer.name,
      path: pathData,
      // Animation settings
      pathSpeed: pathData?.settings?.pathSpeed || 0.5,
      walkSpeed: pathData?.settings?.walkSpeed || 1.0,
      heightOffset: pathData?.settings?.heightOffset || 0.0,  // Vertical offset (Y-axis)
      duration: (pathData?.settings?.duration || 10) * 1000, // Convert to ms
      loop: true,
      pingPong: true,         // Enable ping-pong (back and forth) animation
      // State
      startTime: null,
      samples: null,
      yawAngles: null,
      center: [0, 0, 0],      // Object's bounding box center
      pathStart: [0, 0, 0],   // Path starting point
      currentOffset: [0, 0, 0],
      currentYaw: 0,
      direction: 1,           // 1 = forward, -1 = backward
      currentT: 0             // Current position on path [0, 1]
    };

    this._initializePath(obj);
    return obj;
  }

  _initializePath(obj) {
    const pathData = obj.path;
    if (!pathData || !pathData.controlPoints || pathData.controlPoints.length < 2) {
      obj.samples = null;
      obj.yawAngles = null;
      return;
    }

    // Extract control point positions
    const controlPoints = pathData.controlPoints.map(cp =>
      Array.isArray(cp) ? cp : cp.position
    );

    // Sample path using Catmull-Rom spline
    const numSamples = 100;
    obj.samples = this._interpolateCatmullRom(controlPoints, numSamples);
    obj.yawAngles = this._computeYawAngles(obj.samples);

    // Calculate center of the OBJECT (from layer's gaussians), not path start
    const layer = this.sceneManager.getLayer(obj.layerId);
    if (layer && layer.bakedPositions) {
      const positions = layer.bakedPositions;
      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < layer.count; i++) {
        cx += positions[3 * i];
        cy += positions[3 * i + 1];
        cz += positions[3 * i + 2];
      }
      obj.center = [cx / layer.count, cy / layer.count, cz / layer.count];
    } else {
      // Fallback to path start if layer not available
      obj.center = [...obj.samples[0]];
    }

    // Store path start for offset calculation
    obj.pathStart = [...obj.samples[0]];

    // Update duration from settings
    if (pathData.settings?.duration) {
      let dur = pathData.settings.duration;
      if (dur < 100) dur *= 1000; // Convert seconds to ms
      obj.duration = dur;
    }

    obj.currentT = 0;
    obj.direction = 1;
    obj.lastUpdateTime = undefined;
    obj.currentOffset = [0, 0, 0];
    obj.currentYaw = 0;

    console.log(`[ObjectAnimator] Path initialized: ${controlPoints.length} control points, ${obj.samples.length} samples`);
    console.log(`[ObjectAnimator] Object center: [${obj.center.map(v => v.toFixed(2)).join(', ')}], Path start: [${obj.pathStart.map(v => v.toFixed(2)).join(', ')}]`);
  }

  _interpolatePath(obj, t, direction = 1) {
    const samples = obj.samples;
    const yawAngles = obj.yawAngles;
    const objectCenter = obj.center;

    t = Math.max(0, Math.min(1, t));

    const idx = t * (samples.length - 1);
    const i = Math.floor(idx);
    const alpha = idx - i;

    let position, yaw;

    if (i >= samples.length - 1) {
      position = samples[samples.length - 1];
      yaw = yawAngles[yawAngles.length - 1];
    } else {
      const p0 = samples[i];
      const p1 = samples[i + 1];
      position = [
        p0[0] + alpha * (p1[0] - p0[0]),
        p0[1] + alpha * (p1[1] - p0[1]),
        p0[2] + alpha * (p1[2] - p0[2])
      ];
      yaw = yawAngles[i] + alpha * (yawAngles[i + 1] - yawAngles[i]);
    }

    // When moving backward, rotate 180 degrees to face the opposite direction
    if (direction < 0) {
      yaw += Math.PI;
    }

    // Offset = path position - object center + height offset on Y-axis
    // This moves the object from its original position to the path position
    const heightOffset = obj.heightOffset || 0;
    return {
      offset: [
        position[0] - objectCenter[0],
        position[1] - objectCenter[1] + heightOffset,
        position[2] - objectCenter[2]
      ],
      yaw: yaw
    };
  }

  _interpolateCatmullRom(points, numSamples) {
    if (points.length < 2) return points;

    const samples = [];
    const n = points.length;

    for (let i = 0; i < numSamples; i++) {
      const t = i / (numSamples - 1);
      const totalT = t * (n - 1);
      const segment = Math.floor(totalT);
      const localT = totalT - segment;

      // Get 4 control points for Catmull-Rom
      const p0 = points[Math.max(0, segment - 1)];
      const p1 = points[segment];
      const p2 = points[Math.min(n - 1, segment + 1)];
      const p3 = points[Math.min(n - 1, segment + 2)];

      // Catmull-Rom interpolation
      const t2 = localT * localT;
      const t3 = t2 * localT;

      const sample = [0, 0, 0];
      for (let j = 0; j < 3; j++) {
        sample[j] = 0.5 * (
          (2 * p1[j]) +
          (-p0[j] + p2[j]) * localT +
          (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
          (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3
        );
      }
      samples.push(sample);
    }

    return samples;
  }

  _computeYawAngles(samples) {
    const yawAngles = [];
    for (let i = 0; i < samples.length; i++) {
      let dx, dz;
      if (i < samples.length - 1) {
        dx = samples[i + 1][0] - samples[i][0];
        dz = samples[i + 1][2] - samples[i][2];
      } else {
        dx = samples[i][0] - samples[i - 1][0];
        dz = samples[i][2] - samples[i - 1][2];
      }
      // Yaw angle: rotation around Y axis to face movement direction
      // atan2(dx, dz) gives angle from +Z axis
      yawAngles.push(Math.atan2(dx, dz) - Math.PI / 2);
    }
    return yawAngles;
  }

  _getLayerIndexRange(layer) {
    // Find this layer's position in the merged buffer
    const layers = this.sceneManager.layers.filter(l => l.visible);
    let startIndex = 0;

    for (const l of layers) {
      if (l.id === layer.id) {
        return {
          startIndex: startIndex,
          endIndex: startIndex + layer.count
        };
      }
      startIndex += l.count;
    }

    return null;
  }
}
