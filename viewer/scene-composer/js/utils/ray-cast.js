/**
 * Ray casting utilities for gizmo hit testing and layer selection.
 */

import { multiply4, invert4 } from './matrix-math.js';

// ── Vector helpers ──────────────────────────────────────────────────

function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function scale(v, s) { return [v[0]*s, v[1]*s, v[2]*s]; }
function length(v) { return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
function normalize(v) { const l = length(v); return l > 1e-10 ? [v[0]/l, v[1]/l, v[2]/l] : [0,0,0]; }

// ── Screen to Ray ───────────────────────────────────────────────────

/**
 * Construct a world-space ray from screen coordinates.
 * @param {number} clientX
 * @param {number} clientY
 * @param {number[]} viewMatrix - 4x4 column-major
 * @param {number[]} projMatrix - 4x4 column-major
 * @param {number} width - canvas width
 * @param {number} height - canvas height
 * @returns {{ origin: number[], direction: number[] }}
 */
export function screenToRay(clientX, clientY, viewMatrix, projMatrix, width, height) {
  const vp = multiply4(projMatrix, viewMatrix);
  const inv = invert4(vp);
  if (!inv) return { origin: [0,0,0], direction: [0,0,-1] };

  // NDC coords (consistent with projectToScreen in matrix-math.js)
  const ndcX = (clientX / width) * 2 - 1;
  const ndcY = 1 - (clientY / height) * 2;

  // Unproject near (z=0) and far (z=1) points
  const near = unproject(inv, ndcX, ndcY, 0);
  const far = unproject(inv, ndcX, ndcY, 1);

  const dir = normalize(sub(far, near));
  return { origin: near, direction: dir };
}

function unproject(invVP, ndcX, ndcY, ndcZ) {
  const x = invVP[0]*ndcX + invVP[4]*ndcY + invVP[8]*ndcZ + invVP[12];
  const y = invVP[1]*ndcX + invVP[5]*ndcY + invVP[9]*ndcZ + invVP[13];
  const z = invVP[2]*ndcX + invVP[6]*ndcY + invVP[10]*ndcZ + invVP[14];
  const w = invVP[3]*ndcX + invVP[7]*ndcY + invVP[11]*ndcZ + invVP[15];
  if (Math.abs(w) < 1e-10) return [0,0,0];
  return [x/w, y/w, z/w];
}

// ── Layer Picking ───────────────────────────────────────────────────

/**
 * Compute bounding sphere for a layer's baked positions.
 * Caches result on the layer object.
 * @param {object} layer
 */
function ensureBounds(layer) {
  if (layer._boundsCached) return;
  const pos = layer.bakedPositions;
  const n = layer.count;
  if (n === 0) {
    layer._centroid = [0,0,0];
    layer._boundRadius = 0;
    layer._boundsCached = true;
    return;
  }
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) {
    cx += pos[i*3]; cy += pos[i*3+1]; cz += pos[i*3+2];
  }
  cx /= n; cy /= n; cz /= n;
  let maxR2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = pos[i*3]-cx, dy = pos[i*3+1]-cy, dz = pos[i*3+2]-cz;
    const r2 = dx*dx + dy*dy + dz*dz;
    if (r2 > maxR2) maxR2 = r2;
  }
  layer._centroid = [cx, cy, cz];
  layer._boundRadius = Math.sqrt(maxR2);
  layer._boundsCached = true;
}

/**
 * Invalidate cached bounds (call after bake).
 */
export function invalidateBounds(layer) {
  layer._boundsCached = false;
}

/**
 * Pick a layer by ray-sphere intersection.
 * @returns {{ layerId: number, distance: number } | null}
 */
export function rayVsLayerBounds(ray, sceneManager) {
  let bestHit = null;
  for (const layer of sceneManager.layers) {
    if (!layer.visible || layer.count === 0 || layer.type === 'map' || layer.locked) continue;
    ensureBounds(layer);
    const t = raySphereIntersect(ray, layer._centroid, layer._boundRadius);
    if (t !== null && (bestHit === null || t < bestHit.distance)) {
      bestHit = { layerId: layer.id, distance: t };
    }
  }
  return bestHit;
}

function raySphereIntersect(ray, center, radius) {
  const oc = sub(ray.origin, center);
  const b = dot(oc, ray.direction);
  const c = dot(oc, oc) - radius * radius;
  const disc = b*b - c;
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc);
  return t > 0 ? t : (-b + Math.sqrt(disc) > 0 ? 0 : null);
}

// ── Gizmo Hit Tests ─────────────────────────────────────────────────

/**
 * Ray vs axis (line segment) — closest distance between ray and segment.
 * @param {object} ray - { origin, direction }
 * @param {number[]} axisStart - [x,y,z]
 * @param {number[]} axisEnd - [x,y,z]
 * @param {number} threshold - max distance for hit
 * @returns {{ hit: boolean, t: number, axisT: number } | null}
 */
export function rayVsAxis(ray, axisStart, axisEnd, threshold) {
  const dp = sub(axisEnd, axisStart);
  const dq = ray.direction;
  const r = sub(ray.origin, axisStart);

  const a = dot(dp, dp);
  const b = dot(dp, dq);
  const c = dot(dq, dq);
  const d = dot(dp, r);
  const e = dot(dq, r);

  const denom = a*c - b*b;
  if (Math.abs(denom) < 1e-10) return null;

  let s = (b*e - c*d) / denom;
  s = Math.max(0, Math.min(1, s));
  const t = (b*s + e) / c;
  if (t < 0) return null;

  const closestOnAxis = add(axisStart, scale(dp, s));
  const closestOnRay = add(ray.origin, scale(dq, t));
  const dist = length(sub(closestOnAxis, closestOnRay));

  if (dist < threshold) {
    return { hit: true, t, axisT: s };
  }
  return null;
}

/**
 * Ray vs ring (circle in 3D) for rotate gizmo.
 * @param {object} ray
 * @param {number[]} center
 * @param {number[]} normal - unit normal of the ring's plane
 * @param {number} radius
 * @param {number} threshold
 * @returns {{ hit: boolean, t: number } | null}
 */
export function rayVsRing(ray, center, normal, radius, threshold) {
  const denom = dot(ray.direction, normal);
  if (Math.abs(denom) < 1e-10) return null;

  const t = dot(sub(center, ray.origin), normal) / denom;
  if (t < 0) return null;

  const hitPoint = add(ray.origin, scale(ray.direction, t));
  const dist = length(sub(hitPoint, center));
  if (Math.abs(dist - radius) < threshold) {
    return { hit: true, t };
  }
  return null;
}

/**
 * Ray vs AABB for scale gizmo cube handles.
 * @param {object} ray
 * @param {number[]} cubeCenter
 * @param {number} halfSize
 * @returns {{ hit: boolean, t: number } | null}
 */
export function rayVsCube(ray, cubeCenter, halfSize) {
  const min = [cubeCenter[0]-halfSize, cubeCenter[1]-halfSize, cubeCenter[2]-halfSize];
  const max = [cubeCenter[0]+halfSize, cubeCenter[1]+halfSize, cubeCenter[2]+halfSize];

  let tmin = -Infinity, tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(ray.direction[i]) < 1e-10) {
      if (ray.origin[i] < min[i] || ray.origin[i] > max[i]) return null;
    } else {
      let t1 = (min[i] - ray.origin[i]) / ray.direction[i];
      let t2 = (max[i] - ray.origin[i]) / ray.direction[i];
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
  }
  if (tmax < 0) return null;
  return { hit: true, t: Math.max(0, tmin) };
}

/**
 * Find parameter t on an axis line closest to a ray.
 * Used for translate drag: how far along the axis the ray points.
 * @param {object} ray
 * @param {number[]} axisOrigin
 * @param {number[]} axisDir - unit direction
 * @returns {number} displacement along axis
 */
export function rayAxisClosestT(ray, axisOrigin, axisDir) {
  const w = sub(ray.origin, axisOrigin);
  const a = dot(axisDir, axisDir);
  const b = dot(axisDir, ray.direction);
  const c = dot(ray.direction, ray.direction);
  const d = dot(axisDir, w);
  const e = dot(ray.direction, w);
  const denom = a*c - b*b;
  if (Math.abs(denom) < 1e-10) return 0;
  return (b*e - c*d) / denom;
}
