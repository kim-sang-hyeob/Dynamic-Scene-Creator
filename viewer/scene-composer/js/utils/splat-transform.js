/**
 * CPU-side transform baking for Gaussian Splat texdata.
 * Ported from merge_splat_files.py: rotation_matrix, position/scale baking.
 */

import { packHalf2x16 } from './splat-loader.js';

// ── Half-float unpacking ─────────────────────────────────────────────

function halfToFloat(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Denormalized
    let e = -14;
    let f = frac;
    while (!(f & 0x400)) { f <<= 1; e--; }
    f &= ~0x400;
    return (sign ? -1 : 1) * Math.pow(2, e) * (1 + f / 1024);
  }
  if (exp === 31) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function unpackHalf2x16(packed) {
  return [halfToFloat(packed & 0xffff), halfToFloat((packed >> 16) & 0xffff)];
}

// ── Rotation matrix (ZYX Euler, degrees) ─────────────────────────────

function rotationMatrix(rx, ry, rz) {
  const toRad = Math.PI / 180;
  rx *= toRad; ry *= toRad; rz *= toRad;

  const cx = Math.cos(rx), sx = Math.sin(rx);
  const cy = Math.cos(ry), sy = Math.sin(ry);
  const cz = Math.cos(rz), sz = Math.sin(rz);

  // R = Rz * Ry * Rx (ZYX order, same as merge_splat_files.py)
  return [
    cz * cy,                    cz * sy * sx - sz * cx,   cz * sy * cx + sz * sx,
    sz * cy,                    sz * sy * sx + cz * cx,   sz * sy * cx - cz * sx,
    -sy,                        cy * sx,                   cy * cx,
  ];
}

function mat3MulVec(m, x, y, z) {
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
}

// ── Bake transform ───────────────────────────────────────────────────

/**
 * Bake position/rotation/scale transform into texdata.
 * - Positions: R * pos * scale + offset
 * - Gaussian ellipsoid scales: multiplied by uniform scale
 * - Quaternion rotations: NOT modified (rotation only affects positions)
 *
 * @param {Uint32Array} srcTexdata - Original layer texdata
 * @param {Float32Array} srcPositions - Original positions (count * 3)
 * @param {number} count - Gaussian count
 * @param {{position: number[], rotation: number[], scale: number}} transform
 * @returns {{texdata: Uint32Array, positions: Float32Array}}
 */
export function bakeTransform(srcTexdata, srcPositions, count, transform) {
  const { position: offset, rotation, scale } = transform;

  const texdata = new Uint32Array(srcTexdata.length);
  texdata.set(srcTexdata);
  const texdata_f = new Float32Array(texdata.buffer);
  const positions = new Float32Array(count * 3);

  const hasRotation = rotation[0] !== 0 || rotation[1] !== 0 || rotation[2] !== 0;
  const hasScale = scale !== 1.0;
  const R = hasRotation ? rotationMatrix(rotation[0], rotation[1], rotation[2]) : null;

  for (let j = 0; j < count; j++) {
    let x = srcPositions[3 * j + 0];
    let y = srcPositions[3 * j + 1];
    let z = srcPositions[3 * j + 2];

    // Rotate (around layer origin)
    if (hasRotation) {
      [x, y, z] = mat3MulVec(R, x, y, z);
    }

    // Scale + translate
    x = x * scale + offset[0];
    y = y * scale + offset[1];
    z = z * scale + offset[2];

    // Write positions to texdata (float32 at uint32 indices 0,1,2)
    texdata_f[16 * j + 0] = x;
    texdata_f[16 * j + 1] = y;
    texdata_f[16 * j + 2] = z;

    positions[3 * j + 0] = x;
    positions[3 * j + 1] = y;
    positions[3 * j + 2] = z;

    // Scale gaussian ellipsoid sizes
    if (hasScale) {
      const [s0, s1] = unpackHalf2x16(srcTexdata[16 * j + 5]);
      const [s2] = unpackHalf2x16(srcTexdata[16 * j + 6]);
      texdata[16 * j + 5] = packHalf2x16(s0 * scale, s1 * scale);
      texdata[16 * j + 6] = packHalf2x16(s2 * scale, 0);
    }
  }

  return { texdata, positions };
}
