/**
 * Splat Exporter — Export merged scene as .splatv file with full metadata.
 * Includes layer information, transforms, camera path, and camera settings.
 */

/**
 * Export scene to .splatv format with metadata.
 * @param {SceneManager} sceneManager - Scene manager with layers
 * @param {CameraControls} controls - Camera controls for view matrix
 * @param {PathEditor} pathEditor - Path editor for camera path (optional)
 * @param {Object} options - Export options
 * @param {ObjectAnimator} options.objectAnimator - Object animator for layer paths
 * @returns {Blob} .splatv file as Blob
 */
export function exportSceneToSplatv(sceneManager, controls, pathEditor = null, options = {}) {
  const visibleLayers = sceneManager.layers.filter(l => l.visible);
  const objectAnimator = options.objectAnimator;

  if (visibleLayers.length === 0) {
    throw new Error('No visible layers to export');
  }

  // Calculate total gaussians and merged texdata
  let totalCount = 0;
  for (const layer of visibleLayers) {
    totalCount += layer.count;
  }

  const texwidth = 1024 * 4;
  const texheight = Math.ceil((4 * totalCount) / texwidth);
  const mergedTexdata = new Uint32Array(texwidth * texheight * 4);

  // Build layer metadata and merge texdata
  const layersMetadata = [];
  let offset = 0;

  for (const layer of visibleLayers) {
    const startIndex = offset;
    const endIndex = offset + layer.count;

    // Copy baked texdata
    const srcLen = layer.count * 16;
    const dstStart = offset * 16;
    mergedTexdata.set(layer.bakedTexdata.subarray(0, srcLen), dstStart);

    // Get animation data for this layer
    const animObj = objectAnimator?.getLayerAnimation(layer.id);
    const layerPath = animObj ? {
      controlPoints: animObj.path?.controlPoints || [],
      settings: {
        duration: (animObj.duration || 10000) / 1000, // Convert to seconds
        pathSpeed: animObj.pathSpeed || 0.5,
        walkSpeed: animObj.walkSpeed || 1.0
      }
    } : null;

    // Store layer metadata
    layersMetadata.push({
      name: layer.name,
      type: layer.type,
      count: layer.count,
      start_index: startIndex,
      end_index: endIndex,
      position: [...layer.position],
      rotation: [...layer.rotation],
      scale: layer.scale,
      path: layerPath, // Include path data if exists
    });

    offset += layer.count;
  }

  // Get camera info
  const viewMatrix = controls.getViewMatrix();
  const cameraInfo = extractCameraInfo(viewMatrix, controls);

  // Get camera path data if available (for camera animation, not object paths)
  let pathData = null;
  if (pathEditor && pathEditor.controlPoints && pathEditor.controlPoints.length > 0) {
    pathData = {
      points: pathEditor.controlPoints.map(p => ({
        position: [...p.position],
        target: p.target ? [...p.target] : null,
      })),
      duration: pathEditor.duration || 5.0,
      settings: {
        camDist: pathEditor.camDistance,
        camAzimuth: pathEditor.camAzimuth,
        camElevation: pathEditor.camElevation,
      }
    };
  }

  // Build metadata JSON
  const metadata = [{
    type: 'splat',
    version: '1.0',
    size: mergedTexdata.byteLength,
    texwidth: texwidth,
    texheight: texheight,
    gaussian_count: totalCount,

    // Camera settings
    cameras: [cameraInfo],

    // Layer information (for re-importing with transforms)
    layers: layersMetadata,

    // Camera path (if exists)
    path: pathData,

    // Export metadata
    exported_at: new Date().toISOString(),
    composer_version: '1.0.0',
  }];

  const jsonStr = JSON.stringify(metadata, null, 2);
  const jsonBytes = new TextEncoder().encode(jsonStr);

  // Build binary file
  // Format: magic (4 bytes) + json_length (4 bytes) + json + texdata
  const magic = 0x674b; // 'Kg' in little-endian
  const totalSize = 4 + 4 + jsonBytes.length + mergedTexdata.byteLength;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // Write header
  view.setUint32(0, magic, true);
  view.setUint32(4, jsonBytes.length, true);

  // Write JSON
  const uint8View = new Uint8Array(buffer);
  uint8View.set(jsonBytes, 8);

  // Write texdata
  const texdataBytes = new Uint8Array(mergedTexdata.buffer);
  uint8View.set(texdataBytes, 8 + jsonBytes.length);

  return new Blob([buffer], { type: 'application/octet-stream' });
}

/**
 * Export scene to standard .splat format (no metadata, compatible with other viewers).
 * @param {SceneManager} sceneManager - Scene manager with layers
 * @returns {Blob} .splat file as Blob
 */
export function exportSceneToSplat(sceneManager) {
  const visibleLayers = sceneManager.layers.filter(l => l.visible);

  if (visibleLayers.length === 0) {
    throw new Error('No visible layers to export');
  }

  // Calculate total gaussians
  let totalCount = 0;
  for (const layer of visibleLayers) {
    totalCount += layer.count;
  }

  // Standard .splat format: 32 bytes per gaussian
  // position (3 floats) + scale (3 floats) + rgba (4 bytes) + rotation (4 bytes)
  const buffer = new ArrayBuffer(totalCount * 32);
  const floatView = new Float32Array(buffer);
  const uint8View = new Uint8Array(buffer);

  let gaussianIdx = 0;
  for (const layer of visibleLayers) {
    const texdata = layer.bakedTexdata;
    const texdata_f = new Float32Array(texdata.buffer);
    const texdata_u8 = new Uint8Array(texdata.buffer);

    for (let i = 0; i < layer.count; i++) {
      const base_f = gaussianIdx * 8;
      const base_u8 = gaussianIdx * 32;

      // Position (from texdata floats 0-2)
      floatView[base_f + 0] = texdata_f[16 * i + 0];
      floatView[base_f + 1] = texdata_f[16 * i + 1];
      floatView[base_f + 2] = texdata_f[16 * i + 2];

      // Scale (from packed half floats in texdata[5] and texdata[6])
      const scale0 = unpackHalf2x16Low(texdata[16 * i + 5]);
      const scale1 = unpackHalf2x16High(texdata[16 * i + 5]);
      const scale2 = unpackHalf2x16Low(texdata[16 * i + 6]);
      floatView[base_f + 3] = scale0;
      floatView[base_f + 4] = scale1;
      floatView[base_f + 5] = scale2;

      // Color RGBA (from texdata[7] as uint8s)
      uint8View[base_u8 + 24] = texdata_u8[4 * (16 * i + 7) + 0];
      uint8View[base_u8 + 25] = texdata_u8[4 * (16 * i + 7) + 1];
      uint8View[base_u8 + 26] = texdata_u8[4 * (16 * i + 7) + 2];
      uint8View[base_u8 + 27] = texdata_u8[4 * (16 * i + 7) + 3];

      // Rotation quaternion (from packed half floats in texdata[3] and texdata[4])
      // Convert back to uint8 format: (q * 128 + 128)
      const rot0 = unpackHalf2x16Low(texdata[16 * i + 3]);
      const rot1 = unpackHalf2x16High(texdata[16 * i + 3]);
      const rot2 = unpackHalf2x16Low(texdata[16 * i + 4]);
      const rot3 = unpackHalf2x16High(texdata[16 * i + 4]);

      uint8View[base_u8 + 28] = Math.max(0, Math.min(255, Math.round(rot0 * 128 + 128)));
      uint8View[base_u8 + 29] = Math.max(0, Math.min(255, Math.round(rot1 * 128 + 128)));
      uint8View[base_u8 + 30] = Math.max(0, Math.min(255, Math.round(rot2 * 128 + 128)));
      uint8View[base_u8 + 31] = Math.max(0, Math.min(255, Math.round(rot3 * 128 + 128)));

      gaussianIdx++;
    }
  }

  return new Blob([buffer], { type: 'application/octet-stream' });
}

/**
 * Extract camera information from view matrix.
 */
function extractCameraInfo(viewMatrix, controls) {
  // Invert view matrix to get camera world transform
  const inv = invert4(viewMatrix);

  return {
    id: 0,
    img_name: 'export',
    width: controls.canvas?.width || 1920,
    height: controls.canvas?.height || 1080,
    position: inv ? [inv[12], inv[13], inv[14]] : [0, 0, 5],
    rotation: extractRotationMatrix(inv),
    fx: 1159.58,
    fy: 1164.66,
  };
}

/**
 * Extract 3x3 rotation matrix from 4x4 matrix.
 */
function extractRotationMatrix(m) {
  if (!m) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return [
    [m[0], m[1], m[2]],
    [m[4], m[5], m[6]],
    [m[8], m[9], m[10]],
  ];
}

/**
 * Invert a 4x4 matrix.
 */
function invert4(m) {
  const out = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

  return out;
}

/**
 * Unpack lower half of packed half2x16.
 */
function unpackHalf2x16Low(packed) {
  return halfToFloat(packed & 0xFFFF);
}

/**
 * Unpack upper half of packed half2x16.
 */
function unpackHalf2x16High(packed) {
  return halfToFloat((packed >> 16) & 0xFFFF);
}

/**
 * Convert half-precision float to single-precision.
 */
function halfToFloat(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    let e = -14;
    let m = frac / 1024;
    while (m < 1) { m *= 2; e--; }
    return (sign ? -1 : 1) * m * Math.pow(2, e);
  } else if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Trigger file download in browser.
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
