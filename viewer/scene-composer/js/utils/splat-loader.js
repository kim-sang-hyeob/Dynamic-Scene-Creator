/**
 * Splat file parsers. Extracted from hybrid.js (Lumina Editor) worker code.
 * Pure functions that return parsed texdata + positions without side effects.
 */

// Float-to-half conversion helpers
const _floatView = new Float32Array(1);
const _int32View = new Int32Array(_floatView.buffer);

export function floatToHalf(float) {
  _floatView[0] = float;
  const f = _int32View[0];
  const sign = (f >> 31) & 0x0001;
  const exp = (f >> 23) & 0x00ff;
  let frac = f & 0x007fffff;
  let newExp;
  if (exp == 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac = frac >> (113 - exp);
    if (frac & 0x01000000) { newExp = 1; frac = 0; }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

export function packHalf2x16(x, y) {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

/**
 * Parse .splat file (antimatter15 format: 32 bytes per gaussian).
 * Layout per gaussian: position float32[3], scale float32[3], rgba uint8[4], rotation uint8[4]
 * @param {ArrayBuffer} buffer
 * @returns {{texdata: Uint32Array, texwidth: number, texheight: number, positions: Float32Array, count: number}}
 */
export function parseSplatBuffer(buffer) {
  const count = buffer.byteLength / 32;
  const f_buffer = new Float32Array(buffer);
  const u8_buffer = new Uint8Array(buffer);

  const positions = new Float32Array(3 * count);
  const texwidth = 1024 * 4;
  const texheight = Math.ceil((4 * count) / texwidth);
  const texdata = new Uint32Array(texwidth * texheight * 4);
  const texdata_c = new Uint8Array(texdata.buffer);
  const texdata_f = new Float32Array(texdata.buffer);

  for (let j = 0; j < count; j++) {
    const base_f = j * 8;
    const base_u8 = j * 32;

    const x = f_buffer[base_f + 0];
    const y = f_buffer[base_f + 1];
    const z = f_buffer[base_f + 2];
    positions[3 * j + 0] = x;
    positions[3 * j + 1] = y;
    positions[3 * j + 2] = z;

    texdata_f[16 * j + 0] = x;
    texdata_f[16 * j + 1] = y;
    texdata_f[16 * j + 2] = z;

    const scale_0 = f_buffer[base_f + 3];
    const scale_1 = f_buffer[base_f + 4];
    const scale_2 = f_buffer[base_f + 5];

    const rot_0 = (u8_buffer[base_u8 + 28] - 128) / 128;
    const rot_1 = (u8_buffer[base_u8 + 29] - 128) / 128;
    const rot_2 = (u8_buffer[base_u8 + 30] - 128) / 128;
    const rot_3 = (u8_buffer[base_u8 + 31] - 128) / 128;

    texdata[16 * j + 3] = packHalf2x16(rot_0, rot_1);
    texdata[16 * j + 4] = packHalf2x16(rot_2, rot_3);
    texdata[16 * j + 5] = packHalf2x16(scale_0, scale_1);
    texdata[16 * j + 6] = packHalf2x16(scale_2, 0);

    texdata_c[4 * (16 * j + 7) + 0] = u8_buffer[base_u8 + 24];
    texdata_c[4 * (16 * j + 7) + 1] = u8_buffer[base_u8 + 25];
    texdata_c[4 * (16 * j + 7) + 2] = u8_buffer[base_u8 + 26];
    texdata_c[4 * (16 * j + 7) + 3] = u8_buffer[base_u8 + 27];

    // No motion data for static splat
    texdata[16 * j + 8 + 0] = 0;
    texdata[16 * j + 8 + 1] = 0;
    texdata[16 * j + 8 + 2] = 0;
    texdata[16 * j + 8 + 3] = 0;
    texdata[16 * j + 8 + 4] = 0;
    texdata[16 * j + 8 + 5] = 0;
    texdata[16 * j + 8 + 6] = 0;
    texdata[16 * j + 8 + 7] = packHalf2x16(0.5, 1.0);
  }

  return { texdata, texwidth, texheight, positions, count };
}

/**
 * Parse .ply file (3DGS format with SH coefficients + optional 4DGS motion).
 * Sorts gaussians by importance (size * opacity) for progressive rendering.
 * @param {ArrayBuffer} buffer
 * @returns {{texdata: Uint32Array, texwidth: number, texheight: number, positions: Float32Array, count: number}}
 */
export function parsePlyBuffer(buffer) {
  const ubuf = new Uint8Array(buffer);
  const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
  const header_end = "end_header\n";
  const header_end_index = header.indexOf(header_end);
  if (header_end_index < 0) throw new Error("Unable to read .ply file header");

  const count = parseInt(/element vertex (\d+)\n/.exec(header)[1]);

  let row_offset = 0, offsets = {}, types = {};
  const TYPE_MAP = {
    double: "getFloat64", int: "getInt32", uint: "getUint32",
    float: "getFloat32", short: "getInt16", ushort: "getUint16", uchar: "getUint8",
  };

  for (let prop of header.slice(0, header_end_index).split("\n").filter(k => k.startsWith("property "))) {
    const [p, type, name] = prop.split(" ");
    const arrayType = TYPE_MAP[type] || "getInt8";
    types[name] = arrayType;
    offsets[name] = row_offset;
    row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
  }

  const dataView = new DataView(buffer, header_end_index + header_end.length);
  let row = 0;
  const attrs = new Proxy({}, {
    get(target, prop) {
      if (!types[prop]) throw new Error(prop + " not found");
      return dataView[types[prop]](row * row_offset + offsets[prop], true);
    },
  });

  // Sort by importance (size * opacity)
  let sizeList = new Float32Array(count);
  let sizeIndex = new Uint32Array(count);
  for (row = 0; row < count; row++) {
    sizeIndex[row] = row;
    if (!types["scale_0"]) continue;
    const size = Math.exp(attrs.scale_0) * Math.exp(attrs.scale_1) * Math.exp(attrs.scale_2);
    const opacity = 1 / (1 + Math.exp(-attrs.opacity));
    sizeList[row] = size * opacity;
  }
  sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);

  const positions = new Float32Array(3 * count);
  const texwidth = 1024 * 4;
  const texheight = Math.ceil((4 * count) / texwidth);
  const texdata = new Uint32Array(texwidth * texheight * 4);
  const texdata_c = new Uint8Array(texdata.buffer);
  const texdata_f = new Float32Array(texdata.buffer);

  for (let j = 0; j < count; j++) {
    row = sizeIndex[j];

    positions[3 * j + 0] = attrs.x;
    positions[3 * j + 1] = attrs.y;
    positions[3 * j + 2] = attrs.z;

    texdata_f[16 * j + 0] = attrs.x;
    texdata_f[16 * j + 1] = attrs.y;
    texdata_f[16 * j + 2] = attrs.z;

    texdata[16 * j + 3] = packHalf2x16(attrs.rot_0, attrs.rot_1);
    texdata[16 * j + 4] = packHalf2x16(attrs.rot_2, attrs.rot_3);
    texdata[16 * j + 5] = packHalf2x16(Math.exp(attrs.scale_0), Math.exp(attrs.scale_1));
    texdata[16 * j + 6] = packHalf2x16(Math.exp(attrs.scale_2), 0);

    texdata_c[4 * (16 * j + 7) + 0] = Math.max(0, Math.min(255, attrs.f_dc_0 * 255));
    texdata_c[4 * (16 * j + 7) + 1] = Math.max(0, Math.min(255, attrs.f_dc_1 * 255));
    texdata_c[4 * (16 * j + 7) + 2] = Math.max(0, Math.min(255, attrs.f_dc_2 * 255));
    texdata_c[4 * (16 * j + 7) + 3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;

    // Motion data (4DGS fields, if present)
    if (types["motion_0"]) {
      texdata[16 * j + 8 + 0] = packHalf2x16(attrs.motion_0, attrs.motion_1);
      texdata[16 * j + 8 + 1] = packHalf2x16(attrs.motion_2, attrs.motion_3);
      texdata[16 * j + 8 + 2] = packHalf2x16(attrs.motion_4, attrs.motion_5);
      texdata[16 * j + 8 + 3] = packHalf2x16(attrs.motion_6, attrs.motion_7);
      texdata[16 * j + 8 + 4] = packHalf2x16(attrs.motion_8, 0);
      texdata[16 * j + 8 + 5] = packHalf2x16(attrs.omega_0, attrs.omega_1);
      texdata[16 * j + 8 + 6] = packHalf2x16(attrs.omega_2, attrs.omega_3);
      texdata[16 * j + 8 + 7] = packHalf2x16(attrs.trbf_center, Math.exp(attrs.trbf_scale));
    } else {
      texdata[16 * j + 8 + 0] = 0;
      texdata[16 * j + 8 + 1] = 0;
      texdata[16 * j + 8 + 2] = 0;
      texdata[16 * j + 8 + 3] = 0;
      texdata[16 * j + 8 + 4] = 0;
      texdata[16 * j + 8 + 5] = 0;
      texdata[16 * j + 8 + 6] = 0;
      texdata[16 * j + 8 + 7] = packHalf2x16(0.5, 1.0);
    }
  }

  return { texdata, texwidth, texheight, positions, count };
}

/**
 * Parse .splatv file (binary: magic 0x674b + JSON metadata + texdata).
 * @param {ArrayBuffer} buffer
 * @returns {{texdata: Uint32Array, texwidth: number, texheight: number, positions: Float32Array, count: number, cameras: Array, layers: Array|null}}
 */
export function parseSplatvBuffer(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x674b) throw new Error("Not a valid .splatv file");

  const jsonLen = view.getUint32(4, true);
  const jsonBytes = new Uint8Array(buffer, 8, jsonLen);
  const metadata = JSON.parse(new TextDecoder().decode(jsonBytes));

  const chunk = metadata[0];
  const texwidth = chunk.texwidth;
  const texheight = chunk.texheight;
  const cameras = chunk.cameras || [];
  const layers = chunk.layers || null;  // Layer metadata for restoration

  const dataOffset = 8 + jsonLen;
  const texdata = new Uint32Array(buffer.slice(dataOffset));
  const texdata_f = new Float32Array(texdata.buffer);

  const count = Math.floor(texdata.length / 16);
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[3 * i + 0] = texdata_f[16 * i + 0];
    positions[3 * i + 1] = texdata_f[16 * i + 1];
    positions[3 * i + 2] = texdata_f[16 * i + 2];
  }

  return { texdata, texwidth, texheight, positions, count, cameras, layers };
}

/**
 * Detect file format from ArrayBuffer content.
 * @param {ArrayBuffer} buffer
 * @param {string} filename
 * @returns {'splat'|'ply'|'splatv'|null}
 */
export function detectFormat(buffer, filename) {
  const u8 = new Uint8Array(buffer, 0, Math.min(4, buffer.byteLength));
  if (u8[0] === 0x4b && u8[1] === 0x67) return 'splatv'; // magic 0x674b (little-endian)
  if (u8[0] === 112 && u8[1] === 108 && u8[2] === 121 && u8[3] === 10) return 'ply'; // "ply\n"
  if (/\.splat$/i.test(filename) || buffer.byteLength % 32 === 0) return 'splat';
  return null;
}
