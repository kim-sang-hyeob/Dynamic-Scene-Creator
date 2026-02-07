/**
 * Procedural geometry builders for the 3D transform gizmo.
 * Returns Float32Array vertex data — no GL calls.
 */

// ── Axis Colors ─────────────────────────────────────────────────────

export const AXIS_COLORS = {
  x: [1.0, 0.25, 0.25, 1.0],
  y: [0.25, 0.85, 0.25, 1.0],
  z: [0.35, 0.5, 1.0, 1.0],
  center: [1.0, 1.0, 0.3, 1.0],
};

export const AXIS_COLORS_HIGHLIGHT = {
  x: [1.0, 0.6, 0.4, 1.0],
  y: [0.5, 1.0, 0.5, 1.0],
  z: [0.6, 0.75, 1.0, 1.0],
  center: [1.0, 1.0, 0.7, 1.0],
};

const AXES = [
  { name: 'x', dir: [1,0,0], up: [0,1,0] },
  { name: 'y', dir: [0,1,0], up: [0,0,1] },
  { name: 'z', dir: [0,0,1], up: [0,1,0] },
];

// ── Translate Gizmo ─────────────────────────────────────────────────

/**
 * Build translate gizmo: 3 axis lines + arrow cone tips.
 * @param {number[]} center - [x,y,z]
 * @param {number} axisLen
 * @param {number} [coneLen] - arrow cone length
 * @param {number} [coneRadius] - arrow cone radius
 * @param {number} [segments] - cone segments
 * @returns {{ positions: Float32Array, colors: Float32Array, lineVertexCount: number, triangleVertexStart: number, triangleVertexCount: number, hitAreas: object }}
 */
export function buildTranslateGizmo(center, axisLen, coneLen, coneRadius, segments) {
  coneLen = coneLen || axisLen * 0.18;
  coneRadius = coneRadius || axisLen * 0.06;
  segments = segments || 8;

  const lineVerts = [];
  const lineColors = [];
  const triVerts = [];
  const triColors = [];
  const hitAreas = {};

  for (const axis of AXES) {
    const c = AXIS_COLORS[axis.name];
    const end = [
      center[0] + axis.dir[0] * axisLen,
      center[1] + axis.dir[1] * axisLen,
      center[2] + axis.dir[2] * axisLen,
    ];
    const coneBase = [
      center[0] + axis.dir[0] * (axisLen - coneLen),
      center[1] + axis.dir[1] * (axisLen - coneLen),
      center[2] + axis.dir[2] * (axisLen - coneLen),
    ];

    // Line: center → coneBase
    lineVerts.push(...center, ...coneBase);
    lineColors.push(...c, ...c);

    // Cone tip (triangle fan → triangles)
    const perp1 = cross(axis.dir, axis.up);
    const pLen = vecLen(perp1);
    if (pLen > 0.001) { perp1[0] /= pLen; perp1[1] /= pLen; perp1[2] /= pLen; }
    const perp2 = cross(axis.dir, perp1);

    for (let i = 0; i < segments; i++) {
      const a1 = (i / segments) * Math.PI * 2;
      const a2 = ((i + 1) / segments) * Math.PI * 2;
      const p1 = ringPoint(coneBase, perp1, perp2, coneRadius, a1);
      const p2 = ringPoint(coneBase, perp1, perp2, coneRadius, a2);
      triVerts.push(...end, ...p1, ...p2);
      triColors.push(...c, ...c, ...c);
    }

    // Hit area: axis start to end
    hitAreas[axis.name] = { start: [...center], end };
  }

  // Merge line + triangle verts
  const lineCount = lineVerts.length / 3;
  const triCount = triVerts.length / 3;
  const positions = new Float32Array([...lineVerts, ...triVerts]);
  const colors = new Float32Array([...lineColors, ...triColors]);

  return {
    positions,
    colors,
    lineVertexCount: lineCount,
    triangleVertexStart: lineCount,
    triangleVertexCount: triCount,
    hitAreas,
  };
}

// ── Rotate Gizmo ────────────────────────────────────────────────────

/**
 * Build rotate gizmo: 3 circle rings.
 * @param {number[]} center
 * @param {number} radius
 * @param {number} [segments]
 * @returns {{ positions: Float32Array, colors: Float32Array, lineVertexCount: number, triangleVertexStart: number, triangleVertexCount: number, hitAreas: object }}
 */
export function buildRotateGizmo(center, radius, segments) {
  segments = segments || 32;

  const lineVerts = [];
  const lineColors = [];
  const hitAreas = {};

  // Each ring: rotation around one axis
  // X-axis rotation → ring in YZ plane
  // Y-axis rotation → ring in XZ plane
  // Z-axis rotation → ring in XY plane
  const ringDefs = [
    { name: 'x', normal: [1,0,0], p1: [0,1,0], p2: [0,0,1] },
    { name: 'y', normal: [0,1,0], p1: [1,0,0], p2: [0,0,1] },
    { name: 'z', normal: [0,0,1], p1: [1,0,0], p2: [0,1,0] },
  ];

  for (const ring of ringDefs) {
    const c = AXIS_COLORS[ring.name];

    for (let i = 0; i < segments; i++) {
      const a1 = (i / segments) * Math.PI * 2;
      const a2 = ((i + 1) / segments) * Math.PI * 2;
      const p1 = ringPoint(center, ring.p1, ring.p2, radius, a1);
      const p2 = ringPoint(center, ring.p1, ring.p2, radius, a2);
      lineVerts.push(...p1, ...p2);
      lineColors.push(...c, ...c);
    }

    hitAreas[ring.name] = { center: [...center], normal: ring.normal, radius };
  }

  const lineCount = lineVerts.length / 3;
  return {
    positions: new Float32Array(lineVerts),
    colors: new Float32Array(lineColors),
    lineVertexCount: lineCount,
    triangleVertexStart: lineCount,
    triangleVertexCount: 0,
    hitAreas,
  };
}

// ── Scale Gizmo ─────────────────────────────────────────────────────

/**
 * Build scale gizmo: 3 axis lines + cube handles at endpoints + center cube.
 * @param {number[]} center
 * @param {number} axisLen
 * @param {number} [cubeSize] - half-size of cube handles
 * @returns {{ positions: Float32Array, colors: Float32Array, lineVertexCount: number, triangleVertexStart: number, triangleVertexCount: number, hitAreas: object }}
 */
export function buildScaleGizmo(center, axisLen, cubeSize) {
  cubeSize = cubeSize || axisLen * 0.08;

  const lineVerts = [];
  const lineColors = [];
  const triVerts = [];
  const triColors = [];
  const hitAreas = {};

  for (const axis of AXES) {
    const c = AXIS_COLORS[axis.name];
    const end = [
      center[0] + axis.dir[0] * axisLen,
      center[1] + axis.dir[1] * axisLen,
      center[2] + axis.dir[2] * axisLen,
    ];

    // Line: center → end
    lineVerts.push(...center, ...end);
    lineColors.push(...c, ...c);

    // Cube at endpoint
    addCube(triVerts, triColors, end, cubeSize, c);
    hitAreas[axis.name] = { cubeCenter: end, cubeSize };
  }

  // Center cube (uniform scale)
  const cc = AXIS_COLORS.center;
  addCube(triVerts, triColors, center, cubeSize * 1.2, cc);
  hitAreas.center = { cubeCenter: [...center], cubeSize: cubeSize * 1.2 };

  const lineCount = lineVerts.length / 3;
  const triCount = triVerts.length / 3;
  return {
    positions: new Float32Array([...lineVerts, ...triVerts]),
    colors: new Float32Array([...lineColors, ...triColors]),
    lineVertexCount: lineCount,
    triangleVertexStart: lineCount,
    triangleVertexCount: triCount,
    hitAreas,
  };
}

// ── Geometry Helpers ────────────────────────────────────────────────

function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vecLen(v) { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }

function ringPoint(center, p1, p2, radius, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [
    center[0] + (p1[0]*c + p2[0]*s) * radius,
    center[1] + (p1[1]*c + p2[1]*s) * radius,
    center[2] + (p1[2]*c + p2[2]*s) * radius,
  ];
}

function addCube(verts, colors, center, half, color) {
  // 6 faces, 2 triangles each = 36 vertices
  const [cx, cy, cz] = center;
  const h = half;
  const faces = [
    // +X
    [[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx+h,cy+h,cz+h],[cx+h,cy-h,cz+h]],
    // -X
    [[cx-h,cy-h,cz+h],[cx-h,cy+h,cz+h],[cx-h,cy+h,cz-h],[cx-h,cy-h,cz-h]],
    // +Y
    [[cx-h,cy+h,cz-h],[cx-h,cy+h,cz+h],[cx+h,cy+h,cz+h],[cx+h,cy+h,cz-h]],
    // -Y
    [[cx-h,cy-h,cz+h],[cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy-h,cz+h]],
    // +Z
    [[cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h]],
    // -Z
    [[cx+h,cy-h,cz-h],[cx-h,cy-h,cz-h],[cx-h,cy+h,cz-h],[cx+h,cy+h,cz-h]],
  ];
  for (const face of faces) {
    // Two triangles per face
    verts.push(...face[0], ...face[1], ...face[2]);
    verts.push(...face[0], ...face[2], ...face[3]);
    for (let i = 0; i < 6; i++) colors.push(...color);
  }
}
