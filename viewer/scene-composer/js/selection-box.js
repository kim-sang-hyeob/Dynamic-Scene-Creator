/**
 * SelectionBox — Renders AABB wireframe around the selected layer.
 * Uses the same simple shader pattern as Gizmo (pos+color passthrough).
 */

import { multiply4 } from './utils/matrix-math.js';
import { ensureBounds } from './utils/ray-cast.js';

const BOX_COLOR = [0.4, 0.85, 1.0, 0.7];

export class SelectionBox {
  constructor(renderer, sceneManager) {
    this.renderer = renderer;
    this.gl = renderer.gl;
    this.sceneManager = sceneManager;
    this.program = null;
    this.vao = null;
    this.posBuffer = null;
    this.colorBuffer = null;
    this.u_viewProj = null;
  }

  init() {
    const gl = this.gl;

    const vsSource = `#version 300 es
      precision highp float;
      uniform mat4 u_viewProj;
      in vec3 a_position;
      in vec4 a_color;
      out vec4 vColor;
      void main() {
        gl_Position = u_viewProj * vec4(a_position, 1.0);
        vColor = a_color;
      }
    `;
    const fsSource = `#version 300 es
      precision highp float;
      in vec4 vColor;
      out vec4 fragColor;
      void main() {
        fragColor = vColor;
      }
    `;

    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    this.u_viewProj = gl.getUniformLocation(this.program, 'u_viewProj');

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    this.colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    const colLoc = gl.getAttribLocation(this.program, 'a_color');
    gl.enableVertexAttribArray(colLoc);
    gl.vertexAttribPointer(colLoc, 4, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
  }

  render(viewMatrix, projMatrix) {
    const layer = this.sceneManager.getSelectedLayer();
    if (!layer || layer.count === 0) return;

    ensureBounds(layer);
    const min = layer._aabbMin;
    const max = layer._aabbMax;
    if (!min || !max) return;

    const positions = this._buildBoxLines(min, max);
    const colors = new Float32Array(24 * 4);
    for (let i = 0; i < 24; i++) {
      colors[i * 4]     = BOX_COLOR[0];
      colors[i * 4 + 1] = BOX_COLOR[1];
      colors[i * 4 + 2] = BOX_COLOR[2];
      colors[i * 4 + 3] = BOX_COLOR[3];
    }

    const gl = this.gl;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);

    // Overlay blend mode
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.blendEquation(gl.FUNC_ADD);

    const vp = multiply4(projMatrix, viewMatrix);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.u_viewProj, false, vp);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.LINES, 0, 24);
    gl.bindVertexArray(null);

    // Restore splat blend mode
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
  }

  _buildBoxLines(min, max) {
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;

    // 8 corners
    const c = [
      [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
      [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1],
    ];

    // 12 edges
    const edges = [
      [0,1],[1,2],[2,3],[3,0],
      [4,5],[5,6],[6,7],[7,4],
      [0,4],[1,5],[2,6],[3,7],
    ];

    const arr = new Float32Array(24 * 3);
    let off = 0;
    for (const [a, b] of edges) {
      arr[off++] = c[a][0]; arr[off++] = c[a][1]; arr[off++] = c[a][2];
      arr[off++] = c[b][0]; arr[off++] = c[b][1]; arr[off++] = c[b][2];
    }
    return arr;
  }
}
