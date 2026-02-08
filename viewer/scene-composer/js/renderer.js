/**
 * WebGL2 Gaussian Splat Renderer.
 * Forked from hybrid.js (Lumina Editor) — editor-specific code removed.
 * Supports multi-layer scene upload via uploadScene().
 */

import { multiply4, getProjectionMatrix } from './utils/matrix-math.js';

// ── Sort-only Web Worker ──────────────────────────────────────────────
function createSortWorker(self) {
  let vertexCount = 0;
  let viewProj;
  let lastProj = [];
  let depthIndex = new Uint32Array();
  let lastVertexCount = 0;
  let positions;

  function runSort(viewProj) {
    if (!positions) return;
    if (lastVertexCount == vertexCount) {
      let dist = Math.hypot(...[2, 6, 10].map((k) => lastProj[k] - viewProj[k]));
      if (dist < 0.01) return;
    } else {
      lastVertexCount = vertexCount;
    }

    let maxDepth = -Infinity;
    let minDepth = Infinity;
    let sizeList = new Int32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      let depth =
        ((viewProj[2] * positions[3 * i + 0] +
          viewProj[6] * positions[3 * i + 1] +
          viewProj[10] * positions[3 * i + 2]) *
          4096) |
        0;
      sizeList[i] = depth;
      if (depth > maxDepth) maxDepth = depth;
      if (depth < minDepth) minDepth = depth;
    }

    // 16-bit single-pass counting sort
    let depthInv = (256 * 256) / (maxDepth - minDepth);
    let counts0 = new Uint32Array(256 * 256);
    for (let i = 0; i < vertexCount; i++) {
      sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
      counts0[sizeList[i]]++;
    }
    let starts0 = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
    depthIndex = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;

    lastProj = viewProj;
    self.postMessage({ depthIndex, viewProj, vertexCount }, [depthIndex.buffer]);
  }

  const throttledSort = () => {
    if (!sortRunning) {
      sortRunning = true;
      let lastView = viewProj;
      runSort(lastView);
      setTimeout(() => {
        sortRunning = false;
        if (lastView !== viewProj) throttledSort();
      }, 0);
    }
  };

  let sortRunning;

  self.onmessage = (e) => {
    if (e.data.texture) {
      // Receive Float32Array of texdata — extract positions
      let texture = e.data.texture;
      vertexCount = e.data.vertexCount;
      positions = new Float32Array(vertexCount * 3);
      for (let i = 0; i < vertexCount; i++) {
        positions[3 * i + 0] = texture[16 * i + 0];
        positions[3 * i + 1] = texture[16 * i + 1];
        positions[3 * i + 2] = texture[16 * i + 2];
      }
      lastVertexCount = -1;
      throttledSort();
    } else if (e.data.view) {
      viewProj = e.data.view;
      throttledSort();
    }
  };
}

// ── Shaders (from hybrid.js) ──────────────────────────────────────────
const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
uniform float time;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    uvec4 motion1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 3u, uint(index) >> 10), 0);
    vec2 trbf = unpackHalf2x16(motion1.w);
    float dt = time - trbf.x;

    float topacity = exp(-1.0 * pow(dt / trbf.y, 2.0));
    if(topacity < 0.02) return;

    uvec4 motion0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 2u, uint(index) >> 10), 0);
    uvec4 static0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2), uint(index) >> 10), 0);

    vec2 m0 = unpackHalf2x16(motion0.x), m1 = unpackHalf2x16(motion0.y), m2 = unpackHalf2x16(motion0.z),
         m3 = unpackHalf2x16(motion0.w), m4 = unpackHalf2x16(motion1.x);

    vec4 trot = vec4(unpackHalf2x16(motion1.y).xy, unpackHalf2x16(motion1.z).xy) * dt;
    vec3 tpos = (vec3(m0.xy, m1.x) * dt + vec3(m1.y, m2.xy) * dt*dt + vec3(m3.xy, m4.x) * dt*dt*dt);

    vec4 cam = view * vec4(uintBitsToFloat(static0.xyz) + tpos, 1);
    vec4 pos = projection * cam;

    float clip = 1.2 * pos.w;
    if (pos.z < -clip || pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
    uvec4 static1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 1u, uint(index) >> 10), 0);

    vec4 rot = vec4(unpackHalf2x16(static0.w).xy, unpackHalf2x16(static1.x).xy) + trot;
    vec3 scale = vec3(unpackHalf2x16(static1.y).xy, unpackHalf2x16(static1.z).x);
    rot /= sqrt(dot(rot, rot));

    mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 R = mat3(
      1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w), 2.0 * (rot.y * rot.w + rot.x * rot.z),
      2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w - rot.x * rot.y),
      2.0 * (rot.y * rot.w - rot.x * rot.z), 2.0 * (rot.z * rot.w + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z));
    mat3 M = S * R;
    mat3 Vrk = 4.0 * transpose(M) * M;
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    uint rgba = static1.w;
    vColor =
      clamp(pos.z/pos.w+1.0, 0.0, 1.0) *
      vec4(1.0, 1.0, 1.0, topacity) *
      vec4(
        (rgba) & 0xffu,
        (rgba >> 8) & 0xffu,
        (rgba >> 16) & 0xffu,
        (rgba >> 24) & 0xffu) / 255.0;

    vec2 vCenter = vec2(pos) / pos.w;
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport
        + position.y * minorAxis / viewport, 0.0, 1.0);

    vPosition = position;
}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}
`.trim();

// ── Renderer Class ────────────────────────────────────────────────────
export class SplatRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.program = null;
    this.worker = null;
    this.vertexCount = 0;
    this.texture = null;
    this.indexBuffer = null;
    this.vertexBuffer = null;
    this.projectionMatrix = null;
    this.camera = { fx: 1159.5880733038064, fy: 1164.6601287484507 };
    this.fps = 0;

    // Uniform locations
    this.u_projection = null;
    this.u_viewport = null;
    this.u_focal = null;
    this.u_view = null;
    this.u_time = null;

    // Callbacks
    this.onAfterDraw = null; // (viewMatrix, projMatrix) => {} for overlay rendering
  }

  init() {
    const gl = this.canvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    // Compile shaders
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vertexShaderSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vs));

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fragmentShaderSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fs));

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.useProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
    this.program = program;

    // Blend state
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    // Uniforms
    this.u_projection = gl.getUniformLocation(program, "projection");
    this.u_viewport = gl.getUniformLocation(program, "viewport");
    this.u_focal = gl.getUniformLocation(program, "focal");
    this.u_view = gl.getUniformLocation(program, "view");
    this.u_time = gl.getUniformLocation(program, "time");

    // Quad vertices (instanced billboard)
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    this.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
    this.a_position = a_position;

    // Texture
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniform1i(gl.getUniformLocation(program, "u_texture"), 0);

    // Index buffer (depth-sorted indices from worker)
    this.indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);
    this.a_index = a_index;

    // Worker
    this.worker = new Worker(
      URL.createObjectURL(
        new Blob(["(", createSortWorker.toString(), ")(self)"], {
          type: "application/javascript",
        })
      )
    );
    this.worker.onmessage = (e) => {
      if (e.data.depthIndex) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
        this.vertexCount = e.data.vertexCount;
      }
    };

    // Initial resize
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const gl = this.gl;
    gl.uniform2fv(this.u_focal, new Float32Array([this.camera.fx, this.camera.fy]));
    this.projectionMatrix = getProjectionMatrix(this.camera.fx, this.camera.fy, innerWidth, innerHeight);
    gl.uniform2fv(this.u_viewport, new Float32Array([innerWidth, innerHeight]));
    gl.canvas.width = innerWidth;
    gl.canvas.height = innerHeight;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.uniformMatrix4fv(this.u_projection, false, this.projectionMatrix);
  }

  /**
   * Upload scene data to GPU and worker.
   * @param {Uint32Array} texdata - Packed gaussian texture data
   * @param {number} texwidth
   * @param {number} texheight
   * @param {number} gaussianCount
   */
  uploadScene(texdata, texwidth, texheight, gaussianCount) {
    const gl = this.gl;

    // Upload to GPU texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);

    // Send to worker for depth sorting
    const texF = new Float32Array(texdata.buffer, texdata.byteOffset, texdata.length);
    this.worker.postMessage({ texture: texF, vertexCount: gaussianCount });
  }

  /**
   * Start the render loop.
   * @param {Function} getViewMatrix - () => Float32Array(16) returns current view matrix
   */
  startLoop(getViewMatrix) {
    let lastFrame = 0;
    let avgFps = 0;

    const frame = (now) => {
      const viewMatrix = getViewMatrix();
      const viewProj = multiply4(this.projectionMatrix, viewMatrix);
      this.worker.postMessage({ view: viewProj });

      const currentFps = 1000 / (now - lastFrame) || 0;
      avgFps = (isFinite(avgFps) && avgFps) * 0.9 + currentFps * 0.1;

      const gl = this.gl;

      if (this.vertexCount > 0) {
        // Restore splat program state
        gl.useProgram(this.program);
        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.enableVertexAttribArray(this.a_position);
        gl.vertexAttribPointer(this.a_position, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
        gl.enableVertexAttribArray(this.a_index);
        gl.vertexAttribIPointer(this.a_index, 1, gl.INT, false, 0, 0);
        gl.vertexAttribDivisor(this.a_index, 1);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);

        gl.uniformMatrix4fv(this.u_view, false, viewMatrix);
        gl.uniform1f(this.u_time, Math.sin(Date.now() / 1000) / 2 + 0.5);

        // Splat blend mode
        gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
        gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.vertexCount);

        // Overlay callback (for Gizmo, etc.)
        if (this.onAfterDraw) {
          this.onAfterDraw(viewMatrix, this.projectionMatrix);
        }
      } else {
        gl.clear(gl.COLOR_BUFFER_BIT);
      }

      this.fps = Math.round(avgFps);
      lastFrame = now;
      requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
  }
}
