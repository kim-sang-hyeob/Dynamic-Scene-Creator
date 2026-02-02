// overlay-renderer.js - WebGL2 overlay for bezier curves and control points

const OverlayRenderer = {
  gl: null,
  lineProgram: null,
  pointProgram: null,
  lineVAO: null,
  pointVAO: null,
  indicatorVAO: null,
  lineBuffer: null,
  pointBuffer: null,
  pointColorBuffer: null,
  indicatorPosBuffer: null,
  indicatorColorBuffer: null,
  camLinesVAO: null,
  camLinesBuffer: null,
  camLinesVertexCount: 0,
  lineVertexCount: 0,
  pointCount: 0,
  animIndicatorPos: null, // [x,y,z] or null
  hideOverlay: false, // Set true during recording to hide path/points

  // Cached uniform locations
  _uniforms: null,

  init(gl) {
    this.gl = gl;

    // --- Line shader (thick lines via screen-space offset) ---
    const lineVS = `#version 300 es
      precision highp float;
      uniform mat4 u_viewProj;
      uniform vec2 u_offset;
      uniform vec2 u_resolution;
      in vec3 a_position;
      void main() {
        vec4 pos = u_viewProj * vec4(a_position, 1.0);
        pos.xy += u_offset * 2.0 * pos.w / u_resolution;
        gl_Position = pos;
      }
    `;
    const lineFS = `#version 300 es
      precision highp float;
      uniform vec4 u_color;
      out vec4 fragColor;
      void main() {
        fragColor = u_color;
      }
    `;

    // --- Point shader (billboard quads via gl_PointSize) ---
    const pointVS = `#version 300 es
      precision highp float;
      uniform mat4 u_viewProj;
      uniform float u_pointSize;
      in vec3 a_position;
      in vec4 a_color;
      out vec4 vColor;
      void main() {
        gl_Position = u_viewProj * vec4(a_position, 1.0);
        gl_PointSize = u_pointSize / gl_Position.w;
        vColor = a_color;
      }
    `;
    const pointFS = `#version 300 es
      precision highp float;
      in vec4 vColor;
      out vec4 fragColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = dot(coord, coord);
        if (dist > 0.25) discard;
        // Dark border ring for visibility
        float border = smoothstep(0.16, 0.19, dist);
        vec3 color = mix(vColor.rgb, vColor.rgb * 0.25, border * 0.8);
        // Soft edge
        float alpha = 1.0 - smoothstep(0.2, 0.25, dist);
        fragColor = vec4(color, vColor.a * alpha);
      }
    `;

    this.lineProgram = this._createProgram(lineVS, lineFS);
    this.pointProgram = this._createProgram(pointVS, pointFS);

    // Cache uniform locations
    this._uniforms = {
      line_viewProj: gl.getUniformLocation(this.lineProgram, 'u_viewProj'),
      line_color: gl.getUniformLocation(this.lineProgram, 'u_color'),
      line_offset: gl.getUniformLocation(this.lineProgram, 'u_offset'),
      line_resolution: gl.getUniformLocation(this.lineProgram, 'u_resolution'),
      point_viewProj: gl.getUniformLocation(this.pointProgram, 'u_viewProj'),
      point_pointSize: gl.getUniformLocation(this.pointProgram, 'u_pointSize'),
    };

    // Line VAO
    this.lineVAO = gl.createVertexArray();
    gl.bindVertexArray(this.lineVAO);
    this.lineBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    const linePosLoc = gl.getAttribLocation(this.lineProgram, 'a_position');
    gl.enableVertexAttribArray(linePosLoc);
    gl.vertexAttribPointer(linePosLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Point VAO
    const ptPosLoc = gl.getAttribLocation(this.pointProgram, 'a_position');
    const ptColLoc = gl.getAttribLocation(this.pointProgram, 'a_color');

    this.pointVAO = gl.createVertexArray();
    gl.bindVertexArray(this.pointVAO);
    this.pointBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
    gl.enableVertexAttribArray(ptPosLoc);
    gl.vertexAttribPointer(ptPosLoc, 3, gl.FLOAT, false, 0, 0);
    this.pointColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointColorBuffer);
    gl.enableVertexAttribArray(ptColLoc);
    gl.vertexAttribPointer(ptColLoc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Indicator VAO (separate buffers, no thrashing)
    this.indicatorVAO = gl.createVertexArray();
    gl.bindVertexArray(this.indicatorVAO);
    this.indicatorPosBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indicatorPosBuffer);
    gl.enableVertexAttribArray(ptPosLoc);
    gl.vertexAttribPointer(ptPosLoc, 3, gl.FLOAT, false, 0, 0);
    this.indicatorColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indicatorColorBuffer);
    gl.enableVertexAttribArray(ptColLoc);
    gl.vertexAttribPointer(ptColLoc, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Camera frustum lines VAO (GL_LINES, reuses line shader)
    this.camLinesVAO = gl.createVertexArray();
    gl.bindVertexArray(this.camLinesVAO);
    this.camLinesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.camLinesBuffer);
    const camLinePosLoc = gl.getAttribLocation(this.lineProgram, 'a_position');
    gl.enableVertexAttribArray(camLinePosLoc);
    gl.vertexAttribPointer(camLinePosLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Pre-upload static indicator color (bright white)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indicatorColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([1.0, 1.0, 1.0, 1.0]), gl.DYNAMIC_DRAW);
  },

  _createProgram(vsSrc, fsSrc) {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('Overlay VS error:', gl.getShaderInfoLog(vs));
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('Overlay FS error:', gl.getShaderInfoLog(fs));
    }

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('Overlay link error:', gl.getProgramInfoLog(prog));
    }
    return prog;
  },

  // Update line data from Float32Array of positions
  updateLines(positionsFloat32, vertexCount) {
    const gl = this.gl;
    this.lineVertexCount = vertexCount;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positionsFloat32, gl.DYNAMIC_DRAW);
  },

  // Update camera frustum lines (GL_LINES pairs)
  updateCameraLines(positionsFloat32, vertexCount) {
    const gl = this.gl;
    this.camLinesVertexCount = vertexCount;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.camLinesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positionsFloat32, gl.DYNAMIC_DRAW);
  },

  // Update point markers
  // positions: Float32Array [x,y,z, x,y,z, ...]
  // colors: Float32Array [r,g,b,a, r,g,b,a, ...]
  updatePoints(positions, colors, count) {
    const gl = this.gl;
    this.pointCount = count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
  },

  render(viewMatrix, projectionMatrix) {
    const gl = this.gl;
    if (!gl) return;
    if (this.hideOverlay) return;
    if (!this.lineVertexCount && !this.pointCount && !this.animIndicatorPos && !this.camLinesVertexCount) return;

    // Compute viewProj
    const vp = this._multiply4(projectionMatrix, viewMatrix);
    const u = this._uniforms;

    // Save splat blend state and switch to standard alpha blending
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.DEPTH_TEST);

    // Draw thick lines (multi-pass with screen-space offsets)
    if (this.lineVertexCount > 1) {
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(u.line_viewProj, false, vp);
      gl.uniform4f(u.line_color, 1.0, 0.25, 0.42, 1.0); // Vivid coral-pink
      gl.uniform2f(u.line_resolution, gl.canvas.width, gl.canvas.height);
      gl.bindVertexArray(this.lineVAO);

      // Draw 5 passes for ~3px thick line
      const offsets = [[0, 0], [1.2, 0], [-1.2, 0], [0, 1.2], [0, -1.2]];
      for (const [ox, oy] of offsets) {
        gl.uniform2f(u.line_offset, ox, oy);
        gl.drawArrays(gl.LINE_STRIP, 0, this.lineVertexCount);
      }
      gl.bindVertexArray(null);
    }

    // Draw camera frustum wireframes (GL_LINES)
    if (this.camLinesVertexCount > 1) {
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(u.line_viewProj, false, vp);
      gl.uniform4f(u.line_color, 0.3, 0.85, 1.0, 0.85); // Light cyan
      gl.uniform2f(u.line_resolution, gl.canvas.width, gl.canvas.height);
      gl.bindVertexArray(this.camLinesVAO);

      const camOffsets = [[0, 0], [0.7, 0], [-0.7, 0], [0, 0.7], [0, -0.7]];
      for (const [ox, oy] of camOffsets) {
        gl.uniform2f(u.line_offset, ox, oy);
        gl.drawArrays(gl.LINES, 0, this.camLinesVertexCount);
      }
      gl.bindVertexArray(null);
    }

    // Draw control points
    if (this.pointCount > 0) {
      gl.useProgram(this.pointProgram);
      gl.uniformMatrix4fv(u.point_viewProj, false, vp);
      gl.uniform1f(u.point_pointSize, 200.0);
      gl.bindVertexArray(this.pointVAO);
      gl.drawArrays(gl.POINTS, 0, this.pointCount);
      gl.bindVertexArray(null);
    }

    // Draw animation indicator (uses its own VAO/buffers)
    if (this.animIndicatorPos) {
      gl.useProgram(this.pointProgram);
      gl.uniformMatrix4fv(u.point_viewProj, false, vp);
      gl.uniform1f(u.point_pointSize, 280.0);

      // Upload indicator position to its own buffer
      gl.bindBuffer(gl.ARRAY_BUFFER, this.indicatorPosBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.animIndicatorPos), gl.DYNAMIC_DRAW);

      gl.bindVertexArray(this.indicatorVAO);
      gl.drawArrays(gl.POINTS, 0, 1);
      gl.bindVertexArray(null);
    }

    // Restore splat blend mode
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
  },

  // Helper: 4x4 matrix multiply (column-major)
  _multiply4(a, b) {
    return [
      b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
      b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
      b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
      b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
      b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
      b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
      b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
      b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
      b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
      b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
      b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
      b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
      b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
      b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
      b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
      b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
  },
};

window.OverlayRenderer = OverlayRenderer;
