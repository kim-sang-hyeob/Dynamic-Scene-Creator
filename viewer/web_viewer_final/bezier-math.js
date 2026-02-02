// bezier-math.js - Bezier/Catmull-Rom math for 3D path editing

const BezierMath = {
  // Evaluate cubic bezier at parameter t (0-1)
  // P0, P1, P2, P3 are [x,y,z] control points
  cubicBezier(P0, P1, P2, P3, t) {
    const mt = 1 - t;
    return [
      mt*mt*mt*P0[0] + 3*mt*mt*t*P1[0] + 3*mt*t*t*P2[0] + t*t*t*P3[0],
      mt*mt*mt*P0[1] + 3*mt*mt*t*P1[1] + 3*mt*t*t*P2[1] + t*t*t*P3[1],
      mt*mt*mt*P0[2] + 3*mt*mt*t*P1[2] + 3*mt*t*t*P2[2] + t*t*t*P3[2],
    ];
  },

  // Convert Catmull-Rom segment (4 points) to cubic Bezier control points
  // tension: 0 = sharp corners, 0.5 = standard, 1 = very smooth
  catmullRomToBezier(p0, p1, p2, p3, tension = 0.5) {
    const alpha = tension;
    return [
      p1, // B0 = p1 (start)
      [ // B1
        p1[0] + (p2[0] - p0[0]) / (6 * alpha || 6),
        p1[1] + (p2[1] - p0[1]) / (6 * alpha || 6),
        p1[2] + (p2[2] - p0[2]) / (6 * alpha || 6),
      ],
      [ // B2
        p2[0] - (p3[0] - p1[0]) / (6 * alpha || 6),
        p2[1] - (p3[1] - p1[1]) / (6 * alpha || 6),
        p2[2] - (p3[2] - p1[2]) / (6 * alpha || 6),
      ],
      p2, // B3 = p2 (end)
    ];
  },

  // Build complete bezier path from array of user waypoints
  // Returns array of bezier segments, each {p0, p1, p2, p3}
  // For endpoints, mirror the first/last interior point
  buildBezierPath(waypoints, tension = 0.5) {
    if (waypoints.length < 2) return [];

    // Create extended array with virtual endpoints for end tangents
    const pts = [];
    // Mirror first point
    pts.push([
      2 * waypoints[0][0] - waypoints[1][0],
      2 * waypoints[0][1] - waypoints[1][1],
      2 * waypoints[0][2] - waypoints[1][2],
    ]);
    pts.push(...waypoints);
    // Mirror last point
    const n = waypoints.length;
    pts.push([
      2 * waypoints[n-1][0] - waypoints[n-2][0],
      2 * waypoints[n-1][1] - waypoints[n-2][1],
      2 * waypoints[n-1][2] - waypoints[n-2][2],
    ]);

    const segments = [];
    for (let i = 0; i < pts.length - 3; i++) {
      const [p0, p1, p2, p3] = BezierMath.catmullRomToBezier(
        pts[i], pts[i+1], pts[i+2], pts[i+3], tension
      );
      segments.push({ p0, p1, p2, p3 });
    }
    return segments;
  },

  // Sample bezier path into Float32Array for WebGL rendering
  // Returns {positions: Float32Array, count: number}
  sampleBezierPath(segments, samplesPerSegment = 32) {
    if (segments.length === 0) return { positions: new Float32Array(0), count: 0 };

    const totalSamples = segments.length * samplesPerSegment + 1;
    const positions = new Float32Array(totalSamples * 3);
    let idx = 0;

    for (let s = 0; s < segments.length; s++) {
      const seg = segments[s];
      const endT = (s === segments.length - 1) ? samplesPerSegment : samplesPerSegment - 1;
      for (let i = 0; i <= endT; i++) {
        const t = i / samplesPerSegment;
        const pt = BezierMath.cubicBezier(seg.p0, seg.p1, seg.p2, seg.p3, t);
        positions[idx++] = pt[0];
        positions[idx++] = pt[1];
        positions[idx++] = pt[2];
      }
    }

    return { positions: positions.subarray(0, idx), count: idx / 3 };
  },

  // Evaluate a position on the entire path at parameter t (0-1)
  // t=0 is start, t=1 is end
  evaluatePathAt(segments, t) {
    if (segments.length === 0) return null;

    const totalT = t * segments.length;
    const segIdx = Math.min(Math.floor(totalT), segments.length - 1);
    const localT = totalT - segIdx;

    const seg = segments[segIdx];
    return BezierMath.cubicBezier(seg.p0, seg.p1, seg.p2, seg.p3, Math.min(localT, 1));
  },

  // Calculate total arc length (approximate) of the path
  arcLength(segments, samples = 100) {
    let length = 0;
    let prev = BezierMath.evaluatePathAt(segments, 0);
    for (let i = 1; i <= samples; i++) {
      const curr = BezierMath.evaluatePathAt(segments, i / samples);
      const dx = curr[0] - prev[0], dy = curr[1] - prev[1], dz = curr[2] - prev[2];
      length += Math.sqrt(dx*dx + dy*dy + dz*dz);
      prev = curr;
    }
    return length;
  },

  // Build natural cubic spline path (C2 continuous - smoother than Catmull-Rom)
  // Returns array of bezier segments {p0, p1, p2, p3} in the same format
  buildNaturalSplinePath(waypoints) {
    const numPts = waypoints.length;
    if (numPts < 2) return [];

    const n = numPts - 1; // number of segments

    // Solve tridiagonal system for second-derivative coefficients (per axis)
    // Natural boundary: c_0 = 0, c_n = 0
    // Interior: c_{j-1} + 4*c_j + c_{j+1} = 3*(P_{j+1} - 2*P_j + P_{j-1})
    const c = Array.from({ length: n + 1 }, () => [0, 0, 0]);

    if (n >= 2) {
      const m = n - 1; // number of interior unknowns
      for (let dim = 0; dim < 3; dim++) {
        // Right-hand side
        const rhs = new Array(m);
        for (let j = 0; j < m; j++) {
          const idx = j + 1; // waypoint index
          rhs[j] = 3 * (waypoints[idx + 1][dim] - 2 * waypoints[idx][dim] + waypoints[idx - 1][dim]);
        }

        // Thomas algorithm (forward sweep)
        const w = new Array(m);
        const g = new Array(m);
        w[0] = 4;
        g[0] = rhs[0];
        for (let i = 1; i < m; i++) {
          const mu = 1 / w[i - 1];
          w[i] = 4 - mu;
          g[i] = rhs[i] - mu * g[i - 1];
        }

        // Back substitution
        const sol = new Array(m);
        sol[m - 1] = g[m - 1] / w[m - 1];
        for (let i = m - 2; i >= 0; i--) {
          sol[i] = (g[i] - sol[i + 1]) / w[i];
        }

        for (let j = 0; j < m; j++) {
          c[j + 1][dim] = sol[j];
        }
      }
    }

    // Convert spline coefficients to Bezier control points
    const segments = [];
    for (let i = 0; i < n; i++) {
      const Pi = waypoints[i];
      const Pnext = waypoints[i + 1];

      // b_i = (P_{i+1} - P_i) - (2*c_i + c_{i+1}) / 3
      const b = [
        (Pnext[0] - Pi[0]) - (2 * c[i][0] + c[i + 1][0]) / 3,
        (Pnext[1] - Pi[1]) - (2 * c[i][1] + c[i + 1][1]) / 3,
        (Pnext[2] - Pi[2]) - (2 * c[i][2] + c[i + 1][2]) / 3,
      ];

      segments.push({
        p0: [...Pi],
        p1: [Pi[0] + b[0] / 3, Pi[1] + b[1] / 3, Pi[2] + b[2] / 3],
        p2: [Pi[0] + 2 * b[0] / 3 + c[i][0] / 3, Pi[1] + 2 * b[1] / 3 + c[i][1] / 3, Pi[2] + 2 * b[2] / 3 + c[i][2] / 3],
        p3: [...Pnext],
      });
    }
    return segments;
  },
};

window.BezierMath = BezierMath;
