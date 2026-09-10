/**
 * Real-Time PPG Waveform & 1D Grad-CAM Explainability Visualizer (PLAN §2 & RQ3)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Features:
 * - High-speed HTML5 Canvas waveform renderer (60 FPS, Retinal devicePixelRatio aware).
 * - ECG/PPG medical monitor grid with millimeter scaling simulation.
 * - 1D Grad-CAM Relevance Vector rendered as a synchronized heat-strip under the waveform.
 * - Dynamic color shading on waveform peaks based on Grad-CAM attention weights.
 */

class WaveformVisualizer {
  constructor(canvasId, heatCanvasId) {
    this.canvas = document.getElementById(canvasId);
    this.heatCanvas = document.getElementById(heatCanvasId);
    
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.heatCtx = this.heatCanvas ? this.heatCanvas.getContext('2d') : null;

    this.points = [];
    this.weights = [];
    this.maxPoints = 250; // Visual display window

    this.animationFrameId = null;
    this.isRendering = false;

    this._setupResizeObserver();
    this.resizeCanvases();
  }

  _setupResizeObserver() {
    window.addEventListener('resize', () => {
      this.resizeCanvases();
      this.render();
    });
  }

  resizeCanvases() {
    if (!this.canvas || !this.heatCanvas) return;

    const dpr = window.devicePixelRatio || 1;
    
    // Waveform Canvas
    const wRect = this.canvas.parentElement.getBoundingClientRect();
    const wWidth = Math.floor(wRect.width);
    const wHeight = Math.floor(wRect.height);

    this.canvas.width = wWidth * dpr;
    this.canvas.height = wHeight * dpr;
    this.ctx.scale(dpr, dpr);
    this.wWidth = wWidth;
    this.wHeight = wHeight;

    // Heat Canvas
    const hRect = this.heatCanvas.getBoundingClientRect();
    const hWidth = Math.floor(hRect.width);
    const hHeight = Math.floor(hRect.height);

    this.heatCanvas.width = hWidth * dpr;
    this.heatCanvas.height = hHeight * dpr;
    this.heatCtx.scale(dpr, dpr);
    this.hWidth = hWidth;
    this.hHeight = hHeight;
  }

  /**
   * Updates display data from incoming edge inference window.
   * @param {Array<number>} rawWindow - Array of filtered pulsatile PPG points
   * @param {Array<number>} gradcamWeights - Array of Grad-CAM relevance scores [0.0 - 1.0]
   */
  /**
   * Blanks the trace and heat strip, leaving only the grid. Used when the edge
   * reports no valid measurement — returning early instead would leave the last
   * good waveform frozen on screen, which reads as a live signal.
   */
  clear() {
    this.points = [];
    this.weights = [];
    this.render();
  }

  updateData(rawWindow, gradcamWeights) {
    if (!Array.isArray(rawWindow) || rawWindow.length === 0) {
      this.clear();
      return;
    }

    this.points = rawWindow;
    this.weights = gradcamWeights || new Array(rawWindow.length).fill(0.1);
    this.render();
  }

  /**
   * Maps Grad-CAM relevance score [0.0 - 1.0] to a clinical heatmap color.
   */
  getHeatColor(weight) {
    const w = Math.max(0, Math.min(1, weight));
    if (w < 0.4) {
      // Low relevance: calm deep teal (#0E6B76)
      return `rgba(14, 107, 118, ${0.4 + w * 0.6})`;
    } else if (w < 0.7) {
      // Moderate relevance: warning amber (#B5792B)
      return `rgba(181, 121, 43, 0.85)`;
    } else {
      // High clinical relevance (arrhythmia trigger zone): clinical red (#C13B3B)
      return `rgba(193, 59, 59, ${0.85 + (w - 0.7) * 0.5})`;
    }
  }

  /**
   * Draws medical monitor grid lines.
   */
  drawGrid() {
    const ctx = this.ctx;
    const w = this.wWidth;
    const h = this.wHeight;

    ctx.clearRect(0, 0, w, h);

    // Light clinical surface background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, w, h);

    // Minor Grid (Clinical ECG millimeter grid style)
    ctx.strokeStyle = 'rgba(14, 107, 118, 0.07)';
    ctx.lineWidth = 1;
    const minorStep = 20;

    ctx.beginPath();
    for (let x = 0; x < w; x += minorStep) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, h);
    }
    for (let y = 0; y < h; y += minorStep) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(w, y + 0.5);
    }
    ctx.stroke();

    // Major Grid
    ctx.strokeStyle = 'rgba(14, 107, 118, 0.16)';
    ctx.lineWidth = 1;
    const majorStep = 100;

    ctx.beginPath();
    for (let x = 0; x < w; x += majorStep) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, h);
    }
    for (let y = 0; y < h; y += majorStep) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(w, y + 0.5);
    }
    ctx.stroke();
  }

  /**
   * Renders the PPG Waveform and Grad-CAM Attention Heat-Strip.
   */
  render() {
    if (!this.ctx || !this.heatCtx) return;
    if (this.wWidth === 0 || this.wHeight === 0) this.resizeCanvases();

    // 1. Draw Grid
    this.drawGrid();

    const pts = this.points;
    const weights = this.weights;
    const len = pts.length;

    if (len < 2) {
      // Placeholder text when no stream
      this.ctx.fillStyle = '#6B6862';
      this.ctx.font = '14px "Public Sans", -apple-system, sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText('Awaiting live PPG stream from edge gateway...', this.wWidth / 2, this.wHeight / 2);
      return;
    }

    const ctx = this.ctx;
    const w = this.wWidth;
    const h = this.wHeight;

    // Determine scale
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < len; i++) {
      if (pts[i] < minVal) minVal = pts[i];
      if (pts[i] > maxVal) maxVal = pts[i];
    }
    const range = (maxVal - minVal) || 1.0;
    const padding = h * 0.15;
    const drawHeight = h - 2 * padding;

    // Helper to calculate coordinate
    const getX = (i) => (i / (len - 1)) * w;
    const getY = (val) => h - (padding + ((val - minVal) / range) * drawHeight);

    // 2. Draw Grad-CAM Background Highlight Bands on Waveform
    for (let i = 0; i < len - 1; i++) {
      const weight = weights[i] || 0;
      if (weight > 0.6) {
        const x1 = getX(i);
        const x2 = getX(i + 1);
        ctx.fillStyle = weight > 0.75 ? 'rgba(193, 59, 59, 0.12)' : 'rgba(181, 121, 43, 0.10)';
        ctx.fillRect(x1, 0, (x2 - x1) + 1, h);
      }
    }

    // 3. Draw Waveform Path (Clean flat stroke, zero neon blur)
    ctx.save();
    ctx.lineWidth = 2.2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let i = 0; i < len - 1; i++) {
      const x1 = getX(i);
      const y1 = getY(pts[i]);
      const x2 = getX(i + 1);
      const y2 = getY(pts[i + 1]);

      const weight = weights[i] || 0;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      if (weight > 0.7) {
        ctx.strokeStyle = '#C13B3B'; // Clinical red for AF region
      } else if (weight > 0.4) {
        ctx.strokeStyle = '#B5792B'; // Clinical amber
      } else {
        ctx.strokeStyle = '#0E6B76'; // Clinical teal
      }

      ctx.stroke();
    }
    ctx.restore();

    // 4. Render Grad-CAM Heat-Strip Canvas (Under the Waveform)
    this.renderHeatStrip(weights);
  }

  /**
   * Renders the 1D Grad-CAM attention vector as a contiguous color-coded bar.
   */
  renderHeatStrip(weights) {
    const hCtx = this.heatCtx;
    const hw = this.hWidth;
    const hh = this.hHeight;
    const len = weights.length;

    hCtx.clearRect(0, 0, hw, hh);

    if (len === 0) return;

    const sliceWidth = hw / len;

    for (let i = 0; i < len; i++) {
      const weight = weights[i];
      const x = i * sliceWidth;
      hCtx.fillStyle = this.getHeatColor(weight);
      hCtx.fillRect(x, 0, sliceWidth + 1, hh);
    }
  }
}

window.WaveformVisualizer = WaveformVisualizer;
