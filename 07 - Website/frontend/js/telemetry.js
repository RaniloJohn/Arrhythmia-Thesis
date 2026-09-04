/**
 * ISO/IEC 25010 Performance Efficiency & Latency Budget Tracker (PLAN §5)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Separate Budgets:
 *  - Edge 1D-CNN Inference Budget: < 25 ms (per ANTIGRAVITY.md §5 on BCM2711)
 *  - Sensor-to-Browser Round-Trip Budget: < 150 ms (Total Acquisition -> Ingest -> CNN -> WS -> Canvas)
 */

class TelemetryManager {
  constructor() {
    this.history = [];
    this.maxHistory = 60; // 1-minute rolling window @ 1Hz updates

    // Displays
    this.edgeInferDisplay = document.getElementById('telemetryInferMs');
    this.edgeDspDisplay = document.getElementById('telemetryDspMs');
    this.bridgeTransitDisplay = document.getElementById('telemetryBridgeMs');
    this.wsRelayDisplay = document.getElementById('telemetryWsMs');
    this.renderDisplay = document.getElementById('telemetryRenderMs');
    this.totalLatencyDisplay = document.getElementById('telemetryTotalMs');
    this.cnnBudgetBadge = document.getElementById('cnnBudgetBadge');
    this.totalBudgetBadge = document.getElementById('totalBudgetBadge');

    // Telemetry Tab Displays
    this.avgInferDisplay = document.getElementById('avgInferMs');
    this.avgTotalDisplay = document.getElementById('avgTotalMs');
    this.p95TotalDisplay = document.getElementById('p95TotalMs');
    this.complianceRateDisplay = document.getElementById('complianceRateDisplay');
  }

  recordFrameLatency({
    edgeTs,
    relayTime,
    transitLatencyMs,
    inferenceMs,
    dspMs,
    totalEdgeMs,
    renderDurationMs
  }) {
    const now = Date.now();
    const wsTransitMs = Math.max(0, now - relayTime);
    const totalSensorToBrowserMs = Math.max(0, (now - edgeTs) + renderDurationMs);

    const record = {
      timestamp: now,
      inferenceMs: Math.round(inferenceMs * 10) / 10,
      dspMs: Math.round(dspMs * 10) / 10,
      totalEdgeMs: Math.round(totalEdgeMs * 10) / 10,
      transitLatencyMs: Math.round(transitLatencyMs * 10) / 10,
      wsTransitMs: Math.round(wsTransitMs * 10) / 10,
      renderDurationMs: Math.round(renderDurationMs * 10) / 10,
      totalLatencyMs: Math.round(totalSensorToBrowserMs * 10) / 10,
      cnnCompliant: inferenceMs < 25.0,
      totalCompliant: totalSensorToBrowserMs < 150.0
    };

    this.history.push(record);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    this.updateLiveDisplays(record);
    this.updateAggregateDisplays();
  }

  updateLiveDisplays(r) {
    if (this.edgeInferDisplay) this.edgeInferDisplay.innerText = `${r.inferenceMs.toFixed(1)} ms`;
    if (this.edgeDspDisplay) this.edgeDspDisplay.innerText = `${r.dspMs.toFixed(1)} ms`;
    if (this.bridgeTransitDisplay) this.bridgeTransitDisplay.innerText = `${r.transitLatencyMs.toFixed(1)} ms`;
    if (this.wsRelayDisplay) this.wsRelayDisplay.innerText = `${r.wsTransitMs.toFixed(1)} ms`;
    if (this.renderDisplay) this.renderDisplay.innerText = `${r.renderDurationMs.toFixed(1)} ms`;
    if (this.totalLatencyDisplay) this.totalLatencyDisplay.innerText = `${r.totalLatencyMs.toFixed(1)} ms`;

    // Budgets
    if (this.cnnBudgetBadge) {
      this.cnnBudgetBadge.className = `budget-badge ${r.cnnCompliant ? 'pass' : 'warn'}`;
      this.cnnBudgetBadge.innerText = r.cnnCompliant ? '< 25ms (PASS)' : '> 25ms (WARN)';
    }

    if (this.totalBudgetBadge) {
      this.totalBudgetBadge.className = `budget-badge ${r.totalCompliant ? 'pass' : 'warn'}`;
      this.totalBudgetBadge.innerText = r.totalCompliant ? '< 150ms (PASS)' : '> 150ms (WARN)';
    }
  }

  updateAggregateDisplays() {
    if (this.history.length === 0) return;

    let sumInfer = 0;
    let sumTotal = 0;
    let compliantCount = 0;
    const allTotals = [];

    for (const r of this.history) {
      sumInfer += r.inferenceMs;
      sumTotal += r.totalLatencyMs;
      allTotals.push(r.totalLatencyMs);
      if (r.totalCompliant && r.cnnCompliant) compliantCount++;
    }

    allTotals.sort((a, b) => a - b);
    const p95Idx = Math.floor(allTotals.length * 0.95);
    const p95 = allTotals[p95Idx] || allTotals[allTotals.length - 1];

    const avgInfer = sumInfer / this.history.length;
    const avgTotal = sumTotal / this.history.length;
    const complianceRate = (compliantCount / this.history.length) * 100;

    if (this.avgInferDisplay) this.avgInferDisplay.innerText = `${avgInfer.toFixed(1)} ms`;
    if (this.avgTotalDisplay) this.avgTotalDisplay.innerText = `${avgTotal.toFixed(1)} ms`;
    if (this.p95TotalDisplay) this.p95TotalDisplay.innerText = `${p95.toFixed(1)} ms`;
    if (this.complianceRateDisplay) this.complianceRateDisplay.innerText = `${complianceRate.toFixed(0)}%`;
  }
}

window.telemetryManager = new TelemetryManager();
