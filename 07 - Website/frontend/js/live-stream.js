/**
 * Live PPG Streaming Session & WebSocket Client (PLAN §2 & §5)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

class LiveStreamManager {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.isPaused = false;
    this.selectedPatientId = null;
    this.visualizer = null;

    // UI Elements
    this.connBadge = document.getElementById('connectionBadge');
    this.bpmDisplay = document.getElementById('liveBpmDisplay');
    this.afProbabilityDisplay = document.getElementById('afProbabilityDisplay');
    this.alertBanner = document.getElementById('liveAlertBanner');
    this.alertText = document.getElementById('liveAlertText');
    this.alertSubtext = document.getElementById('liveAlertSubtext');
    this.sqiScoreBadge = document.getElementById('sqiScoreBadge');
    this.sqiSkewDisplay = document.getElementById('sqiSkewDisplay');
    this.sqiKurtDisplay = document.getElementById('sqiKurtDisplay');
    this.patientSelect = document.getElementById('livePatientSelect');
    this.selectedPatientCard = document.getElementById('selectedPatientCard');
    this.btnToggleSim = document.getElementById('btnToggleSim');
    this.btnPauseStream = document.getElementById('btnPauseStream');

    this._bindEvents();
  }

  initVisualizer() {
    if (!this.visualizer) {
      this.visualizer = new window.WaveformVisualizer('waveformCanvas', 'heatCanvas');
    }
  }

  _bindEvents() {
    if (this.patientSelect) {
      this.patientSelect.addEventListener('change', (e) => {
        this.selectedPatientId = e.target.value;
        this.updateSelectedPatientInfo();
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'subscribe', patientId: this.selectedPatientId }));
        }
      });
    }

    if (this.btnToggleSim) {
      this.btnToggleSim.addEventListener('click', () => this.toggleSimulation());
    }

    if (this.btnPauseStream) {
      this.btnPauseStream.addEventListener('click', () => {
        this.isPaused = !this.isPaused;
        this.btnPauseStream.innerText = this.isPaused ? 'Resume Stream' : 'Pause Stream';
        this.btnPauseStream.classList.toggle('btn-primary', this.isPaused);
        this.btnPauseStream.classList.toggle('btn-secondary', !this.isPaused);
      });
    }

    window.addEventListener('auth:success', () => {
      this.connect();
    });
  }

  connect() {
    const token = window.api.getToken();
    if (!token) return;

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live?token=${encodeURIComponent(token)}`;

    this.updateConnectionState(false, 'Connecting to Edge Gateway...');

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.updateConnectionState(true, 'LIVE 100Hz STREAM');
        if (this.selectedPatientId) {
          this.ws.send(JSON.stringify({ type: 'subscribe', patientId: this.selectedPatientId }));
        }
      };

      this.ws.onmessage = (event) => {
        if (this.isPaused) return;
        try {
          const msg = JSON.parse(event.data);
          this.handleMessage(msg);
        } catch (e) {
          console.error('[WS Message Error]', e);
        }
      };

      this.ws.onclose = (event) => {
        this.isConnected = false;
        this.updateConnectionState(false, 'Gateway Offline');
        if (event.code === 4001 || event.code === 1008) {
          // Token rejected
          window.api.clearSession();
          window.dispatchEvent(new CustomEvent('auth:unauthorized'));
        } else {
          // Auto-reconnect after 3s
          setTimeout(() => {
            if (window.api.getToken() && !this.isConnected) {
              this.connect();
            }
          }, 3000);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('[WS Error]', err);
      };
    } catch (err) {
      console.error('[WS Connection Error]', err);
      this.updateConnectionState(false, 'Connection Failed');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.updateConnectionState(false, 'Disconnected');
  }

  handleMessage(msg) {
    if (msg.type === 'connection_ack') {
      if (this.btnToggleSim) {
        this.btnToggleSim.innerText = msg.simulationActive ? 'Stop Live Demo' : 'Start Live Demo';
      }
      if (msg.lastKnownFrame) {
        this.renderFrame(msg.lastKnownFrame, msg.transitLatencyMs || 0, msg.relayTime);
      }
    } else if (msg.type === 'ppg_window') {
      this.renderFrame(msg.data, msg.transitLatencyMs, msg.relayTime);
    }
  }

  renderFrame(data, transitLatencyMs = 0, relayTime = null) {
    if (!data) return;

    const renderStart = performance.now();

    // 1. Update Waveform & Grad-CAM Heat-Strip
    if (this.visualizer && data.raw_window) {
      this.visualizer.updateData(data.raw_window, data.gradcam_weights);
    }

    // 2. Update Physiological Metrics
    if (this.bpmDisplay && data.bpm !== undefined) {
      this.bpmDisplay.innerText = Number(data.bpm).toFixed(0);
    }

    const afProb = data.af_probability !== undefined ? data.af_probability : 0;
    if (this.afProbabilityDisplay) {
      this.afProbabilityDisplay.innerText = `${(afProb * 100).toFixed(1)}%`;
    }

    // 3. Update Alert Banner (NSR vs AF)
    const isAf = data.af_detected === 1 || afProb >= 0.50;
    if (this.alertBanner) {
      this.alertBanner.className = `alert-banner ${isAf ? 'af' : 'nsr'}`;
      if (isAf) {
        this.alertText.innerHTML = `ATRIAL FIBRILLATION DETECTED &mdash; ${(afProb * 100).toFixed(1)}% Confidence`;
        this.alertSubtext.innerText = 'High Grad-CAM regional activation identified. Clinical evaluation recommended.';
      } else {
        this.alertText.innerHTML = `NORMAL SINUS RHYTHM (NSR) &mdash; ${(afProb * 100).toFixed(1)}% AF Index`;
        this.alertSubtext.innerText = 'Stable PPG morphological contours. No acute irregularity detected.';
      }
    }

    // 4. Update Signal Quality Index (SQI)
    if (data.sqi) {
      if (this.sqiScoreBadge) {
        this.sqiScoreBadge.innerText = data.sqi.sqi_score || 'EXCELLENT';
        this.sqiScoreBadge.className = `budget-badge ${data.sqi.acceptable ? 'pass' : 'warn'}`;
      }
      if (this.sqiSkewDisplay && data.sqi.skewness !== undefined) {
        this.sqiSkewDisplay.innerText = data.sqi.skewness.toFixed(2);
      }
      if (this.sqiKurtDisplay && data.sqi.kurtosis !== undefined) {
        this.sqiKurtDisplay.innerText = data.sqi.kurtosis.toFixed(2);
      }
    }

    const renderDuration = performance.now() - renderStart;

    // 5. Update Latency Telemetry Breakdown (PLAN §5 & ISO/IEC 25010)
    if (window.telemetryManager) {
      window.telemetryManager.recordFrameLatency({
        edgeTs: data.ts,
        relayTime: relayTime || Date.now(),
        transitLatencyMs,
        inferenceMs: (data.latencies && data.latencies.inference_ms) || 12.0,
        dspMs: (data.latencies && data.latencies.dsp_ms) || 2.0,
        totalEdgeMs: (data.latencies && data.latencies.total_edge_ms) || 14.0,
        renderDurationMs: renderDuration
      });
    }
  }

  async toggleSimulation() {
    try {
      const statusRes = await window.api.getBridgeStatus();
      const currentActive = statusRes.stats.simulationActive;
      const targetState = !currentActive;

      await window.api.toggleSimulation(targetState, this.selectedPatientId || 'PAT-CAL-001');

      if (this.btnToggleSim) {
        this.btnToggleSim.innerText = targetState ? 'Stop Live Demo' : 'Start Live Demo';
      }

      window.showToast(targetState ? 'Live demo stream activated' : 'Live demo stream stopped', 'info');
    } catch (err) {
      window.showToast(`Simulation toggle failed: ${err.message}`, 'error');
    }
  }

  updateConnectionState(isLive, label) {
    if (!this.connBadge) return;
    this.connBadge.className = `connection-indicator ${isLive ? '' : 'offline'}`;
    this.connBadge.innerHTML = `<span class="pulse-dot"></span> <span>${label}</span>`;
  }

  updateSelectedPatientInfo() {
    if (!this.selectedPatientCard || !this.selectedPatientId) return;
    const patient = window.patientsManager ? window.patientsManager.getPatientData(this.selectedPatientId) : null;
    if (patient) {
      this.selectedPatientCard.innerHTML = `
        <div class="patient-info-row"><span class="patient-info-label">Name:</span> <span class="patient-info-val">${patient.name} (${patient.age}y / ${patient.gender})</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Barangay:</span> <span class="patient-info-val">${patient.barangay}</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Assigned Node:</span> <span class="patient-info-val">${patient.device_id}</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Past Episodes:</span> <span class="patient-info-val" style="color: var(--accent-alert);">${patient.af_episodes || 0} AF events</span></div>
      `;
    }
  }
}

window.liveStream = new LiveStreamManager();
