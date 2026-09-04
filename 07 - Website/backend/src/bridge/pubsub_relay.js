/**
 * Local Pub/Sub Bridge & Authenticated WebSocket Relay (PLAN §2 & §5)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Functions:
 * 1. TCP Server (Port 5051): Ingests high-throughput JSON from Python edge process.
 * 2. WebSocket Server (/ws/live): Re-broadcasts telemetry exclusively to authenticated browser clients.
 * 3. Latency instrumentation: Tracks sensor->edge->bridge->browser latency stages.
 * 4. Fallback Standalone Simulator: Provides realistic physiological stream for clinician evaluation.
 */

const net = require('net');
const { WebSocketServer, WebSocket } = require('ws');
const { parse } = require('url');
const config = require('../config');
const { verifySocketToken } = require('../middleware/auth');

class PubSubRelay {
  constructor() {
    this.tcpServer = null;
    this.wss = null;
    this.clients = new Set(); // Set of { ws, user, subscribedPatientId }
    this.tcpClients = new Set();
    this.tcpBuffer = '';

    // Simulator state
    this.simulationTimer = null;
    this.simRunning = false;
    this.simSampleIndex = 0;

    // Latency & performance telemetry statistics
    this.stats = {
      framesReceived: 0,
      framesRelayed: 0,
      activeWsClients: 0,
      lastFrameTs: null,
      avgBridgeLatencyMs: 0,
      latestPayload: null
    };
  }

  /**
   * Initialize both TCP pub/sub bridge and WebSocket relay on the HTTP server.
   */
  initialize(httpServer) {
    this._initTcpServer();
    this._initWebSocketServer(httpServer);
  }

  /**
   * TCP Bridge Server: receives newline-delimited JSON from Python edge_inference/runner.py
   */
  _initTcpServer() {
    this.tcpServer = net.createServer((socket) => {
      const clientAddr = `${socket.remoteAddress}:${socket.remotePort}`;
      console.log(`[TCP Bridge] Edge runner connected from ${clientAddr}`);
      this.tcpClients.add(socket);

      socket.on('data', (chunk) => {
        this.tcpBuffer += chunk.toString('utf8');
        let boundary;
        while ((boundary = this.tcpBuffer.indexOf('\n')) !== -1) {
          const line = this.tcpBuffer.substring(0, boundary).trim();
          this.tcpBuffer = this.tcpBuffer.substring(boundary + 1);
          if (line) {
            try {
              const payload = JSON.parse(line);
              this.handleIncomingWindow(payload, 'tcp_bridge');
            } catch (err) {
              console.error('[TCP Bridge] JSON parse error:', err.message);
            }
          }
        }
      });

      socket.on('close', () => {
        console.log(`[TCP Bridge] Edge runner disconnected: ${clientAddr}`);
        this.tcpClients.delete(socket);
      });

      socket.on('error', (err) => {
        console.warn(`[TCP Bridge] Socket error (${clientAddr}):`, err.message);
        this.tcpClients.delete(socket);
      });
    });

    this.tcpServer.listen(config.TCP_BRIDGE_PORT, '127.0.0.1', () => {
      console.log(`[TCP Bridge] Listening on 127.0.0.1:${config.TCP_BRIDGE_PORT} for Python edge inference frames`);
    });

    this.tcpServer.on('error', (err) => {
      console.error('[TCP Bridge] Server error:', err.message);
    });
  }

  /**
   * WebSocket Server: relays live telemetry to authenticated browser sessions.
   */
  _initWebSocketServer(httpServer) {
    this.wss = new WebSocketServer({ noServer: true });

    // Intercept upgrade requests on /ws/live
    httpServer.on('upgrade', (request, socket, head) => {
      const parsedUrl = parse(request.url, true);
      const pathname = parsedUrl.pathname;

      if (pathname === '/ws/live') {
        // ENFORCE AUTHENTICATION (PLAN §2 & §3)
        const token = parsedUrl.query.token;
        const user = verifySocketToken(token);

        if (!user) {
          console.warn('[WebSocket] Rejected unauthenticated connection attempt');
          socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
          socket.destroy();
          return;
        }

        this.wss.handleUpgrade(request, socket, head, (ws) => {
          this.wss.emit('connection', ws, request, user);
        });
      }
    });

    this.wss.on('connection', (ws, req, user) => {
      const clientInfo = {
        ws,
        user,
        subscribedPatientId: null,
        connectedAt: Date.now()
      };
      this.clients.add(clientInfo);
      this.stats.activeWsClients = this.clients.size;

      console.log(`[WebSocket] Authenticated clinician connected: ${user.username} (${user.role}). Total active: ${this.clients.size}`);

      // Send initial handshake and state
      ws.send(JSON.stringify({
        type: 'connection_ack',
        user: { username: user.username, role: user.role, fullName: user.fullName },
        serverTime: Date.now(),
        simulationActive: this.simRunning,
        lastKnownFrame: this.stats.latestPayload || null
      }));

      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message.toString());
          if (data.type === 'subscribe') {
            clientInfo.subscribedPatientId = data.patientId || null;
            ws.send(JSON.stringify({
              type: 'subscribe_ack',
              subscribedPatientId: clientInfo.subscribedPatientId
            }));
          } else if (data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong', clientTime: data.time, serverTime: Date.now() }));
          } else if (data.type === 'toggle_simulation') {
            // Allow authenticated clinician to toggle simulation if hardware runner is offline
            if (data.enable) {
              this.startSimulation(data.patientId);
            } else {
              this.stopSimulation();
            }
          }
        } catch (e) {
          console.error('[WebSocket] Message handling error:', e.message);
        }
      });

      ws.on('close', () => {
        this.clients.delete(clientInfo);
        this.stats.activeWsClients = this.clients.size;
        console.log(`[WebSocket] Clinician disconnected: ${user.username}. Active: ${this.clients.size}`);
      });

      ws.on('error', (err) => {
        console.warn(`[WebSocket] Client error (${user.username}):`, err.message);
        this.clients.delete(clientInfo);
        this.stats.activeWsClients = this.clients.size;
      });
    });
  }

  /**
   * Processes incoming window payload from Python runner (or HTTP fallback/simulation).
   * Stamps latency tracking and broadcasts to authenticated WebSocket sessions.
   */
  handleIncomingWindow(payload, source = 'tcp_bridge') {
    const bridgeRecvTime = Date.now();
    this.stats.framesReceived++;
    this.stats.lastFrameTs = bridgeRecvTime;
    this.stats.latestPayload = payload;

    // Calculate transit latency between edge window generation and bridge reception
    const edgeTs = payload.ts || bridgeRecvTime;
    const bridgeTransitMs = Math.max(0, bridgeRecvTime - edgeTs);
    this.stats.avgBridgeLatencyMs = this.stats.avgBridgeLatencyMs === 0
      ? bridgeTransitMs
      : Math.round((this.stats.avgBridgeLatencyMs * 0.9 + bridgeTransitMs * 0.1) * 10) / 10;

    // Package framed transmission with complete latency budget breakdown (ISO/IEC 25010)
    const outboundMessage = JSON.stringify({
      type: 'ppg_window',
      source,
      bridgeRecvTime,
      transitLatencyMs: bridgeTransitMs,
      relayTime: Date.now(),
      data: payload
    });

    // Broadcast to authenticated WebSocket clients
    let relayedCount = 0;
    for (const client of this.clients) {
      if (client.ws.readyState === WebSocket.OPEN) {
        // Filter by subscription if set
        if (!client.subscribedPatientId || client.subscribedPatientId === payload.patient_id) {
          client.ws.send(outboundMessage);
          relayedCount++;
        }
      }
    }

    this.stats.framesRelayed += relayedCount;
  }

  /**
   * Standalone Simulation Generator (for bench testing without physical Pi/ESP32).
   * Produces realistic 1-second stepped windows (250 display points, 100Hz equivalent).
   */
  startSimulation(patientId = 'PAT-CAL-001') {
    if (this.simRunning) return;
    this.simRunning = true;
    console.log(`[Simulator] Starting standalone physiological stream simulation for ${patientId}...`);

    let t = 0;
    let mode = 'nsr'; // Alternates between 'nsr' and 'af'
    let stepCount = 0;

    this.simulationTimer = setInterval(() => {
      stepCount++;
      // Toggle rhythm state every 15 windows (15 seconds) to demonstrate Grad-CAM explainability
      if (stepCount % 15 === 0) {
        mode = mode === 'nsr' ? 'af' : 'nsr';
        console.log(`[Simulator] Switching simulated rhythm mode to: ${mode.toUpperCase()}`);
      }

      const isAf = mode === 'af';
      const windowLength = 250; // Visual display downsampled window
      const rawWindow = [];
      const gradcamWeights = [];

      const baseHr = isAf ? 118.0 + Math.sin(stepCount * 0.5) * 15.0 : 72.0 + Math.sin(stepCount * 0.2) * 4.0;
      const bpm = Math.round(baseHr * 10) / 10;

      for (let i = 0; i < windowLength; i++) {
        t += 0.04; // 25 Hz display rate step
        const freq = (bpm / 60.0);
        const phase = (2 * Math.PI * freq * t) % (2 * Math.PI);

        let pulse;
        let relevance;

        if (!isAf) {
          // Normal Sinus Rhythm: clear systolic peak + dicrotic wave
          const systolic = Math.exp(-Math.pow(phase - 1.2, 2) / 0.18);
          const dicrotic = 0.32 * Math.exp(-Math.pow(phase - 2.3, 2) / 0.28);
          pulse = (systolic + dicrotic) * 1.5 - 0.4 + (Math.random() - 0.5) * 0.03;
          // Normal rhythm: Grad-CAM has low attention
          relevance = Math.max(0.05, Math.min(0.25, 0.10 + (Math.random() - 0.5) * 0.05));
        } else {
          // Atrial Fibrillation: irregular pulse amplitude, erratic peaks, missing dicrotic wave
          const amp = 0.7 + 0.6 * Math.sin(t * 1.8);
          const erraticPhase = (phase + (Math.sin(t * 3.1) * 0.4)) % (2 * Math.PI);
          const systolic = amp * Math.exp(-Math.pow(erraticPhase - 1.2, 2) / 0.22);
          const noise = (Math.random() - 0.5) * 0.12;
          pulse = systolic * 1.4 - 0.3 + noise;

          // Grad-CAM focuses strongly on the irregular, erratic peak intervals
          const isIrregularZone = (phase > 0.8 && phase < 1.8) || (amp > 1.0);
          relevance = isIrregularZone
            ? Math.min(0.98, 0.75 + Math.random() * 0.22)
            : Math.max(0.12, 0.25 + Math.random() * 0.20);
        }

        rawWindow.push(Math.round(pulse * 100) / 100);
        gradcamWeights.push(Math.round(relevance * 1000) / 1000);
      }

      // Latencies simulated within hardware benchmark targets
      const inferMs = Math.round((12.5 + Math.random() * 4.0) * 10) / 10; // Well within <25ms budget
      const dspMs = Math.round((1.8 + Math.random() * 0.6) * 10) / 10;
      const totalEdgeMs = Math.round((inferMs + dspMs + 1.1) * 10) / 10;

      const payload = {
        ts: Date.now(),
        iso_time: new Date().toISOString(),
        patient_id: patientId,
        device_id: 'ESP32C3-NODE-01',
        bpm,
        af_detected: isAf ? 1 : 0,
        af_probability: isAf ? Math.round((0.88 + Math.random() * 0.09) * 10000) / 10000 : Math.round((0.02 + Math.random() * 0.05) * 10000) / 10000,
        sqi: {
          skewness: isAf ? 0.32 : 0.84,
          kurtosis: isAf ? 2.65 : 3.82,
          perfusion_index_pct: 1.85,
          acceptable: true,
          sqi_score: isAf ? 'MODERATE' : 'EXCELLENT'
        },
        hrv: {
          rmssd_ms: isAf ? 78.4 : 26.2,
          cv_ibi: isAf ? 0.24 : 0.06,
          irregular: isAf
        },
        raw_window: rawWindow,
        gradcam_weights: gradcamWeights,
        event_id: isAf ? `sim-evt-${Date.now()}` : null,
        latencies: {
          dsp_ms: dspMs,
          inference_ms: inferMs,
          total_edge_ms: totalEdgeMs,
          budget_ms: config.INFERENCE_BUDGET_MS,
          budget_met: inferMs < config.INFERENCE_BUDGET_MS
        }
      };

      this.handleIncomingWindow(payload, 'standalone_sim');
    }, 1000); // 1-second update step
  }

  stopSimulation() {
    if (this.simulationTimer) {
      clearInterval(this.simulationTimer);
      this.simulationTimer = null;
    }
    this.simRunning = false;
    console.log('[Simulator] Standalone simulation stopped.');
  }

  getStats() {
    return {
      ...this.stats,
      simulationActive: this.simRunning,
      tcpConnectedCount: this.tcpClients.size,
      wsConnectedCount: this.clients.size
    };
  }
}

module.exports = new PubSubRelay();
