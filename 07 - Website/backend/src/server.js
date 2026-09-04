/**
 * Clinician Web Dashboard Server (PLAN §1, §2, §3, §4, §5)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning
 * Authors: Bauzon, Condino, Delos Angeles, Rana, Villaflor (3CPE-2A, Univ. of the East)
 * Adviser: Dr. Nelson Rodelas
 * 
 * Port allocations:
 * - HTTP / WebSocket: 8080 (or process.env.PORT) for Clinician Dashboard & Authenticated WS /ws/live
 * - TCP Pub/Sub Bridge: 5051 (Local ingestion from Python 03 - ML/edge_inference/runner.py)
 */

const express = require('express');
const http = require('http');
const path = require('path');
const cors = require('cors');

const config = require('./config');
const db = require('./db');
const pubsubRelay = require('./bridge/pubsub_relay');

const authRoutes = require('./routes/auth');
const patientRoutes = require('./routes/patients');
const eventRoutes = require('./routes/events');
const edgeRoutes = require('./routes/edge');

const app = express();
const server = http.createServer(app);

// Middleware
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true }));

// REST API Endpoints
app.use('/api/auth', authRoutes);
app.use('/api/patients', patientRoutes);
app.use('/api/events', eventRoutes);
app.use('/api/edge', edgeRoutes);

// System Health & Telemetry Status
app.get('/api/health', (req, res) => {
  const chainValidation = db.validateLedgerChain();
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'Clinician Web Dashboard & Edge Gateway',
    location: '07 - Website/backend',
    database: {
      path: db.dbPath,
      journal_mode: 'WAL',
      chain_valid: chainValidation.valid,
      total_events: chainValidation.count
    },
    bridge: pubsubRelay.getStats()
  });
});

// Serve Frontend Static Client Assets
const frontendPath = path.resolve(__dirname, '../../frontend');
app.use(express.static(frontendPath));

// Fallback to frontend index.html for SPA client navigation
app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api/') || req.path.startsWith('/ws/')) {
    return next();
  }
  res.sendFile(path.join(frontendPath, 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('[Server Error]', err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message || 'An unexpected error occurred.'
  });
});

// Start Pub/Sub Relay (TCP on 5051 + WebSocket on /ws/live)
pubsubRelay.initialize(server);

// Start HTTP Server
server.listen(config.PORT, () => {
  console.log(`================================================================`);
  console.log(` Arrhythmia Clinician Dashboard Server is ACTIVE`);
  console.log(` HTTP & Frontend:  http://localhost:${config.PORT}`);
  console.log(` WebSocket Relay:  ws://localhost:${config.PORT}/ws/live`);
  console.log(` TCP Edge Bridge:  127.0.0.1:${config.TCP_BRIDGE_PORT}`);
  console.log(` SQLite Database:  ${db.dbPath}`);
  console.log(`================================================================`);
});

// Graceful Shutdown
function handleShutdown() {
  console.log('\n[Server] Gracefully shutting down...');
  pubsubRelay.stopSimulation();
  server.close(() => {
    console.log('[Server] HTTP and WebSocket listeners closed.');
    if (pubsubRelay.tcpServer) {
      pubsubRelay.tcpServer.close(() => {
        console.log('[Server] TCP bridge server closed.');
        process.exit(0);
      });
    } else {
      process.exit(0);
    }
  });
}

process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);

module.exports = { app, server };
