/**
 * Application Configuration
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

const path = require('path');

module.exports = {
  PORT: parseInt(process.env.PORT || '8080', 10),
  TCP_BRIDGE_PORT: parseInt(process.env.TCP_BRIDGE_PORT || '5051', 10),
  
  // Shared SQLite database path with 03 - ML/storage/ per ADR-001 WAL-mode contract
  DB_PATH: process.env.DB_PATH || path.resolve(__dirname, '../../../03 - ML/storage/arrhythmia_edge.db'),
  
  // JWT Authentication Secret and Expiry
  JWT_SECRET: process.env.JWT_SECRET || 'ue-caloocan-arrhythmia-secret-key-2026-sha256',
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '12h',
  
  // Target sensor-to-browser latency budget in milliseconds (ISO/IEC 25010)
  SENSOR_TO_BROWSER_BUDGET_MS: 150.0,
  INFERENCE_BUDGET_MS: 25.0
};
