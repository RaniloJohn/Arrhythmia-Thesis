/**
 * Clinician Authentication & RBAC Middleware (PLAN §3 & ANTIGRAVITY.md §5)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

const jwt = require('jsonwebtoken');
const config = require('../config');

/**
 * Extracts and verifies JWT token from Authorization header or URL query parameter.
 */
function authenticateToken(req, res, next) {
  let token = null;

  const authHeader = req.headers['authorization'];
  if (authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.substring(7);
  } else if (req.query && req.query.token) {
    token = req.query.token;
  }

  if (!token) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Access denied. Authenticated clinician session required.'
    });
  }

  try {
    const decoded = jwt.verify(token, config.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Session expired or invalid token. Please log in again.'
    });
  }
}

/**
 * Role-Based Access Control (RBAC) guard.
 * Supports single role or array of allowed roles.
 */
function requireRole(allowedRoles) {
  const roles = Array.isArray(allowedRoles) ? allowedRoles : [allowedRoles];

  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({
        error: 'Forbidden',
        message: `Action restricted. Required role(s): [${roles.join(', ')}]. Current: ${req.user ? req.user.role : 'none'}`
      });
    }
    next();
  };
}

/**
 * Verify token for WebSocket upgrade or connection verification.
 * Returns decoded user payload or null if invalid.
 */
function verifySocketToken(token) {
  if (!token) return null;
  try {
    return jwt.verify(token, config.JWT_SECRET);
  } catch (e) {
    return null;
  }
}

module.exports = {
  authenticateToken,
  requireRole,
  verifySocketToken
};
