/**
 * Clinician Authentication Routes (PLAN §3)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const db = require('../db');
const config = require('../config');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

/**
 * POST /api/auth/login
 * Authenticates clinician or admin with bcrypt-hashed credentials.
 */
router.post('/login', async (req, res) => {
  const { username, password } = req.body || {};

  if (!username || !password) {
    return res.status(400).json({
      error: 'Bad Request',
      message: 'Username and password are required.'
    });
  }

  const user = db.getUserByUsername(username);
  if (!user) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid credentials. User not found.'
    });
  }

  const isMatch = await bcrypt.compare(password, user.password_hash);
  if (!isMatch) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid credentials. Incorrect password.'
    });
  }

  const payload = {
    userId: user.user_id,
    username: user.username,
    role: user.role,
    fullName: user.full_name
  };

  const token = jwt.sign(payload, config.JWT_SECRET, {
    expiresIn: config.JWT_EXPIRES_IN
  });

  return res.json({
    message: 'Authentication successful',
    token,
    user: payload
  });
});

/**
 * GET /api/auth/me
 * Returns profile information for the active session.
 */
router.get('/me', authenticateToken, (req, res) => {
  const user = db.getUserById(req.user.userId);
  if (!user) {
    return res.status(404).json({ error: 'Not Found', message: 'User record not found.' });
  }
  return res.json({
    user: {
      userId: user.user_id,
      username: user.username,
      role: user.role,
      fullName: user.full_name,
      createdAt: user.created_at
    }
  });
});

/**
 * POST /api/auth/logout
 * Acknowledges session termination.
 */
router.post('/logout', authenticateToken, (req, res) => {
  return res.json({ message: 'Session successfully terminated.' });
});

module.exports = router;
