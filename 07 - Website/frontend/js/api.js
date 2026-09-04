/**
 * Clinician Dashboard API Client
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

class ApiClient {
  constructor() {
    this.baseUrl = '/api';
    this.tokenKey = 'arrhythmia_auth_token';
    this.userKey = 'arrhythmia_auth_user';
  }

  getToken() {
    return localStorage.getItem(this.tokenKey) || sessionStorage.getItem(this.tokenKey);
  }

  setSession(token, user, remember = true) {
    const storage = remember ? localStorage : sessionStorage;
    storage.setItem(this.tokenKey, token);
    storage.setItem(this.userKey, JSON.stringify(user));
  }

  clearSession() {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    sessionStorage.removeItem(this.tokenKey);
    sessionStorage.removeItem(this.userKey);
  }

  getUser() {
    const raw = localStorage.getItem(this.userKey) || sessionStorage.getItem(this.userKey);
    try {
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    };

    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers
      });

      if (response.status === 401) {
        // Session expired or invalid
        this.clearSession();
        window.dispatchEvent(new CustomEvent('auth:unauthorized'));
        throw new Error('Session expired or unauthorized. Please log in again.');
      }

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || `Request failed with status ${response.status}`);
      }

      return data;
    } catch (err) {
      console.error(`API Error [${endpoint}]:`, err.message);
      throw err;
    }
  }

  // Auth Methods
  async login(username, password) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password })
    });
    this.setSession(data.token, data.user);
    return data;
  }

  async getProfile() {
    return this.request('/auth/me', { method: 'GET' });
  }

  async logout() {
    try {
      await this.request('/auth/logout', { method: 'POST' });
    } finally {
      this.clearSession();
      window.dispatchEvent(new CustomEvent('auth:logout'));
    }
  }

  // Patient CRUD Methods
  async getPatients(search = '') {
    const q = search ? `?q=${encodeURIComponent(search)}` : '';
    return this.request(`/patients${q}`, { method: 'GET' });
  }

  async getPatient(id) {
    return this.request(`/patients/${id}`, { method: 'GET' });
  }

  async getPatientEvents(id, options = {}) {
    const params = new URLSearchParams(options);
    return this.request(`/patients/${id}/events?${params.toString()}`, { method: 'GET' });
  }

  async createPatient(patientData) {
    return this.request('/patients', {
      method: 'POST',
      body: JSON.stringify(patientData)
    });
  }

  async updatePatient(id, patientData) {
    return this.request(`/patients/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patientData)
    });
  }

  async deletePatient(id) {
    return this.request(`/patients/${id}`, { method: 'DELETE' });
  }

  // Arrhythmia Event & Verification Methods
  async getEvents(filters = {}) {
    const params = new URLSearchParams();
    if (filters.patientId) params.append('patientId', filters.patientId);
    if (filters.afOnly) params.append('afOnly', 'true');
    if (filters.syncStatus !== undefined && filters.syncStatus !== '') params.append('syncStatus', filters.syncStatus);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.offset) params.append('offset', filters.offset);

    return this.request(`/events?${params.toString()}`, { method: 'GET' });
  }

  async getEvent(id) {
    return this.request(`/events/${id}`, { method: 'GET' });
  }

  async verifyChain() {
    return this.request('/events/verify/chain', { method: 'GET' });
  }

  // Edge & Simulation Telemetry
  async getBridgeStatus() {
    return this.request('/edge/bridge/status', { method: 'GET' });
  }

  async toggleSimulation(enable, patientId = 'PAT-CAL-001') {
    return this.request('/edge/simulation/toggle', {
      method: 'POST',
      body: JSON.stringify({ enable, patientId })
    });
  }
}

window.api = new ApiClient();
