/**
 * Clinician Authentication Controller (PLAN §3)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

class AuthManager {
  constructor() {
    this.loginView = document.getElementById('loginView');
    this.loginForm = document.getElementById('loginForm');
    this.usernameInput = document.getElementById('usernameInput');
    this.passwordInput = document.getElementById('passwordInput');
    this.loginError = document.getElementById('loginError');
    this.userBadge = document.getElementById('userProfileBadge');
    this.userNameSpan = document.getElementById('userNameDisplay');
    this.userRoleSpan = document.getElementById('userRoleDisplay');
    this.btnLogout = document.getElementById('btnLogout');

    this._bindEvents();
  }

  _bindEvents() {
    if (this.loginForm) {
      this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
    }

    if (this.btnLogout) {
      this.btnLogout.addEventListener('click', () => this.handleLogout());
    }

    // Custom unauthorized event from API client
    window.addEventListener('auth:unauthorized', () => {
      this.showLogin('Your session has expired. Please sign in again.');
    });

    window.addEventListener('auth:logout', () => {
      this.showLogin();
    });
  }

  async checkInitialAuth() {
    const token = window.api.getToken();
    if (!token) {
      this.showLogin();
      return false;
    }

    try {
      const res = await window.api.getProfile();
      this.hideLogin(res.user);
      return true;
    } catch (e) {
      this.showLogin();
      return false;
    }
  }

  async handleLogin(e) {
    e.preventDefault();
    this.loginError.style.display = 'none';

    const username = this.usernameInput.value.trim();
    const password = this.passwordInput.value.trim();

    if (!username || !password) {
      this.showError('Please enter both username and password.');
      return;
    }

    try {
      const btn = this.loginForm.querySelector('button[type="submit"]');
      btn.disabled = true;
      btn.innerText = 'Authenticating...';

      const data = await window.api.login(username, password);
      this.hideLogin(data.user);
      window.dispatchEvent(new CustomEvent('auth:success', { detail: data.user }));
    } catch (err) {
      this.showError(err.message || 'Authentication failed. Please check credentials.');
    } finally {
      const btn = this.loginForm.querySelector('button[type="submit"]');
      if (btn) {
        btn.disabled = false;
        btn.innerText = 'Sign In to Dashboard';
      }
    }
  }

  async handleLogout() {
    try {
      await window.api.logout();
    } catch (e) {
      // Ignored, session cleared locally
    }
    this.showLogin();
  }

  showLogin(message = '') {
    if (this.loginView) this.loginView.style.display = 'flex';
    if (this.userBadge) this.userBadge.style.display = 'none';
    if (message) this.showError(message);
    if (window.liveStream) window.liveStream.disconnect();
  }

  hideLogin(user) {
    if (this.loginView) this.loginView.style.display = 'none';
    if (this.userBadge) this.userBadge.style.display = 'flex';
    if (this.userNameSpan) this.userNameSpan.innerText = user.fullName || user.username;
    if (this.userRoleSpan) this.userRoleSpan.innerText = user.role;
  }

  showError(msg) {
    if (this.loginError) {
      this.loginError.innerText = msg;
      this.loginError.style.display = 'block';
    }
  }
}

window.authManager = new AuthManager();
