/* consent.js — GDPR Cookie Consent v2 with Consent Mode v2 support */
(function () {
  'use strict';

  var CONSENT_KEY = 'cookie_consent_v1';
  var BANNER_ID   = 'cookie-consent-banner';

  function getCookie(name) {
    var escapedName = name.replace(/([.*+?^=!:${}()|[\]/\\])/g, '\\$1');
    var pattern = '(?:^|;)\\s*' + escapedName + '=([^;]*)';
    var m = document.cookie.match(new RegExp(pattern));
    return m ? decodeURIComponent(m[1]) : null;
  }

  function setCookie(name, value, days) {
    var expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie =
      name + '=' + encodeURIComponent(value) +
      '; expires=' + expires +
      '; path=/' +
      '; SameSite=Lax';
  }

  function removeBanner() {
    var el = document.getElementById(BANNER_ID);
    if (el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(100%)';
      setTimeout(function () {
        if (el.parentNode) el.remove();
      }, 350);
    }
  }

  function pushConsentDefault(granted) {
    window.dataLayer = window.dataLayer || [];
    function gtag() { window.dataLayer.push(arguments); }
    var state = granted ? 'granted' : 'denied';
    gtag('consent', 'default', {
      ad_storage:             state,
      ad_user_data:           state,
      ad_personalization:     state,
      analytics_storage:      state,
      functionality_storage:  state,
      personalization_storage: state,
      wait_for_update: granted ? 0 : 500
    });
  }

  function updateConsent(granted) {
    if (typeof gtag !== 'function') {
      window.dataLayer = window.dataLayer || [];
      window.gtag = function () { window.dataLayer.push(arguments); };
    }
    var state = granted ? 'granted' : 'denied';
    gtag('consent', 'update', {
      ad_storage:             state,
      ad_user_data:           state,
      ad_personalization:     state,
      analytics_storage:      state,
      functionality_storage:  state,
      personalization_storage: state
    });
    window.dataLayer.push({
      event: 'consent_update',
      consent_granted: granted
    });
  }

  function accept() {
    setCookie(CONSENT_KEY, 'accepted', 365);
    updateConsent(true);
    removeBanner();
  }

  function decline() {
    setCookie(CONSENT_KEY, 'declined', 180);
    updateConsent(false);
    removeBanner();
  }

  function showBanner(privacyUrl) {
    if (document.getElementById(BANNER_ID)) return;

    var banner = document.createElement('div');
    banner.id = BANNER_ID;
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.setAttribute('aria-modal', 'false');
    banner.setAttribute('aria-live', 'polite');

    banner.innerHTML = [
      '<div style="max-width:760px;margin:0 auto;display:flex;flex-wrap:wrap;',
      'align-items:center;gap:0.75rem 1.5rem;justify-content:space-between;">',
      '<p style="margin:0;font-size:0.86rem;color:#333;flex:1 1 260px;line-height:1.5;">',
      'We use cookies to improve your experience and serve relevant ads. ',
      '<a href="' + privacyUrl + '" style="color:#6366f1;text-decoration:underline;">',
      'Privacy Policy</a>',
      '</p>',
      '<div style="display:flex;gap:0.5rem;flex-shrink:0;">',
      '<button id="cc-accept" aria-label="Accept cookies" style="background:#6366f1;color:#fff;border:none;padding:0.45rem 1.1rem;border-radius:20px;cursor:pointer;font-size:0.84rem;font-weight:600;">Accept</button>',
      '<button id="cc-decline" aria-label="Decline cookies" style="background:#f0f0f0;color:#555;border:none;padding:0.45rem 1.1rem;border-radius:20px;cursor:pointer;font-size:0.84rem;">Decline</button>',
      '</div></div>'
    ].join('');

    Object.assign(banner.style, {
      position:           'fixed',
      bottom:             '0',
      left:               '0',
      right:              '0',
      background:         'rgba(255,255,255,0.97)',
      backdropFilter:     'blur(8px)',
      WebkitBackdropFilter: 'blur(8px)',
      borderTop:          '1px solid #e0e0e0',
      padding:            '0.9rem 1.25rem',
      zIndex:             '99999',
      boxShadow:          '0 -2px 16px rgba(0,0,0,0.07)',
      opacity:            '0',
      transform:          'translateY(20px)',
      transition:         'opacity 0.3s ease, transform 0.3s ease'
    });

    document.body.appendChild(banner);

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        banner.style.opacity = '1';
        banner.style.transform = 'translateY(0)';
      });
    });

    document.getElementById('cc-accept').addEventListener('click', accept);
    document.getElementById('cc-decline').addEventListener('click', decline);

    document.addEventListener('keydown', function onKey(e) {
      if (e.key === 'Escape') {
        decline();
        document.removeEventListener('keydown', onKey);
      }
    });
  }

  var existing = getCookie(CONSENT_KEY);

  if (existing === 'accepted') {
    pushConsentDefault(true);
    updateConsent(true);
  } else if (existing === 'declined') {
    pushConsentDefault(false);
    updateConsent(false);
  } else {
    pushConsentDefault(false);

    var basePath = (
      document.querySelector('meta[name="base-path"]') &&
      document.querySelector('meta[name="base-path"]').getAttribute('content')
    ) || '';

    var privacyUrl = basePath + '/privacy-policy/';

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () {
        showBanner(privacyUrl);
      });
    } else {
      showBanner(privacyUrl);
    }
  }
}());
