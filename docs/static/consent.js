/* consent.js — GDPR Cookie Consent v2 with Consent Mode v2 support
 *
 * NOTE: the *default* consent signal (granted/denied) is now pushed
 * synchronously via a tiny inline <script> in <head> (emitted directly
 * in each page template) BEFORE this file is even requested. That
 * inline snippet has zero network cost and guarantees Consent Mode
 * defaults are set before the async GA/AdSense tags execute.
 *
 * This file is now loaded with `defer` — it only needs to run before
 * the user can interact with the page, not before any tag fires. It
 * still re-asserts the consent state on load (harmless, idempotent)
 * and owns all banner UI / accept-decline logic.
 */
(function () {
  'use strict';

  var CONSENT_KEY = 'cookie_consent_v1';
  var BANNER_ID   = 'cookie-consent-banner';
  var STYLE_ID    = 'cookie-consent-banner-styles';

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
      el.style.transform = 'translateY(20px)';
      setTimeout(function () {
        if (el.parentNode) el.remove();
      }, 300);
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

  // FIX: banner styling now lives in an injected <style> tag (rather than
  // inline JS styles) so it can respond to `prefers-color-scheme` — dark
  // theme by default, lighter theme for users who have light mode set,
  // matching the design used elsewhere on the site. This also gives us
  // a proper :focus-visible ring for keyboard users.
  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = [
      '#' + BANNER_ID + '{',
        'position:fixed;bottom:0;left:0;right:0;z-index:99999;',
        'background:#1e1b4b;color:#e0e7ff;padding:16px 24px;',
        'font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.5;',
        'box-shadow:0 -4px 24px rgba(0,0,0,0.3);',
        'opacity:0;transform:translateY(20px);',
        'transition:opacity .3s ease,transform .3s ease;',
      '}',
      '#' + BANNER_ID + ' .cc-inner{',
        'max-width:1024px;margin:0 auto;display:flex;',
        'align-items:center;gap:16px;flex-wrap:wrap;',
      '}',
      '#' + BANNER_ID + ' .cc-text{flex:1 1 260px;min-width:240px;margin:0;}',
      '#' + BANNER_ID + ' .cc-text a{color:#a5b4fc;}',
      '#' + BANNER_ID + ' .cc-buttons{display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap;}',
      '#' + BANNER_ID + ' .cc-btn{',
        'padding:8px 18px;border-radius:6px;border:none;',
        'font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap;',
      '}',
      '#' + BANNER_ID + ' .cc-btn:focus-visible{outline:3px solid #818cf8;outline-offset:2px;}',
      '#' + BANNER_ID + ' .cc-accept{background:#6366f1;color:#fff;}',
      '#' + BANNER_ID + ' .cc-accept:hover{background:#4f46e5;}',
      '#' + BANNER_ID + ' .cc-decline{background:transparent;color:#e0e7ff;border:1px solid #6366f1;}',
      '#' + BANNER_ID + ' .cc-decline:hover{background:rgba(99,102,241,0.15);}',
      '@media (prefers-color-scheme: light){',
        '#' + BANNER_ID + '{background:#f5f3ff;color:#1e1b4b;box-shadow:0 -4px 24px rgba(0,0,0,0.12);}',
        '#' + BANNER_ID + ' .cc-text a{color:#4f46e5;}',
        '#' + BANNER_ID + ' .cc-decline{color:#1e1b4b;border-color:#4f46e5;}',
        '#' + BANNER_ID + ' .cc-decline:hover{background:rgba(79,70,229,0.08);}',
      '}'
    ].join('');
    document.head.appendChild(style);
  }

  function showBanner(privacyUrl) {
    if (document.getElementById(BANNER_ID)) return;

    injectStyles();

    var banner = document.createElement('div');
    banner.id = BANNER_ID;
    // FIX: role="dialog" + aria-modal="true" + aria-hidden management,
    // matching the accessible banner pattern (was aria-modal="false" with
    // no aria-hidden state before).
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-modal', 'true');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.setAttribute('aria-hidden', 'false');

    banner.innerHTML = [
      '<div class="cc-inner">',
        '<p class="cc-text">',
          'We use cookies to improve your experience and serve relevant ads. ',
          '<a href="' + privacyUrl + '">Privacy Policy</a>',
        '</p>',
        '<div class="cc-buttons">',
          '<button type="button" class="cc-btn cc-decline" id="cc-decline" aria-label="Decline cookies">Decline</button>',
          '<button type="button" class="cc-btn cc-accept" id="cc-accept" aria-label="Accept cookies">Accept</button>',
        '</div>',
      '</div>'
    ].join('');

    document.body.appendChild(banner);

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        banner.style.opacity = '1';
        banner.style.transform = 'translateY(0)';
      });
    });

    document.getElementById('cc-accept').addEventListener('click', accept);
    document.getElementById('cc-decline').addEventListener('click', decline);

    // FIX: focus management — move focus into the banner once it renders,
    // so keyboard and screen-reader users land on it immediately rather
    // than having to tab through the rest of the page first.
    var firstBtn = banner.querySelector('.cc-btn');
    if (firstBtn) {
      setTimeout(function () { firstBtn.focus(); }, 100);
    }

    // FIX: keyboard trap (WCAG 2.1 SC 2.1.2) — Tab/Shift+Tab cycle between
    // the two buttons while the banner is open, instead of letting focus
    // escape into the rest of the page.
    banner.addEventListener('keydown', function (e) {
      if (e.key !== 'Tab') return;
      var focusable = this.querySelectorAll('.cc-btn');
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    });

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
