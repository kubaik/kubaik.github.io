// pwa.js — service worker registration + install prompt + GA4 analytics

(function () {
    'use strict';

    // ── 0. GA4 helper ──────────────────────────────────────────────────
    function trackEvent(name, params) {
        if (typeof gtag === 'function') {
            gtag('event', name, params || {});
        }
    }

    // ── 1. Register service worker ─────────────────────────────────────
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function () {
            navigator.serviceWorker
                .register('/sw.js', { scope: '/' })
                .then(function (reg) {
                    console.log('[PWA] Service worker registered, scope:', reg.scope);

                    reg.addEventListener('updatefound', function () {
                        var newWorker = reg.installing;
                        if (!newWorker) return;
                        newWorker.addEventListener('statechange', function () {
                            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                showUpdateBanner();
                            }
                        });
                    });
                })
                .catch(function (err) {
                    console.warn('[PWA] Service worker registration failed:', err);
                });

            navigator.serviceWorker.addEventListener('controllerchange', function () {
                if (window.__pwaReloading) return;
                window.__pwaReloading = true;
                window.location.reload();
            });
        });
    }

    // ── 2. Install prompt ──────────────────────────────────────────────
    var deferredPrompt = null;
    var installBtn = null;

    window.addEventListener('beforeinstallprompt', function (e) {
        e.preventDefault();
        deferredPrompt = e;
        showInstallButton();
        // How many people saw the install button?
        trackEvent('pwa_prompt_shown');
    });

    window.addEventListener('appinstalled', function () {
        deferredPrompt = null;
        hideInstallButton();
        console.log('[PWA] App installed');
        // Confirmed install (Android / desktop Chrome / Edge only — iOS never fires this)
        trackEvent('pwa_installed');
    });

    function showInstallButton() {
        if (installBtn) return;

        installBtn = document.createElement('button');
        installBtn.id = 'pwa-install-btn';
        installBtn.innerHTML =
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Install App';
        installBtn.setAttribute('aria-label', 'Install Tech Blog as an app');

        Object.assign(installBtn.style, {
            position: 'fixed',
            bottom: '5rem',
            right: '1.25rem',
            zIndex: '999',
            background: 'linear-gradient(135deg,#667eea,#764ba2)',
            color: '#fff',
            border: 'none',
            borderRadius: '50px',
            padding: '0.6rem 1.25rem',
            fontSize: '0.875rem',
            fontWeight: '600',
            cursor: 'pointer',
            boxShadow: '0 4px 14px rgba(102,126,234,0.45)',
            display: 'flex',
            alignItems: 'center',
            opacity: '0',
            transform: 'translateY(8px)',
            transition: 'opacity 0.3s ease, transform 0.3s ease',
            fontFamily: 'inherit',
        });

        document.body.appendChild(installBtn);

        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                installBtn.style.opacity = '1';
                installBtn.style.transform = 'translateY(0)';
            });
        });

        installBtn.addEventListener('click', function () {
            if (!deferredPrompt) return;

            // User tapped your button — about to see the OS dialog
            trackEvent('pwa_install_clicked');

            deferredPrompt.prompt();
            deferredPrompt.userChoice.then(function (result) {
                // result.outcome is 'accepted' or 'dismissed'
                trackEvent('pwa_install_' + result.outcome);   // pwa_install_accepted / pwa_install_dismissed
                console.log('[PWA] Install choice:', result.outcome);
                deferredPrompt = null;
                hideInstallButton();
            });
        });
    }

    function hideInstallButton() {
        if (!installBtn) return;
        installBtn.style.opacity = '0';
        installBtn.style.transform = 'translateY(8px)';
        setTimeout(function () {
            if (installBtn && installBtn.parentNode) {
                installBtn.parentNode.removeChild(installBtn);
            }
            installBtn = null;
        }, 300);
    }

    // ── 3. Update banner ───────────────────────────────────────────────
    function showUpdateBanner() {
        if (document.getElementById('pwa-update-banner')) return;

        var banner = document.createElement('div');
        banner.id = 'pwa-update-banner';

        Object.assign(banner.style, {
            position: 'fixed',
            bottom: '0',
            left: '0',
            right: '0',
            zIndex: '9999',
            background: '#1a1d2e',
            color: '#e2e8f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.9rem 1.5rem',
            fontSize: '0.9rem',
            boxShadow: '0 -4px 16px rgba(0,0,0,0.2)',
            gap: '1rem',
            flexWrap: 'wrap',
        });

        var msg = document.createElement('span');
        msg.textContent = '🚀 A new version of Tech Blog is available.';

        var actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:0.6rem;flex-shrink:0';

        var refreshBtn = document.createElement('button');
        refreshBtn.textContent = 'Refresh now';
        Object.assign(refreshBtn.style, {
            background: 'linear-gradient(135deg,#667eea,#764ba2)',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            padding: '0.45rem 1rem',
            fontWeight: '600',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontFamily: 'inherit',
        });
        refreshBtn.addEventListener('click', function () {
            window.location.reload();
        });

        var dismissBtn = document.createElement('button');
        dismissBtn.textContent = 'Later';
        Object.assign(dismissBtn.style, {
            background: 'transparent',
            color: '#94a3b8',
            border: '1px solid #4a5568',
            borderRadius: '6px',
            padding: '0.45rem 1rem',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontFamily: 'inherit',
        });
        dismissBtn.addEventListener('click', function () {
            banner.remove();
        });

        actions.appendChild(refreshBtn);
        actions.appendChild(dismissBtn);
        banner.appendChild(msg);
        banner.appendChild(actions);
        document.body.appendChild(banner);
    }

    // ── 4. Online / Offline indicator ──────────────────────────────────
    var offlineToast = null;

    function showOfflineToast() {
        if (offlineToast) return;
        offlineToast = document.createElement('div');
        offlineToast.textContent = '⚡ You are offline — cached content available';
        Object.assign(offlineToast.style, {
            position: 'fixed',
            top: '1rem',
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#1a1d2e',
            color: '#e2e8f0',
            padding: '0.6rem 1.25rem',
            borderRadius: '50px',
            fontSize: '0.875rem',
            fontWeight: '600',
            zIndex: '9999',
            boxShadow: '0 4px 14px rgba(0,0,0,0.3)',
            whiteSpace: 'nowrap',
        });
        document.body.appendChild(offlineToast);
    }

    function hideOfflineToast() {
        if (!offlineToast) return;
        offlineToast.remove();
        offlineToast = null;
    }

    if (!navigator.onLine) showOfflineToast();
    window.addEventListener('offline', showOfflineToast);
    window.addEventListener('online', hideOfflineToast);

})();