/**
 * Tech Blog Service Worker
 * Strategy:
 *   - Shell assets (CSS, JS, fonts): Cache-First with long TTL
 *   - HTML pages: Network-First (fresh content), fallback to cache
 *   - posts.json: Network-First, fallback to cache
 *   - Images: Cache-First with size limit
 *   - Offline fallback page for uncached HTML
 *
 * Bump CACHE_VERSION whenever you deploy significant CSS/JS changes.
 * Post pages are cached automatically as users visit them.
 */

'use strict';

const CACHE_VERSION = 'v1781354098';
const CACHE_SHELL = `shell-${CACHE_VERSION}`;
const CACHE_PAGES = `pages-${CACHE_VERSION}`;
const CACHE_IMAGES = `images-${CACHE_VERSION}`;

// Assets that must be cached immediately on install
const SHELL_ASSETS = [
    '/',
    '/static/style.css',
    '/static/navigation.js',
    '/static/enhanced-blog-post-styles.css',
    '/static/code_runner.js',
    '/offline.html',
    '/manifest.json',
];

// Runtime limits
const IMAGE_CACHE_MAX = 60;   // max images to keep
const PAGE_CACHE_MAX = 50;   // max HTML pages to keep

// ─── Install: pre-cache shell ──────────────────────────────────
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_SHELL)
            .then(cache => cache.addAll(SHELL_ASSETS))
            .then(() => self.skipWaiting())   // activate immediately
    );
});

// ─── Activate: delete old caches ───────────────────────────────
self.addEventListener('activate', event => {
    const currentCaches = [CACHE_SHELL, CACHE_PAGES, CACHE_IMAGES];
    event.waitUntil(
        caches.keys()
            .then(keys => Promise.all(
                keys
                    .filter(key => !currentCaches.includes(key))
                    .map(key => caches.delete(key))
            ))
            .then(() => self.clients.claim())  // take control of all tabs
    );
});

// ─── Fetch: route requests ─────────────────────────────────────
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // Only handle same-origin requests
    if (url.origin !== location.origin) return;

    // Skip non-GET requests
    if (request.method !== 'GET') return;

    // Skip browser-extension and chrome-extension URLs
    if (url.protocol !== 'https:' && url.protocol !== 'http:') return;

    const path = url.pathname;

    // ── 1. Shell assets → Cache-First ──────────────────────────
    if (isShellAsset(path)) {
        event.respondWith(cacheFirst(request, CACHE_SHELL));
        return;
    }

    // ── 2. Images → Cache-First with size limit ─────────────────
    if (isImage(path)) {
        event.respondWith(cacheFirstWithLimit(request, CACHE_IMAGES, IMAGE_CACHE_MAX));
        return;
    }

    // ── 3. posts.json → Network-First ───────────────────────────
    if (path === '/posts.json') {
        event.respondWith(networkFirst(request, CACHE_PAGES, 3000));
        return;
    }

    // ── 4. HTML pages → Network-First with offline fallback ─────
    if (request.headers.get('accept')?.includes('text/html')) {
        event.respondWith(networkFirstHtml(request));
        return;
    }

    // ── 5. Everything else → Network-First ──────────────────────
    event.respondWith(networkFirst(request, CACHE_PAGES, 4000));
});

// ─── Strategies ────────────────────────────────────────────────

/** Return cached response immediately; fetch & update cache in background */
async function cacheFirst(request, cacheName) {
    const cached = await caches.match(request);
    if (cached) return cached;
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, response.clone());
        }
        return response;
    } catch {
        return new Response('Offline', { status: 503 });
    }
}

/** Cache-First but evict oldest entries when over limit */
async function cacheFirstWithLimit(request, cacheName, maxItems) {
    const cached = await caches.match(request);
    if (cached) return cached;
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(cacheName);
            await cache.put(request, response.clone());
            // Evict if over limit
            const keys = await cache.keys();
            if (keys.length > maxItems) {
                await cache.delete(keys[0]);
            }
        }
        return response;
    } catch {
        return new Response('', { status: 408 });
    }
}

/** Try network first; fall back to cache within timeout */
async function networkFirst(request, cacheName, timeoutMs = 4000) {
    const cache = await caches.open(cacheName);
    try {
        const networkPromise = fetch(request);
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('timeout')), timeoutMs)
        );
        const response = await Promise.race([networkPromise, timeoutPromise]);
        if (response.ok) {
            cache.put(request, response.clone());
        }
        return response;
    } catch {
        const cached = await cache.match(request);
        return cached || new Response('Offline', { status: 503 });
    }
}

/** Network-First for HTML with offline.html fallback */
async function networkFirstHtml(request) {
    const cache = await caches.open(CACHE_PAGES);
    try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(request, { signal: controller.signal });
        clearTimeout(timer);
        if (response.ok) {
            cache.put(request, response.clone());
            // Evict old pages if over limit
            const keys = await cache.keys();
            if (keys.length > PAGE_CACHE_MAX) {
                await cache.delete(keys[0]);
            }
        }
        return response;
    } catch {
        const cached = await cache.match(request);
        if (cached) return cached;
        // Serve offline page as fallback
        const offline = await caches.match('/offline.html');
        return offline || new Response('<h1>You are offline</h1>', {
            status: 503,
            headers: { 'Content-Type': 'text/html' }
        });
    }
}

// ─── Helpers ───────────────────────────────────────────────────

function isShellAsset(path) {
    return (
        path === '/' ||
        path.startsWith('/static/style') ||
        path.startsWith('/static/navigation') ||
        path.startsWith('/static/enhanced-blog') ||
        path.startsWith('/static/code_runner') ||
        path.startsWith('/static/icons/') ||
        path === '/offline.html' ||
        path === '/manifest.json'
    );
}

function isImage(path) {
    return /\.(png|jpe?g|gif|webp|svg|ico)(\?.*)?$/.test(path);
}

// ─── Background Sync: queue failed contact form POSTs ──────────
// (future-proof hook — extend if you add a form backend)
self.addEventListener('sync', event => {
    if (event.tag === 'contact-form-sync') {
        event.waitUntil(syncContactForm());
    }
});

async function syncContactForm() {
    // placeholder — wire up to IndexedDB queue when you add a form backend
}

// ─── Push notifications (opt-in future feature) ─────────────────
self.addEventListener('push', event => {
    if (!event.data) return;
    const data = event.data.json();
    event.waitUntil(
        self.registration.showNotification(data.title || 'Tech Blog', {
            body: data.body || 'New post published',
            icon: '/static/icons/icon-192x192.png',
            badge: '/static/icons/icon-72x72.png',
            data: { url: data.url || '/' }
        })
    );
});

self.addEventListener('notificationclick', event => {
    event.notification.close();
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then(windowClients => {
                const url = event.notification.data?.url || '/';
                for (const client of windowClients) {
                    if (client.url === url && 'focus' in client) return client.focus();
                }
                if (clients.openWindow) return clients.openWindow(url);
            })
    );
});