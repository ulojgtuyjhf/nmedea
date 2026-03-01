const CACHE = 'nmedea-v3';

const BASE = 'https://ulojgtuyjhf.github.io/nmedea';

const FILES = [
    BASE + '/',
    BASE + '/index.html',
    BASE + '/sw.js'
];

// Install — cache everything
self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE).then(cache => {
            return cache.addAll(FILES);
        })
    );
    self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', e => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

// Fetch — cache first, fallback to network
self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(cached => {
            if (cached) return cached;
            return fetch(e.request).then(response => {
                if (e.request.url.startsWith('http')) {
                    caches.open(CACHE).then(cache => {
                        cache.put(e.request, response.clone());
                    });
                }
                return response;
            }).catch(() => {
                return caches.match(BASE + '/index.html');
            });
        })
    );
});