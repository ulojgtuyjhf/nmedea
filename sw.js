const CACHE = 'nmedea-v2';

const BASE = 'https://ulojgtuyjhf.github.io/nmedea';

const FILES = [
    BASE + '/index.html',
    BASE + '/dashboard.html',
    BASE + '/pay.html',
    BASE + '/sw.js',
    'https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,700;9..40,800&family=DM+Mono:wght@400;500&display=swap'
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

// Fetch — serve from cache first, fallback to network
self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(cached => {
            if (cached) return cached;
            return fetch(e.request).then(response => {
                // Cache new requests dynamically
                if (e.request.url.startsWith('http')) {
                    caches.open(CACHE).then(cache => {
                        cache.put(e.request, response.clone());
                    });
                }
                return response;
            }).catch(() => {
                // If offline and not cached — return index.html
                return caches.match('/index.html');
            });
        })
    );
});
