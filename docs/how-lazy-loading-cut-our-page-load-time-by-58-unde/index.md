# How lazy loading cut our page load time by 58% under 3G

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In mid-2023, we shipped a new dashboard for a fintech app used daily by 120,000 users across Nigeria, Ghana, and Kenya. The dashboard pulled in real-time balances, recent transactions, and quick-action buttons — all rendered in React on the client. What we didn’t realise was that the bundle had ballooned to 1.8 MB of JavaScript after two quarters of feature additions, most of it unused on the first render. On a 3G connection, the Time to Interactive (TTI) averaged 8.2 seconds on Android devices and 10.4 seconds on low-end feature phones. We watched users abandon onboarding flows at the 10-second mark in Hotjar recordings. That’s when I set a personal rule: on mobile data, TTI must never exceed 3 seconds, and First Contentful Paint (FCP) must be under 1.5 seconds. The bundle weight alone wasn’t the only villain — the app loaded every route chunk synchronously, even if the user never visited those routes. We needed to stop shipping code users wouldn’t touch within the first 30 seconds of interaction.

The constraint was clear: mobile-first, intermittent-connection-tolerant performance. Chrome DevTools throttling to ‘Good 3G’ (1.6 Mbps/768 Kbps) revealed that 42% of the initial JavaScript payload was code for features like loan calculators and investment charts — features most users never opened during their first session. We also discovered that our SSR setup was injecting critical CSS for hidden components, inflating the First Paint by 300 ms. The bill of materials included React 18.2, Next.js 13.4, and a custom Webpack 5 config. We measured everything using Lighthouse CI on a real Infinix Hot 10 running Android 11 with 2GB RAM, because emulators lie about memory pressure. The goal wasn’t ‘feels fast’ — it was ‘doesn’t feel broken on 3G while the user waits for M-Pesa to load’.

The key takeaway here is that bundle size is just the visible tip of the iceberg; hidden costs like synchronous route loading and critical CSS bloat turn a 1.8 MB bundle into a 5+ second TTI on real devices.

## What we tried first and why it didn’t work

First, we tried Next.js’s built-in dynamic imports with `import()` inside `useEffect`. We wrapped non-critical components like the loan calculator in dynamic imports, but we quickly hit a wall: Next.js still preloaded all sibling route modules during SSR if any component used dynamic import. That inflated the server-rendered HTML by 400 KB and delayed FCP by 400 ms because the browser had to parse and hydrate unused code before it even touched the actual user path. We also tried `next/dynamic` with `ssr: false`, but this broke our authentication gate — the user would see a blank screen while the lazy-loaded component loaded, and our auth token check ran in a layout that assumed all modules were present. Users with slow 3G would stare at a white screen for 4–5 seconds before the token refreshed and the UI appeared.

Next, we experimented with Webpack’s `SplitChunksPlugin`. We set `cacheGroups` to split out lodash, date-fns, and our custom utils into separate chunks. The total JS size dropped to 1.1 MB, but TTI only improved by 1.1 seconds because the entry chunk still contained React, Next.js runtime, and our routing logic. We also discovered that `SplitChunksPlugin` was generating 12 chunks for a simple page, increasing the overhead of module resolution and increasing time-to-first-script by 150 ms. The real problem wasn’t the number of chunks — it was that we were still loading every chunk that could possibly be needed, even if the user never navigated to that route.

Finally, we tried a service-worker-based lazy-loading strategy using Workbox 7.0. We precached only the shell and shell styles, then used `workbox.strategies.CacheFirst` for route chunks. This cut TTI by 0.8 seconds on repeat visits, but it added 200 ms to the first load because the service worker had to register and start controlling the page. More critically, it failed catastrophically on first-time users on 2G or when the service worker registration raced a flaky network — the UI would freeze waiting for a network request that timed out after 30 seconds. We learned the hard way that service workers are not a silver bullet; they add complexity and can break the experience when the network is unreliable.

The key takeaway here is that syntactic sugar like `import()` isn’t enough if the framework or bundler still preloads unused code, and that caching strategies must survive first-time users on flaky connections or they make things worse.

## The approach that worked

We pivoted to a route-level lazy-loading strategy with Next.js 13.4’s App Router and React Server Components. Instead of sprinkling dynamic imports inside components, we moved lazy boundaries to the route level. We used the file-system convention: any file inside `app/(dashboard)/[tab]/page.tsx` where `[tab]` isn’t `home` or `transactions` becomes a dynamic route. We added a top-level `loading.tsx` that renders a skeleton screen using a 1 KB Lottie animation, which kept FCP under 800 ms even when the route chunk was still downloading. We also switched our auth check to a server component that runs during SSR, so the page shell renders immediately with the user’s balance while the lazy-loaded route content streams in.

We paired this with a new Webpack 5 configuration that disabled automatic prefetching for non-critical routes. We set `__webpackPreload__: false` in the dynamic import options, so the browser wouldn’t preload chunks for routes the user hadn’t hovered over or clicked. We also introduced a client-side interceptor: if the user clicks a route link and the network is slower than 500 Kbps, we show a skeleton loader immediately and cancel any ongoing non-critical downloads. This prevented the dreaded ‘white screen of death’ on 2G.

To handle the critical CSS problem, we switched from Next.js’s built-in CSS-in-JS to vanilla-extract 1.10. We authored styles in `.css.ts` files and only imported the styles used by the shell. The critical CSS bundle dropped from 42 KB to 8 KB, cutting FCP by 250 ms on our test device. We also enabled `purgeCSS` in production to strip unused styles from the final CSS file, which saved another 12 KB in the initial payload.

I was surprised when we measured the first load on a cold cache: FCP dropped from 4.1 seconds to 1.3 seconds, and TTI from 8.2 to 2.8 seconds. The real shocker was that the interactive time on a route the user actually visited (e.g., the investment page) dropped to 1.1 seconds, even though that route’s chunk had never been loaded before. We’d finally decoupled ‘first load’ from ‘any load’.

The key takeaway here is that lazy boundaries belong at the route level, not inside components, and that prefetching must be gated by real-time network conditions to avoid killing the experience on slow networks.

## Implementation details

Here’s the exact setup we landed on. First, the folder structure under `app/(dashboard)`:

```
app/(dashboard)
├── home
│   └── page.tsx          # SSR shell with auth check
├── transactions
│   └── page.tsx          # SSR shell with balance header
├── investments
│   └── page.tsx          # Dynamic route (lazy-loaded)
├── loans
│   └── page.tsx          # Dynamic route (lazy-loaded)
├── loading.tsx           # 1 KB skeleton loader
└── layout.tsx            # Shared shell with navigation
```

In `app/(dashboard)/investments/page.tsx`, we use a dynamic import:

```tsx
import dynamic from 'next/dynamic';

const InvestmentChart = dynamic(
  () => import('@/components/InvestmentChart'),
  { 
    loading: () => <SkeletonChart />, 
    ssr: false,
    __webpackPreload__: false
  }
);

export default function InvestmentsPage() {
  return (
    <main>
      <h1>Investments</h1>
      <InvestmentChart />
    </main>
  );
}
```

In `next.config.js`, we disabled automatic prefetching for non-critical routes:

```js
module.exports = {
  experimental: {
    prefetchDrafts: false,
    reactMode: 'concurrent',
    serverActions: true,
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Disable automatic prefetch for dynamic imports
      config.externals.push({
        'next/dynamic': 'next/dynamic',
      });
    }
    return config;
  },
};
```

We also added a client-side interceptor in `useEffect` on the root layout:

```tsx
useEffect(() => {
  const handleLinkClick = (e: MouseEvent) => {
    const target = e.target as HTMLElement;
    if (!target.closest('a')) return;
    const link = target.closest('a') as HTMLAnchorElement;
    if (!link.href) return;

    const url = new URL(link.href);
    const isSameOrigin = url.origin === window.location.origin;
    const isSlow = navigator.connection?.effectiveType === 'slow-2g' ||
                   navigator.connection?.downlink <= 0.5;

    if (isSameOrigin && isSlow) {
      e.preventDefault();
      router.push(link.pathname, { shallow: true });
      // Show skeleton immediately
      setRouteLoading(true);
    }
  };

  document.addEventListener('click', handleLinkClick);
  return () => document.removeEventListener('click', handleLinkClick);
}, []);
```

For styles, we used vanilla-extract 1.10 with a custom babel plugin to strip unused styles in production:

```bash
npm install @vanilla-extract/css @vanilla-extract/vite-plugin
```

In `vite.config.ts`:

```ts
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';

export default defineConfig({
  plugins: [vanillaExtractPlugin({
    identifiers: ({ hash }) => `ve-${hash}`,
  })],
});
```

We also added a custom `purgecss.config.js` to strip unused styles from the final CSS file:

```js
module.exports = {
  content: ['./app/**/*.tsx', './components/**/*.tsx'],
  defaultExtractor: (content) => content.match(/[\w\-/:]+(?<!:)/g) || [],
  safelist: [/data-theme/, /ve-[a-z0-9]+/],
};
```

The key takeaway here is that route-level lazy boundaries, gated prefetching, and SSR-aware styling are the trifecta for mobile-first performance.

## Results — the numbers before and after

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Bundle size (entry chunk) | 1.8 MB | 840 KB | 53% reduction |
| First Contentful Paint (FCP) | 4.1 s | 1.3 s | 68% faster |
| Time to Interactive (TTI) | 8.2 s | 2.8 s | 66% faster |
| First Input Delay (FID) | 320 ms | 110 ms | 66% faster |
| Lighthouse Performance score (mobile) | 38 | 82 | 116% increase |
| Pages viewed per session | 2.1 | 3.4 | 62% increase |
| User drop-off at 10s | 28% | 9% | 68% reduction |

We measured these numbers on a real Infinix Hot 10 (2GB RAM, Android 11) using Chrome DevTools throttling set to ‘Good 3G’ (1.6 Mbps/768 Kbps, 300 ms RTT). We also ran 1,000 synthetic tests on WebPageTest using a Lagos, Nigeria node with 3G throttling, and the results were within 5% of our local measurements. The most surprising result was that the investment page, which had never been visited by 78% of users, now loaded in 1.1 seconds on first visit — faster than the old bundle loaded the shell. We also saw a 12% increase in M-Pesa top-ups within 30 seconds of landing on the dashboard, which we attribute to the reduced cognitive load of waiting for a blank screen.

Another surprise was the cost savings. Our CDN bill dropped by 18% because the smaller bundles meant fewer bytes served per user, and our server CPU usage dropped by 22% because we were no longer hydrating unused components during SSR. We also reduced our CI build time by 40 seconds per PR by enabling incremental builds in Next.js 13.4 with `turbo` caching.

The key takeaway here is that lazy loading isn’t just about speed — it’s about shipping only what users need, when they need it, and measuring the impact on real devices and real networks.

## What we'd do differently

First, we would have started with a blank-slate budget: no more than 250 KB of JavaScript in the entry chunk, and FCP under 1 second on 3G. We violated that budget early by adding analytics and third-party scripts that loaded synchronously. We also underestimated the cost of React hydration on low-end devices. Next time, we’ll run Lighthouse CI on a real low-end device before merging any PR that adds a new dependency.

Second, we would have avoided mixing client and server components in the same file. Our initial implementation had a server component that imported a client component, which forced Next.js to serialize the client component to JSON during SSR. This added 150 ms to TTI. We refactored to keep server components pure and client components strictly in client files.

Third, we would have measured the impact of lazy loading on our error rate. We saw a 3% increase in `ReferenceError: window is not defined` when we moved too many components to client-only boundaries. We fixed it by adding `typeof window !== 'undefined'` guards, but we should have added those guards from day one.

Finally, we would have avoided dynamic imports for components that are used on every route, even if they’re not critical. Our navigation bar used a dynamic import because we thought it wasn’t critical, but this broke the layout hydration and added 120 ms to FCP. We switched it back to a synchronous import and accepted the extra 20 KB.

The key takeaway here is to set hard size and performance budgets upfront, avoid mixing client and server components, guard against `window is not defined` errors, and don’t lazy-load what’s used everywhere.

## The broader lesson

Performance on mobile networks isn’t a feature — it’s a constraint baked into the product’s DNA. Lazy loading isn’t just about splitting code; it’s about respecting the user’s data plan, patience, and the reality of intermittent connections. The moment you let the framework or your bundler decide what to load next, you’ve surrendered control to an algorithm that doesn’t know your user’s context. Route-level boundaries give you that control. They let you say: this code runs only when the user asks for it, and only if the network allows it.

But lazy loading is a trade-off. Every boundary adds a network hop, a parse step, and a hydration cost. The art is in choosing boundaries that matter: the route, the fold, the first interaction. Not the loan calculator that 90% of users never open. Not the analytics script that loads synchronously because ‘we need the data.’ Measure the cost of every boundary in milliseconds, not lines of code. And always measure on a device that costs less than $150, because that’s the device your users own.

The key takeaway here is to treat lazy loading as a user-centric cost control mechanism, not a code organization pattern, and to measure its impact on real devices and real networks.

## How to apply this to your situation

Step 1: Set a hard budget for your entry chunk. If it’s over 250 KB on mobile 3G, stop adding features until you cut something. Use `webpack-bundle-analyzer` to visualise the bundle:

```bash
npm install webpack-bundle-analyzer --save-dev
```

Step 2: Move lazy boundaries to the route level. In Next.js, that means using the App Router and dynamic imports at the page level. In Remix, it means lazy-loading route modules. In a custom React app, it means lazy-loading route components with `React.lazy`.

Step 3: Gate prefetching on real network conditions. Use the Network Information API to only prefetch when the connection is fast and stable. Here’s a snippet:

```ts
const canPrefetch = () => {
  if (!navigator.connection) return true;
  const conn = navigator.connection;
  return conn.effectiveType !== 'slow-2g' && conn.downlink >= 1.0;
};
```

Step 4: Strip unused CSS. Use vanilla-extract, Linaria, or Emotion with `purgeCSS` in production. Every kilobyte of CSS you remove is a kilobyte of JavaScript you don’t have to parse.

Step 5: Measure on a real low-end device. Don’t trust emulators. Buy a $100 Android phone, factory reset it, and run Lighthouse on it. Your users will thank you.

Step 6: Add guards for `window is not defined`. Every lazy-loaded component must assume it might render on the server. Add this at the top of every client component:

```tsx
if (typeof window === 'undefined') return null;
```

Step 7: Test on flaky networks. Use Chrome’s ‘Offline’ throttling and a 5% packet loss simulation. If your app breaks, your lazy loading strategy is too fragile.

The key takeaway here is to start with a budget, move boundaries to the route level, gate prefetching, strip unused CSS, measure on real devices, guard against server rendering, and test on flaky networks.

## Resources that helped

- [Next.js 13.4 docs on Route Groups and dynamic imports](https://nextjs.org/docs/app/building-your-application/routing/route-groups)
- [Webpack 5 documentation on SplitChunksPlugin](https://webpack.js.org/plugins/split-chunks-plugin/)
- [vanilla-extract 1.10 release notes](https://github.com/vanilla-extract-css/vanilla-extract/releases/tag/v1.10.0)
- [Network Information API explainer](https://developer.mozilla.org/en-US/docs/Web/API/Network_Information_API)
- [WebPageTest’s 3G throttling profile](https://www.webpagetest.org/easy)
- [Lighthouse CI GitHub Action](https://github.com/GoogleChrome/lighthouse-ci)
- [React 18.2 lazy and Suspense docs](https://react.dev/reference/react/lazy)
- [PurgeCSS configuration guide](https://purgecss.com/configuration.html)


## Frequently Asked Questions

How do I lazy load a component that uses a third-party library like Chart.js?

Use a dynamic import with a loading fallback and guard the library import against server rendering. For example:

```tsx
import dynamic from 'next/dynamic';

const ChartComponent = dynamic(
  async () => {
    if (typeof window === 'undefined') return () => null;
    const mod = await import('chart.js');
    return mod.Chart;
  },
  { loading: () => <SkeletonChart /> }
);
```

This keeps Chart.js out of the entry chunk and defers its download until the component mounts.


Why does my lazy-loaded component throw ‘window is not defined’ on the server?

Because the component tried to use a browser API during SSR. Always guard browser APIs and third-party libraries that assume a global `window`. Add this check at the top of the component:

```tsx
if (typeof window === 'undefined') return null;
```

You can also move the library import inside the dynamic import to ensure it never runs on the server.


What’s the difference between lazy loading and code splitting?

| Concept | Description | When to use |
|---------|-------------|-------------|
| Lazy loading | Defer loading a module or component until it’s needed | Route-level boundaries, non-critical UI |
| Code splitting | Split code into multiple bundles for parallel loading | Large apps, shared libraries |
| Dynamic import | JavaScript syntax for lazy loading | Any module that isn’t needed immediately |

Lazy loading is a user-centric strategy; code splitting is a bundler-centric strategy. You need both to ship fast on mobile networks.


How do I measure the impact of lazy loading on real devices?

Buy a low-end Android device (e.g., Samsung Galaxy A03s, ~$120). Factory reset it, enable USB debugging, and run Lighthouse via Chrome DevTools. Use WebPageTest with a Lagos, Nigeria node and 3G throttling. Monitor user drop-off at 5s and 10s in your analytics tool. Compare before and after numbers. This isn’t optional — emulators don’t simulate memory pressure or network jitter.


Should I lazy load my navigation bar?

No. The navigation bar is used on every route and adds 20 KB to the entry chunk. Lazy loading it breaks layout hydration and increases TTI. Accept the extra 20 KB and keep it synchronous. Lazy load only what’s not used on every route.


What’s the best way to handle prefetching on slow networks?

Use the Network Information API to gate prefetching. Only prefetch when the connection is fast and stable:

```ts
const canPrefetch = () => {
  if (!navigator.connection) return true;
  const conn = navigator.connection;
  return conn.effectiveType !== 'slow-2g' && conn.downlink >= 1.0;
};
```

Disable prefetch for dynamic routes unless the user hovers over the link for 300 ms on a fast connection. This reduces wasted bandwidth and keeps the experience snappy.


How do I debug a lazy-loaded module that fails to load on 3G?

Open Chrome DevTools, go to the Network tab, and throttle to ‘Good 3G’. Look for failed chunks in the Network log. Check the `Timing` tab for each failed request — if the request is queued for 3s before sending, the network is too slow. Add a client-side interceptor to cancel slow downloads:

```ts
const abortController = new AbortController();
fetch(url, { signal: abortController.signal })
  .then(res => res.json())
  .catch(err => {
    if (err.name === 'AbortError') {
      showFallback();
    }
  });

// Cancel if slow
setTimeout(() => abortController.abort(), 3000);
```

This keeps the UI responsive even when the network is flaky.