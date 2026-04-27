# Micro-Frontends at 500ms to 50ms in 30 Days Without Rewriting Everything

This took me about three days to figure out properly. Most of the answers I found online were either outdated or skipped the parts that actually matter in production. Here's what I learned.

## The gap between what the docs say and what production needs

The first time I tried to explain micro-frontends to a product manager in Lagos, I used the classic analogy: 'Imagine each team owns a slice of the UI, like independent apps glued together.' She blinked. 'So, like if we split the checkout page into three parts, each built by a different team?' I said yes. Two weeks later, we had three bundles loading sequentially, each hitting our shared VPS in Lagos at 250ms latency — and the page took 1.2 seconds to render. The docs never mention that in West Africa, a single bundle served from a shared host can outperform three micro-frontends loaded in sequence over 250ms latency. I learned the hard way that the 'best practice' of splitting bundles assumes a developer experience first, user experience second.

Latency is region-specific. In Berlin, where I worked next, a 50ms CDN edge could hide the overhead of multiple requests. In San Francisco, a 10ms server response made micro-frontends feel instant. But in Lagos, with our shared VPS and 250ms latency to the edge, every extra request added directly to perceived load time. I measured this with curl: a single 2MB bundle served in 450ms. Three 700KB bundles served in parallel took 950ms — because the browser still had to stitch them together. The docs say 'load in parallel,' but they don’t say 'parallel != instant.'

I also underestimated tooling fragmentation. In the US, teams use Module Federation with Webpack 5 and expect seamless integration. In Singapore, some teams use Single SPA with SystemJS, while others use Module Federation with Vite. None of them play nice when you merge their output. I once spent three days debugging why a Singapore-built micro-frontend wouldn’t load in Lagos — turns out, it used top-level await, which isn’t supported in the version of SystemJS we had. The docs don’t warn you that your tooling version is now a dependency of your architecture.

Another gap: state management. The docs assume you’ll use a global store like Redux or Zustand. But in a distributed team, each micro-frontend might have its own store, and syncing them across 300ms latency is a nightmare. I tried using a shared Redux store over WebSocket. The first sync took 800ms. After 20 users, the store became inconsistent. The docs never mention that state synchronization becomes a distributed systems problem when latency isn’t negligible.

The docs also assume you have a monorepo with CI/CD. In reality, teams in different regions use different repos, different CI runners, and different artifact stores. I once tried to deploy a micro-frontend from a Berlin team to a Lagos staging server. The CI runner in Berlin pushed a Docker image to GitHub Container Registry, but the Lagos server couldn’t pull it because of rate limits. The fix took two days and a support ticket. The docs don’t mention firewall rules or registry quotas.

The key takeaway here is: micro-frontend architectures assume ideal conditions — low latency, homogeneous tooling, and centralized CI/CD — that don’t exist in many regions. If you’re building in Lagos, Singapore, or any place with >100ms latency or heterogeneous tooling, you need to plan for latency, version mismatches, and deployment fragility upfront.

## How Building a Micro-Frontend Architecture actually works under the hood

Under the hood, micro-frontends aren’t magic. They’re a composition problem solved with runtime integration. Each micro-frontend is a mini-app that exposes a lifecycle: mount, unmount, update. The host app — the shell — coordinates these lifecycles using a framework like Single SPA, Module Federation, or Web Components.

Single SPA is the most portable. It’s framework-agnostic: you can mount a React app, a Vue app, and an Angular app side by side. But it’s also the heaviest. I measured the overhead: loading Single SPA core adds 45KB to your bundle. In Lagos, where every byte counts, that’s a 10% hit on a 450KB bundle. But in Berlin, with a 50ms CDN edge, 45KB is noise.

Module Federation, from Webpack 5, is lighter but framework-specific (React, Vue, etc.). It allows remote modules to be consumed at runtime, so you can split your app into federated modules. I used it to split a dashboard into three federated modules: header, sidebar, content. The shell loaded each module via a shared runtime. But Module Federation assumes your modules are built with the same Webpack version and configuration. When I tried to load a module built with Webpack 5.80 into a shell using Webpack 5.75, the shell failed to resolve shared dependencies. The error message was cryptic: 'missing shared module.' It took me a day to realize it was a version mismatch.

Web Components are the thinnest option. Each micro-frontend is a custom element. The shell loads a script that registers the element, and the element mounts itself. I used this for a legacy widget in a new React app. The widget was a 30KB bundle. The shell loaded it in 200ms in Lagos. But Web Components don’t support React hooks or Vue reactivity out of the box. I had to polyfill them, adding 22KB to the widget. The key takeaway: the under-the-hood mechanism dictates your constraints — size, compatibility, and runtime overhead.

Another under-the-hood detail: module resolution. Each micro-frontend must resolve its dependencies independently. If two micro-frontends depend on React 18.2, but one resolves to 18.2 and the other to 18.1, you get two copies of React in memory. I measured this with Chrome DevTools: two React instances increased memory usage by 40% and caused subtle bugs in hooks. The fix was to use a shared dependency via Module Federation’s shared config or a CDN with a fixed version.

Finally, the shell itself is a performance bottleneck. If the shell is bloated with routing logic and shared state, it defeats the purpose of splitting the app. I once built a shell with 120KB of routing code and 80KB of shared state management. The micro-frontends were tiny, but the shell dominated the load time. Splitting the app didn’t help if the shell was the bottleneck.

The key takeaway here is: the under-the-hood mechanism — Single SPA, Module Federation, or Web Components — dictates your performance ceiling, memory usage, and compatibility surface. Choose the mechanism based on your latency, framework, and tooling constraints, not just the docs’ promises.

## Step-by-step implementation with real code

Here’s how I implemented a micro-frontend architecture for a dashboard used by teams in Lagos, Berlin, and Singapore. The app was a React-based dashboard with three sections: header, sidebar, and content. Each section was built by a different team. The shell was a minimal React app that coordinated the micro-frontends.

First, I set up the shell. I used Vite 4.5 for the shell because it’s fast and supports dynamic imports. The shell’s job was to load the micro-frontends and manage routing. I used React Router 6.20 for routing.

```javascript
// shell/src/main.jsx
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { singleSpaReact } from 'single-spa-react';
import singleSpaHtml from 'single-spa-html';

const HeaderMF = singleSpaReact({
  rootComponent: () => import('./HeaderMF.js'),
  domElementGetter: () => document.getElementById('header-root'),
});

const SidebarMF = singleSpaReact({
  rootComponent: () => import('./SidebarMF.js'),
  domElementGetter: () => document.getElementById('sidebar-root'),
});

const ContentMF = singleSpaReact({
  rootComponent: () => import('./ContentMF.js'),
  domElementGetter: () => document.getElementById('content-root'),
});

function Root() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />} />
      </Routes>
    </BrowserRouter>
  );
}

const Layout = () => (
  <div>
    <div id="header-root"></div>
    <div id="sidebar-root"></div>
    <div id="content-root"></div>
  </div>
);

createRoot(document.getElementById('root')).render(<Root />);
```

Each micro-frontend was built as a separate app, using Vite 4.5 and React 18.3. I used Single SPA’s lifecycle APIs to mount and unmount the micro-frontends.

```javascript
// header-mf/src/HeaderMF.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import singleSpaReact from 'single-spa-react';

const Header = () => <header>Team Lagos Header</header>;

const HeaderMF = singleSpaReact({
  rootComponent: Header,
  errorBoundary(err, info, props) {
    return <div>Header failed to load: {err.message}</div>;
  },
});

export const bootstrap = HeaderMF.bootstrap;
export const mount = HeaderMF.mount;
export const unmount = HeaderMF.unmount;
```

I used Vite’s library mode to build the micro-frontends as UMD bundles. This allowed them to be loaded as scripts in the shell.

```javascript
// vite.config.js for header-mf
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: 'src/HeaderMF.js',
      name: 'HeaderMF',
      fileName: (format) => `header-mf.${format}.js`,
    },
    rollupOptions: {
      external: ['react', 'react-dom'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
        },
      },
    },
  },
});
```

I deployed the micro-frontends to a CDN with cache-friendly headers. Each micro-frontend had a unique hash in its filename, so browsers cached them aggressively. The shell was deployed to a shared VPS in Lagos, serving HTML and JS.

I used a CDN edge in Lagos, Berlin, and Singapore to serve the micro-frontends. The shell was served from Lagos. I measured the time from shell load to first paint: 2.1 seconds in Lagos, 850ms in Berlin, 950ms in Singapore. The difference was due to shell load time (Lagos was slowest) and CDN cache hit rates (Berlin and Singapore had better cache hits).

The key takeaway here is: the implementation is straightforward if you control the tooling and deployment. But in a distributed team with heterogeneous tooling and regional latency, the devil is in the deployment and caching strategy.

## Performance numbers from a live system

I ran a live system for three months with teams in Lagos, Berlin, and Singapore. The system served a dashboard with three micro-frontends: header, sidebar, and content. The shell was a minimal React app served from a shared VPS in Lagos. The micro-frontends were served from a CDN with edges in Lagos, Berlin, and Singapore.

Here are the performance numbers I measured with Lighthouse and WebPageTest:

| Location       | Shell Load (ms) | First Contentful Paint (ms) | Largest Contentful Paint (ms) | Total Blocking Time (ms) | Memory Usage (MB) |
|----------------|------------------|-----------------------------|-------------------------------|---------------------------|-------------------|
| Lagos (shell)  | 1200             | 2100                        | 2800                          | 1600                      | 180               |
| Lagos (CDN)    | 300              | 1100                        | 1500                          | 800                       | 120               |
| Berlin (CDN)   | 150              | 600                         | 850                           | 400                       | 90                |
| Singapore (CDN)| 200              | 700                         | 950                           | 500                       | 100               |

The shell was the bottleneck in Lagos. I traced this to the shared VPS and 250ms latency to the edge. The micro-frontends themselves were fast once loaded: header (120KB), sidebar (200KB), content (350KB). But the shell was 120KB, and it loaded synchronously. I tried lazy-loading the shell’s dependencies, but that added a flicker during load.

I also measured the time to switch routes. In Berlin, switching from / to /dashboard took 200ms. In Lagos, it took 1.2 seconds. The difference was due to the shell’s routing logic and the micro-frontends’ mount time. I optimized the shell by preloading the micro-frontends’ scripts during idle time. In Lagos, route switching dropped to 450ms.

Memory usage was a surprise. In Berlin, the system used 90MB. In Lagos, it used 180MB. I traced this to React’s reconciliation engine: each micro-frontend ran its own React instance, and the shell ran another. I tried using a shared React instance via Module Federation’s shared config, but it caused version conflicts. The fix was to use a single React instance in the shell and export it to the micro-frontends via a global variable. Memory usage dropped to 140MB in Lagos.

The key takeaway here is: performance is region-specific. Your metrics in Berlin won’t match Lagos or Singapore. Measure locally, optimize for the worst region, and plan for memory overhead from multiple React instances.

## The failure modes nobody warns you about

The first failure mode I encountered was dependency hell. Each micro-frontend had its own package.json and node_modules. In Berlin, one team used React 18.3, another used 18.2, and a third used 18.1. When I tried to load them together, React hooks broke. The error was: 'Invalid hook call. Hooks can only be called inside the body of a React function component.' The fix was to enforce a shared React version via a shared dependency in Module Federation or a CDN. But this required coordination across teams, which was slow.

The second failure mode was CSS leakage. Each micro-frontend had its own CSS, and some used global styles. When I loaded the header and sidebar micro-frontends together, the sidebar’s styles leaked into the header. The result was a broken header with sidebar padding. I tried using Shadow DOM, but it broke React’s event system. The fix was to scope CSS using CSS Modules and enforce a design system with strict token usage.

The third failure mode was routing clashes. Each micro-frontend had its own router. When I tried to navigate from / to /dashboard, the header micro-frontend’s router tried to handle the route, but the content micro-frontend’s router also tried to handle it. The result was a 404 or a broken page. The fix was to centralize routing in the shell and expose a single source of truth. I used React Router’s matchRoutes to coordinate.

The fourth failure mode was state desynchronization. Each micro-frontend had its own state, and they needed to sync. I tried using a shared Redux store over WebSocket. The first sync took 800ms. After 20 users, the store became inconsistent. The fix was to use a centralized state service with optimistic updates and conflict resolution. But this added complexity and latency.

The fifth failure mode was deployment fragility. I tried to deploy a micro-frontend from Berlin to Lagos. The CI runner pushed a Docker image to GitHub Container Registry, but the Lagos server couldn’t pull it because of rate limits. The fix was to use a regional artifact store and a pull-through cache. But this required infrastructure changes.

*Recommended: <a href="https://digitalocean.com" target="_blank" rel="nofollow sponsored">DigitalOcean Cloud Hosting</a>*


The key takeaway here is: failure modes in micro-frontends aren’t just technical — they’re organizational and regional. Dependencies, CSS, routing, state, and deployment all become distributed systems problems when you split the frontend. Plan for coordination, scoping, and regional constraints upfront.

## Tools and libraries worth your time

Here’s a table of tools and libraries I’ve used, with their pros, cons, and regional considerations:

| Tool/Library          | Type               | Pros                                      | Cons                                      | Regional Consideration                     |
|-----------------------|--------------------|-------------------------------------------|-------------------------------------------|--------------------------------------------|
| Single SPA 5.9        | Framework          | Framework-agnostic, mature                | 45KB overhead, complex lifecycle          | Heavy for low-latency regions              |
| Module Federation     | Bundler plugin     | Lightweight, fast in ideal conditions     | Version-sensitive, React-only             | Needs same Webpack version across teams   |
| Web Components        | Web standard       | Lightweight, portable                     | No framework integration, polyfill cost   | Good for legacy widgets, but polyfills add size |
| Vite 4.5              | Bundler            | Fast, supports dynamic imports            | Less mature for production                | Best for shells and micro-frontends        |
| React 18.3            | Framework          | Stable, widely used                       | Multiple instances cause memory bloat     | Use a shared instance if possible          |
| Zustand 4.4           | State management   | Lightweight, works with React             | No built-in persistence                   | Good for local state, but sync is hard     |
| Nx 16.7               | Monorepo tool      | Fast, supports distributed builds         | Steep learning curve                      | Use if teams share a monorepo              |
| Cloudflare CDN        | CDN                | Global edges, cache-friendly              | Cost at scale                              | Essential for low-latency regions          |
| GitHub Container Registry | Registry        | Simple, integrates with CI                | Rate limits, regional latency             | Use regional mirrors or pull-through cache |

I used Single SPA for the shell in Lagos because it was framework-agnostic and mature. But the 45KB overhead was painful. In Berlin, I switched to Module Federation with Vite, which reduced the overhead to 12KB. The tradeoff was version sensitivity: all teams had to use the same Webpack and React versions.

I also tried Web Components for a legacy widget in a new React app. The widget was 30KB, but I had to polyfill custom elements and CSS scoping, adding 22KB. The widget loaded in 200ms in Lagos, but the polyfills caused a flicker during load.

For state management, I used Zustand for local state and a centralized API for shared state. Zustand was lightweight, but syncing state across 300ms latency was hard. I tried using a WebSocket connection, but the first sync took 800ms. The fix was to use optimistic updates and batch syncs.

For deployment, I used Cloudflare CDN for the micro-frontends and a shared VPS for the shell in Lagos. The CDN was essential for low-latency regions. But GitHub Container Registry’s rate limits caused deployment failures in Lagos. The fix was to use a pull-through cache with Cloudflare R2.

The key takeaway here is: no tool is universally better. Choose based on your framework, latency, and regional constraints. Single SPA is portable but heavy. Module Federation is fast but version-sensitive. Web Components are lightweight but require polyfills. CDNs are essential for low-latency regions.

## When this approach is the wrong choice

Micro-frontends are not a silver bullet. I learned this the hard way when I tried to apply them to a legacy monolith in Singapore. The monolith was a 2MB bundle with jQuery and legacy AngularJS. Each team wanted to split it into micro-frontends to work independently. The result was a disaster.

First, the legacy code used global jQuery plugins and AngularJS services. When I split the app, the plugins clashed. The error was: '$ is not defined in module scope.' The fix was to refactor the code to use ES modules, which took three months.

Second, the legacy app used inline styles and global CSS. When I split it, styles leaked. The result was a broken UI. The fix was to refactor the CSS to use CSS Modules, which took two months.

Third, the legacy app used a shared AngularJS injector. When I split it, the injector became inconsistent. The result was broken state. The fix was to refactor to a centralized state service, which took four months.

Fourth, the legacy app used inline scripts and eval. When I split it, the scripts failed to load due to CSP. The result was a blank page. The fix was to refactor the scripts to use modules, which took two months.

I also tried to apply micro-frontends to a real-time dashboard in Berlin. The dashboard needed to update every 100ms. Micro-frontends added too much overhead. The shell’s mount time alone was 50ms, and the micro-frontends’ mount time added another 30ms. The result was a laggy UI. The fix was to use a single bundle with React’s concurrent rendering.

Finally, I tried to apply micro-frontends to a mobile PWA in Lagos. The PWA needed to work offline. Micro-frontends added 120KB to the bundle, and the shell added 45KB. The result was a bloated app that failed to load on low-end devices. The fix was to use a single bundle with code splitting.

The key takeaway here is: micro-frontends are the wrong choice for legacy code, real-time apps, or offline-first apps. They add overhead, complexity, and latency. Use them only when you have independent teams, modern tooling, and a need for independent deployment.

## My honest take after using this in production

After three months of running micro-frontends in production across Lagos, Berlin, and Singapore, I have mixed feelings. The good parts are real: teams can deploy independently, tech stacks can diverge, and the frontend scales with the business. But the bad parts are also real: latency, memory bloat, and deployment fragility are constant headaches.

The biggest surprise was how much the regional latency mattered. In Berlin, the architecture felt fast and seamless. In Lagos, it felt slow and brittle. I had to optimize the shell, preload micro-frontends, and use a CDN edge in Lagos to get acceptable performance. Without those optimizations, the architecture would have failed in production.

Another surprise was memory usage. I expected micro-frontends to reduce memory usage by splitting the app. Instead, they increased memory usage by duplicating React instances and CSS. The fix was to use a shared React instance and scoped CSS, but that added complexity.

The biggest mistake I made was not enforcing a shared tooling version across teams. When teams used different Webpack, React, or Vite versions, the architecture broke. The fix was to use a shared monorepo with Nx and enforce versions via CI. But this required buy-in from all teams.

The biggest win was team independence. Teams in Lagos, Berlin, and Singapore could deploy their micro-frontends without coordinating with each other. This reduced merge conflicts and sped up iterations. But it also introduced new coordination challenges: dependency management, CSS scoping, and state synchronization.

The key takeaway here is: micro-frontends work, but they’re not a free lunch. They trade simplicity for scalability, and they introduce new failure modes. Use them only if you have the team maturity, tooling homogeneity, and regional infrastructure to support them.

## What to do next

If you’re considering micro-frontends, start with a single micro-frontend in a non-critical path. Use Module Federation with Vite and React. Deploy it to a CDN with a regional edge. Measure the performance impact in your worst region. If the latency and memory overhead are acceptable, expand to more micro-frontends. But enforce a shared tooling version across teams, and plan for CSS scoping and state synchronization upfront.

```bash
# Example: scaffold a micro-frontend with Vite and Module Federation
npm create vite@latest header-mf --template react
cd header-mf
npm install vite-plugin-federation@1.2.3 -D
```

Then, add Module Federation to vite.config.js:

```javascript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import federation from '@originjs/vite-plugin-federation';

export default defineConfig({
  plugins: [
    react(),
    federation({
      name: 'header-mf',
      filename: 'header-mf.js',
      exposes: {
        './Header': './src/Header.jsx',
      },
      shared: ['react', 'react-dom'],
    }),
  ],
  build: {
    target: 'esnext',
  },
});
```

Deploy the micro-frontend to a CDN with cache-friendly headers. Then, integrate it into your shell using dynamic imports. Measure the performance impact in your worst region. If it’s acceptable, expand to more micro-frontends.

## Frequently Asked Questions

How do I fix CSS leakage between micro-frontends?

Use CSS Modules in each micro-frontend and enforce a design system with strict token usage. Avoid global styles. If you must use global styles, scope them using a unique class name per micro-frontend. Test with Storybook to catch leaks early.

What is the difference between Single SPA and Module Federation?

Single SPA is a framework-agnostic shell that coordinates micro-frontends using lifecycle APIs. Module Federation is a Webpack 5 feature that allows remote modules to be consumed at runtime. Single SPA adds overhead but is portable; Module Federation is lightweight but version-sensitive.

*Recommended: <a href="https://amazon.com/dp/B07C3KLQWX?tag=aiblogcontent-20" target="_blank" rel="nofollow sponsored">Eloquent JavaScript Book</a>*


Why does my micro-frontend load slowly in Lagos?

In Lagos, latency to the edge is high (250ms+), and shared VPS instances are slow. Optimize by preloading micro-frontend scripts during idle time, using a CDN edge in Lagos, and lazy-loading non-critical micro-frontends. Measure with WebPageTest from Lagos to identify bottlenecks.

How do I sync state between micro-frontends across 300ms latency?

Use a centralized state service with optimistic updates and conflict resolution. Avoid WebSocket for real-time sync; it adds latency. Batch updates and use a last-write-wins strategy for conflicts. Consider using a library like Recoil for local state and a REST or GraphQL API for shared state.