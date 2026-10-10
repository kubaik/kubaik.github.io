# 7 bundling decisions that still matter

## What this list actually solves

Modern frameworks ship with a build tool that already bundles your code. Vite, Next.js, Remix, SvelteKit, and Astro each wrap esbuild, Rollup, SWC, or Turbopack and expose a `build` command that produces a working `dist/` directory. So the reasonable question is: why would bundling and code-splitting decisions still matter?

The answer is that defaults are tuned for the median app, and the median app is a marketing site with a handful of routes. The moment an app has authenticated dashboards, heavy third-party libraries, or a slow first paint on a mid-range Android device, the defaults stop being free. A common failure mode is discovering this the week before launch, when a Lighthouse score drops from the 90s into the 40s and nobody can explain which import caused it.

This list covers seven bundling and splitting decisions that still produce measurable differences in shipped JavaScript size and load behavior, even with a modern framework doing most of the work. Each entry is a mini-review: what the option is, where it genuinely helps, where it falls over, and who should pick it.

## Evaluation criteria

Before the list, the criteria — otherwise every entry blurs into "it depends."

- **Effect on initial JS payload.** Does this decision reduce the bytes a first-time visitor must download and parse before the page is interactive?
- **Effect on cache hit rate.** After a deploy, how much of what the browser already cached is invalidated?
- **Operational cost.** How much configuration, naming discipline, or CI tooling does it require to keep working?
- **Failure mode.** What breaks, and how loudly, when someone gets it wrong?
- **Reversibility.** If a team adopts it and regrets it, how painful is the removal?

A decision that scores well on payload but badly on operational cost is often a net loss for a small team. A decision that's cheap to reverse is worth trying even if the payoff is uncertain.

## The seven decisions

### 1. Route-level splitting as the baseline

**What it does.** Every modern framework with a file-based router (Next.js App Router, Remix, SvelteKit, TanStack Router) splits at the route boundary by default. Each route becomes its own chunk, loaded on navigation. This is the lowest-effort split available and it's the one that should never be disabled.

**Strength.** It maps cleanly to user intent. A visitor landing on `/pricing` doesn't download the dashboard code. The framework handles the chunk naming and the preload hints automatically in most cases.

**Weakness.** Route-level splitting is coarse. A route that imports a 400 KB charting library ships that library to every visitor of that route, even the ones who never open the chart panel. It also does nothing for the landing route itself, which is where most of the traffic lands and where the initial payload matters most.

**Best for.** Everyone. This is the floor, not a ceiling. If it has somehow been disabled — for example, by importing every page component into a single layout — re-enable it before reading further.

### 2. Component-level lazy loading with dynamic import

**What it does.** `import()` at the component level defers a module until the code path that needs it runs. In React this usually pairs with `lazy()` and `Suspense`; in Vue it's `defineAsyncComponent`; in Svelte it's a dynamic import inside an `{#await}` block.

```jsx
import { lazy, Suspense } from 'react';

const HeavyChart = lazy(() => import('./HeavyChart'));

export function Dashboard({ showChart }) {
  if (!showChart) return <Summary />;
  return (
    <Suspense fallback={<ChartSkeleton />}>
      <HeavyChart />
    </Suspense>
  );
}
```

**Strength.** This is where real savings live. A chart, a rich text editor, a PDF viewer, a map — these are the modules that push a route from 80 KB to 500 KB. Deferring them behind a user action or a viewport intersection removes them from the critical path entirely.

**Weakness.** The failure mode is a layout shift or a spinner where the user expected instant content. Lazy-loading something above the fold makes the user pay a round trip they didn't need to pay. It's also easy to over-split: forty tiny chunks each under 2 KB produce more HTTP overhead than one 40 KB chunk, especially on HTTP/1.1 connections or behind a proxy that doesn't handle multiplexing well.

**Best for.** Apps with one or two genuinely heavy dependencies that only some users touch. If the whole app is under 150 KB gzipped, this is premature.

### 3. Manual chunk grouping via the bundler config

**What it does.** Rollup's `manualChunks` option and Vite's `build.rollupOptions.output.manualChunks` let you name which modules land in which output chunk. Webpack has the equivalent in `splitChunks.cacheGroups`.

```js
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          editor: ['@tiptap/core', '@tiptap/react'],
        },
      },
    },
  },
};
```

**Strength.** It gives you control over cache invalidation. If `vendor` is a stable chunk that only changes when React is upgraded, a deploy that touches only app code leaves that chunk cached in the browser. For repeat visitors this is a meaningful win — they re-download a few KB instead of the whole bundle.

**Weakness.** It's easy to create a chunk that's imported by multiple routes but never actually shared, which defeats the point. Worse, a bad grouping can create a circular chunk dependency that the bundler resolves by duplicating code, silently inflating the total. The config also drifts: six months later nobody remembers why `editor` is separate.

**Best for.** Apps with a stable, large dependency set (a UI kit, a framework, a charting library) and a deploy cadence where cache invalidation actually matters. For a small internal tool deployed weekly to the same fifty users, it's not worth the config.

### 4. Prefetching on hover or viewport

**What it does.** Frameworks expose a prefetch mechanism — Next.js `<Link prefetch>`, Remix `<Link prefetch="intent">`, SvelteKit's `data-sveltekit-preload-data` attribute. These load the next route's chunk before the user clicks, so navigation feels instant.

**Strength.** It converts a perceived latency problem into a bandwidth problem, and bandwidth is usually cheaper than attention. On a fast connection, prefetching on hover makes navigation feel like a native app.

**Weakness.** On a slow connection, prefetching competes with the critical path. A user on a congested mobile network who hovers over five links in a nav bar can end up downloading five route bundles while the page they're actually on is still parsing. Some frameworks have moved to viewport-based prefetching to limit this, but the underlying tension remains.

**Best for.** Apps where navigation is the dominant interaction and the routes are small. If routes are heavy, prefetch on intent only, not on viewport.

### 5. Tree-shaking discipline and side-effect flags

**What it does.** Tree-shaking removes unused exports at build time. It only works when the bundler can prove a module has no side effects. The `sideEffects` field in `package.json` is the signal, and `"sideEffects": false` tells the bundler it can drop any unused export from that package.

**Strength.** When it works, it's free payload reduction with zero runtime cost. A library that exports forty utilities but is imported for one can shrink to a fraction of its published size.

**Weakness.** `"sideEffects": false` is a lie if the package has any module that mutates globals on import — a polyfill, a CSS import, a registration call. Marking such a package as side-effect-free lets the bundler delete code that was load-bearing, producing a runtime error that only appears in production builds. This is a well-documented class of bug and it's why library authors are cautious about the flag.

**Best for.** Library authors, and app developers auditing their dependency tree. If you maintain a package, set `sideEffects` correctly; if you consume one that's misconfigured, file an issue rather than patching around it.

### 6. Differential serving and modern-target output

**What it does.** Build tools can emit a modern bundle (ES2020+, smaller, no transpilation of async/await or optional chaining) and a legacy bundle for older browsers, then let the browser pick via `<script type="module">` and `<script nomodule>`. Vite has this via `@vitejs/plugin-legacy`; some frameworks handle it internally.

**Strength.** Modern browsers get smaller, faster code. The legacy bundle is only downloaded by browsers that need it, so the cost is paid by a shrinking minority.

**Weakness.** It doubles build time and doubles the artifacts to reason about. The `nomodule` trick has known quirks in some older Safari versions, and if analytics or a CDN mangles the script tags, the fallback silently fails. For most apps targeting users on browsers released in the last three years, the legacy bundle is dead weight.

**Best for.** Consumer apps with a long tail of old devices. If analytics show a negligible share of traffic on browsers that can't handle ES2020, skip it and set a modern `build.target` instead.

### 7. Measuring what you shipped

**What it does.** Bundle analysis tools — `rollup-plugin-visualizer`, `webpack-bundle-analyzer`, `source-map-explorer` — read the build output and source maps and show which module contributed which bytes.

**Strength.** It's the only way to answer "why is this chunk 600 KB" without guessing. A treemap view makes it obvious when a single dependency is responsible for most of a chunk, and that's usually the moment a team discovers a library imported for one function.

**Weakness.** It measures the build artifact, not the user experience. A chunk that's large but cached, or large but loaded after interactivity, doesn't hurt the metric that matters. Treating bundle size as the goal rather than a proxy leads to over-splitting and worse real-world performance.

**Best for.** Everyone, but especially teams that don't yet have a size budget in CI. Wire it up once, look at the output, and set a threshold that fails the build if the main chunk grows past it.

## How to measure the effect of any of these

Every claim above is a hypothesis about your app until you instrument it. The measurement loop is the same for all seven decisions:

1. **Capture the baseline.** Run a production build with source maps enabled. Record the total transferred bytes for the initial route and the number of requests, using the browser's network panel with cache disabled and network throttling set to a mid-range mobile profile. Repeat three times and note the spread; single runs on a shared machine are noise.
2. **Record the timing metric you actually care about.** Time-to-interactive, Largest Contentful Paint, or the timestamp of the first user input the app responds to. A bundle that shrinks by 200 KB but doesn't move any of these is not a win.
3. **Make exactly one change.** Add one dynamic import, or one `manualChunks` entry, or one prefetch attribute. Not three.
4. **Re-measure with the same procedure.** Same device profile, same throttling, same number of runs.
5. **Compare cache behavior across a simulated deploy.** Build once, load the page, then rebuild with a trivial source change and load again. In the network panel, count how many bytes come back `200` versus `304`/`from disk cache`. This is the only honest way to evaluate manual chunk grouping, and it takes about ten minutes.

If a change doesn't move the metric from step 2, revert it. Configuration that doesn't pay for itself is a liability, because it has to be understood by everyone who touches the build later.

## The strongest default, and why

If you do nothing else, do route-level splitting (already the default) plus component-level lazy loading for anything over roughly 50 KB that isn't needed for first paint. That combination captures most of the available win with the least configuration and the least chance of a subtle production-only bug.

Manual chunk grouping, prefetching, differential serving, and tree-shaking audits are all second-order. They're worth doing, but they're optimizations on top of a baseline that has to be right first. A team that splits routes and lazy-loads the chart library will beat a team that has an elaborate `manualChunks` config but ships the chart on the landing page.

## Honorable mentions worth knowing about

- **CSS splitting.** Most frameworks split CSS per route alongside JS. The failure mode is a flash of unstyled content when a route's CSS loads after its markup. Inlining critical CSS for the landing route is a common fix.
- **Module federation.** Useful for micro-frontends where independent teams deploy separately. Heavy operational overhead; rarely worth it for a single-team app.
- **Import maps in the browser.** Let you skip the bundler for small apps, but you lose tree-shaking and minification unless you add a separate step. Rarely a net win outside demos.

## Options that look appealing but fail in practice

**Splitting every component.** The theory is maximum cache granularity. In practice, dozens of tiny chunks produce more requests, more waterfall depth, and more chances for a chunk to fail to load behind a flaky network. The bundler's default heuristics are usually better than hand-tuned micro-splitting.

**Aggressive prefetching of everything.** Preloading all route chunks on idle sounds free. It isn't — it competes for bandwidth with the assets the current page actually needs, and on metered connections it spends the user's data on pages they may never visit.

**Chasing a single bundle-size number.** A 200 KB bundle that loads after interactivity is better than a 100 KB bundle on the critical path. Optimizing the number without measuring when the bytes load inverts the goal.

## How to choose based on your situation

| Situation | Start with | Avoid |
|---|---|---|
| Marketing site, few routes | Route splitting (default) | Manual chunks, differential serving |
| Authenticated dashboard | Route splitting + lazy-load heavy widgets | Viewport prefetch on every link |
| Consumer app, old devices | Differential serving + route splitting | Micro-splitting every component |
| Library or design system | Correct `sideEffects`, ship ESM | `manualChunks` (consumer's job) |
| Internal tool, small user base | Defaults | Any manual chunk config |

The pattern: the more varied your users and the more heavy dependencies you have, the more the later entries on this list earn their keep. For a small, homogeneous user base, the defaults are correct and the config is a liability.

## Frequently asked questions

**Why is my bundle so big even though I use a modern framework?**

Frameworks split routes, not dependencies. If a single route imports a charting library, an editor, or a date library with locale data, that route's chunk carries all of it. Run a bundle visualizer and look for one dependency dominating a chunk — that's almost always the cause.

**Does code-splitting actually make my site faster?**

It reduces the bytes on the critical path, which usually reduces time-to-interactive, but only if the split module isn't needed for first paint. Splitting something the user sees immediately just adds a round trip. The win comes from deferring things the user hasn't asked for yet.

**What's the difference between code-splitting and lazy loading?**

Code-splitting is the build-time act of producing separate chunks. Lazy loading is the runtime act of loading a chunk only when needed. You can split without lazy-loading (the chunk loads eagerly anyway) and you can lazy-load without splitting (rare, and usually a mistake). They're complementary, not the same thing.

**How do I know if my chunking is wrong?**

Two signals: a chunk that's much larger than the code you think is in it (usually a transitive dependency), and a chunk that's imported by many routes but never cached between them. A bundle visualizer shows the first; the browser's network panel on a second navigation shows the second.

**Should I set `sideEffects: false` in my package.json?**

Only if it's true. If any module in the package mutates globals, registers a custom element, or imports CSS for its side effect, marking it false lets the bundler delete code that runs at import time. This produces bugs that only appear in production builds, which is the worst kind.

## Final recommendation

Open the build output directory, find the largest JavaScript chunk, and open it in a bundle visualizer — `npx source-map-explorer dist/assets/*.js` works if source maps are enabled. Look at the top three modules by byte count. If any of them isn't needed for the first paint of the route that loads it, wrap it in a dynamic import and move on. That one change is usually the difference between a bundle that ships and one that gets rewritten.
