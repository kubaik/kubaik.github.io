# Signals: derived state without the performance tax

## The problem Signals are meant to solve

Most state bugs in component apps are not "where does the data live" bugs. They are "when does the derived value recompute" bugs. A store holds a source value, a selector derives something from it, and a component reads that derived value. The moment two stores depend on the same source, or one derived value feeds another, the cost of keeping everything consistent climbs faster than the number of stores.

The failure mode is predictable. A selector memoizes on the wrong dependency list, so it recomputes on every render. Or it memoizes correctly but a parent re-renders and recreates the selector, invalidating the cache. Or the derived value is correct but stale because an effect that was supposed to invalidate it never ran. None of these are framework bugs; they are consequences of manual dependency tracking.

Signals move the dependency tracking from the developer to the runtime. A signal is an observable value with a getter. Reading it inside a computation registers a dependency; writing it marks dependents dirty and schedules them. Because the graph is built at read time, you do not maintain a dependency array by hand, and a derived value that nobody reads costs nothing to keep defined.

That last property is the one that changes architecture. In a selector-based store, every derived value is a function you must remember to call and memoize. In a signal graph, a computed value with no active subscribers is simply not evaluated.

## What "fast enough" actually means

Before choosing a library, define the budget. The useful frames of reference:

- **Frame budget.** A 60 Hz display gives roughly 16.7 ms per frame. A state update that triggers layout and paint must fit inside that, alongside everything else the frame is doing. A budget of 8 ms for the update itself is a reasonable target; it leaves headroom.
- **Interaction budget.** For a keystroke or tap, the perceived threshold for "instant" is around 100 ms. Anything above that reads as lag even if no frame is dropped.
- **Allocation budget.** A derived value that allocates a new object on every recomputation puts pressure on the garbage collector. On memory-constrained devices this shows up as periodic stutter rather than steady slowness, which makes it easy to misdiagnose.

These are starting points, not laws. The point is to write the budget down before benchmarking, because otherwise every result looks acceptable.

## How to measure a signal library honestly

The measurements that matter are cheap to take and easy to fake. Here is a procedure that produces numbers you can defend.

**Update latency.** Instrument the write, not the render. Wrap the signal write in `performance.mark` / `performance.measure` and read the measure in the same task, before the browser has a chance to paint. If you want to include render cost, use the Performance panel and look at the scripting and rendering time for the frame that contains the update. Throttle the CPU to simulate a slower device — the Performance panel's CPU throttling multiplies all main-thread work, which is a reasonable approximation of a mid-tier phone. Record the median and the 95th percentile; the median tells you the typical case, the 95th tells you whether users will occasionally see a stall.

**Memory.** Take a heap snapshot before the workload, run a fixed number of updates (1000 is a common choice because it is large enough to expose leaks and small enough to run in a loop), force garbage collection, and take a second snapshot. Compare retained size, not allocated size. Allocated size includes garbage that has not been collected yet and will mislead you. In Chrome DevTools the "Collect garbage" button in the Memory panel does this; in Firefox Profiler, take a snapshot after a forced GC.

**Recomputation count.** This is the measurement most people skip and the one that explains the others. If a derived value recomputes 1000 times when its inputs changed once, latency and memory will both look bad and you will blame the library. Count evaluations by incrementing a counter inside the computed function. A correct signal graph should evaluate each computed value at most once per batch of writes.

**Bundle cost.** Measure the minified and gzipped size of the code you actually import, not the package as published. Tree-shaking removes different amounts depending on which entry point you use. A quick check is to build a minimal app that imports only the signal primitives and inspect the output bundle.

A useful sanity check: if a library's update latency is faster than the time it takes to call a function, you are probably measuring the wrong thing. Signals are fast, but they are not free.

## The landscape, by category

Rather than ranking specific versions, which go stale, it helps to understand the categories and what each one trades away.

### Standalone signal primitives

These are small packages that implement signals and computeds with no framework dependency. They work in browsers, in server runtimes, and in workers. The trade-off is that you get the primitive and nothing else: no devtools integration, no persistence, no time-travel debugging. If you need those, you build them or add another dependency.

The API surface is intentionally tiny — typically a `signal` factory, a `computed` factory, and an `effect` or `batch` function. That smallness is the feature. There is very little to learn and very little to go wrong.

**When to choose this:** you want reactivity in a non-UI context (a server, a worker, a CLI), or you want to add signals to an existing framework incrementally without adopting a new component model.

**What to watch:** the TypeScript types are often permissive. It is frequently possible to write to a signal's `.value` from a context where that write is a bug. Some libraries expose a read-only accessor type to prevent this; check whether yours does.

### Framework-integrated signals

Several frameworks now ship signals as a first-class primitive. The advantage is that the framework's rendering layer understands the graph, so a signal read inside a template or component body registers a fine-grained subscription. Only the DOM nodes that depend on the changed signal update.

The trade-off is coupling. The signal implementation is usually tuned to the framework's scheduler, and using it outside that framework loses the automatic tracking. In some cases the reactivity only works inside the framework's compiler output, so hand-written code behaves differently from compiled code.

**When to choose this:** you are already in that framework and want fine-grained updates without a rewrite.

**What to watch:** framework integration often pulls in the framework's other dependencies. Check what the signal package imports before assuming it is lightweight.

### Reactive-effect systems with scoping

Some reactivity systems group effects into scopes that can be stopped as a unit. This matters in single-page apps where components mount and unmount frequently: without explicit teardown, effects created inside a component keep running after the component is gone, holding references to its state.

The scoping API is the safety mechanism. The failure mode is forgetting to call it. A common pattern is to create effects inside a loop or inside a callback that runs per-item; each iteration creates a scope that must be stopped when the item is removed. If it is not, the graph grows without bound.

**When to choose this:** you have many short-lived reactive regions and want a single teardown call per region.

**What to watch:** measure retained heap after repeated mount/unmount cycles. A leak shows up as a retained size that grows linearly with the number of cycles.

### Observable-to-signal bridges

If an existing codebase is built on streams or observables, a bridge that converts an observable into a signal lets you adopt signals incrementally rather than rewriting the data layer. The conversion is not free: each observable subscription introduces a scheduling hop, and the bridge code itself adds to the bundle.

**When to choose this:** you have a large existing stream-based data layer and want new UI code to use signals.

**What to watch:** the hop adds latency per update. Measure it rather than assuming it is negligible; for high-frequency streams it can dominate the cost.

## A worked example: replacing a selector store

The following is a minimal illustration of the pattern, not a benchmark. It shows what changes when dependency tracking moves from the developer to the runtime.

Before, with a selector-based store, the derived value is a function with a hand-maintained dependency list:

```javascript
import { createStore, createSelector } from 'redux';

const store = createStore(reducer);
const selectExpensiveData = createSelector(
  [selectA, selectB, selectC],
  (a, b, c) => expensiveComputation(a, b, c)
);
```

The dependency list `[selectA, selectB, selectC]` is a promise that `expensiveComputation` reads nothing else. If it later reads `selectD`, the memoization silently returns stale values. Nothing in the type system catches this.

After, with signals, the dependency list does not exist:

```javascript
import { signal, computed, effect, batch } from '@preact/signals-core';

const a = signal(0);
const b = signal(0);
const c = signal(0);

const expensiveData = computed(() => expensiveComputation(a.value, b.value, c.value));

// Subscribe to changes; the effect re-runs when any dependency changes.
const dispose = effect(() => {
  console.log('Derived value changed:', expensiveData.value);
});
```

The computed reads `a`, `b`, and `c` through their getters, so the runtime records exactly those dependencies. If `expensiveComputation` later reads a fourth signal, the graph updates automatically. There is no list to keep in sync.

Two details worth noting. First, `batch` groups multiple writes so dependents evaluate once rather than once per write:

```javascript
batch(() => {
  a.value = 1;
  b.value = 2;
  c.value = 3;
});
```

Without the batch, `expensiveData` would evaluate after each assignment. With it, it evaluates once. This is the mechanism that makes the recomputation count meaningful.

Second, `effect` returns a dispose function. Calling it unsubscribes the effect and releases its references. Omitting this call is the most common source of leaks in signal graphs.

## Failure modes worth designing around

**Circular dependencies.** If computed A reads B and B reads A, the graph has no valid evaluation order. Well-behaved implementations detect the cycle and throw rather than looping. The fix is architectural: introduce a source signal that both computeds read, and make one of them write to a separate signal that the other reads, so the cycle becomes a path. This usually costs a few lines and removes the possibility of an infinite update loop.

**Retained graphs.** A signal holds strong references to its subscribers, and a computed holds references to its dependencies. A disconnected component that never disposes its effects keeps its entire graph alive. In a long-running server process, this shows up as heap growth that tracks the number of connections rather than the number of active ones. Instrument by counting live graphs and comparing to the number of active clients; if the two diverge, something is not being torn down.

**Async interleaving.** When two writes arrive close together and a computed derives from both, the intermediate state may be observed by an effect that runs between them. Batching the writes prevents this. If the writes come from separate async sources, batching at the point of arrival is not enough; the writes must be funneled through a single synchronous step. This is a scheduling problem, not a signals problem, but signals make it visible because the derived value updates eagerly.

**Unbounded derived chains.** A computed that reads another computed that reads another, each allocating a new object, produces allocation proportional to the chain depth on every update. Flatten where possible, and prefer returning primitives from hot computeds.

## Integration notes

Signals are not tied to the DOM. The same primitive works in a server runtime, a worker, or a desktop app's frontend, provided the environment supports the JavaScript features the library relies on. The main portability concern is weak references: some runtimes do not expose `WeakRef`, and libraries that use it for cleanup need a polyfill. Check the target runtime's supported features before assuming a browser-tested library will work unchanged.

In desktop shells that bridge a native backend to a web frontend, the inter-process channel adds latency that dwarfs any signal overhead. Signals are still useful there for organizing frontend state, but they do not make cross-process updates fast. Measure the channel separately.

## A decision checklist

Before adopting a signal library, answer these:

1. **Does the runtime support the primitives the library needs?** Check for `WeakRef` and any proposed APIs.
2. **Is the dependency tracking automatic in the code you will actually write?** If it only works inside a compiler, hand-written code will behave differently.
3. **What is the teardown story?** Every effect needs a dispose path. Confirm the API provides one and that your code calls it.
4. **What does the import cost after tree-shaking?** Build a minimal app and inspect the bundle.
5. **Can you count recomputations?** If not, you cannot diagnose a performance regression.
6. **Does the library detect cycles?** A silent infinite loop is worse than a thrown error.
7. **How does it behave under batched writes?** Confirm that dependents evaluate once per batch.
8. **What is the migration path if you change your mind?** Standalone primitives are easier to remove than framework-integrated ones.

## FAQ

**Do signals replace a state management library?**
They replace the derived-state layer. You still need somewhere to hold source state, and you still need a strategy for persistence, undo, and devtools. Signals make the derived layer automatic; they do not make the rest disappear.

**Are signals faster than selectors?**
Not inherently. They are faster when the alternative recomputes more than necessary, which is common but not universal. Measure recomputation counts in both approaches before assuming a win.

**Can signals and a selector store coexist?**
Yes. A common pattern is to keep the store as the source of truth and expose signals derived from it, so new UI code gets fine-grained updates without rewriting the store.

**Why does my computed run more often than expected?**
Usually because a write is not batched, or because the computed reads a value that changes on every render (a new object identity, for example). Count evaluations and log the inputs.

**Do signals work in server-side rendering?**
The primitives work, but the subscription model is designed for long-lived consumers. For SSR you typically evaluate the graph once per request and serialize the result. Check whether the library provides a helper for this or whether you need to manage scope per request.

## Do this in the next 30 minutes

Pick one derived value in your app that recomputes more often than you think it should. Add a counter inside it, log the count to the console, and interact with the UI for a minute. If the count is much higher than the number of times the underlying data actually changed, you have found a candidate for a signal graph — and you now have a before-number to compare against after the change.
