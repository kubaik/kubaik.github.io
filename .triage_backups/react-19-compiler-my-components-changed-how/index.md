# React 19 Compiler: What It Rewrites and Where It Breaks

The React compiler is a build-time tool that rewrites components to insert memoization automatically. It is the most consequential change to everyday React code in years, because it moves decisions that used to live in your source files into a build step you rarely read. This article covers what the compiler actually changes, the failure modes teams report, how to measure whether it helps, and a checklist for deciding per-component whether to keep it enabled.

## What the compiler actually does

The compiler analyzes each component function, tracks which values are derived from props and state, and inserts memoization around computations and values whose identity would otherwise change on every render. It also attempts to skip re-renders when it can prove the inputs to a subtree have not changed.

The important word is "prove." The compiler is conservative: when it cannot establish that a value is stable, it bails out of optimizing that region and leaves your code effectively as written. That conservatism is why some components show large wins and others show none.

A schematic of the transformation, using the compiler's documented output shape:

```javascript
// Original
function UserAvatar({ user }) {
  const formattedName = formatName(user.firstName, user.lastName);
  return <div>{formattedName}</div>;
}

// Rewritten (conceptual)
function UserAvatar({ user }) {
  const formattedName = useMemo(
    () => formatName(user.firstName, user.lastName),
    [user.firstName, user.lastName]
  );
  return <div>{formattedName}</div>;
}
```

Two things follow from this. First, the memoization is only as good as the compiler's dependency analysis; if a dependency is an object whose identity changes each render, the memo will not help. Second, the compiler is not adding a new runtime primitive — it is emitting the same `useMemo` and `useCallback` you would have written, which means it inherits the same correctness constraints.

## Why the compiler is hard to reason about

Manual memoization is visible. You can read a component and see exactly which values are cached and which dependencies gate the cache. Compiler output is generated, often verbose, and rarely reviewed line by line. That asymmetry produces a specific class of bug: a value that used to be recomputed every render is now cached, and a downstream effect that depended on the fresh identity stops firing.

A typical failure mode looks like this:

```javascript
function ChatPanel({ socket, roomId }) {
  const handleMessage = useCallback(
    (event) => {
      appendMessage(roomId, event.data);
    },
    [roomId]
  );

  useEffect(() => {
    socket.on('message', handleMessage);
    return () => socket.off('message', handleMessage);
  }, [socket, handleMessage]);

  return <MessageList roomId={roomId} />;
}
```

The effect re-subscribes when `handleMessage` changes identity. If the compiler decides `handleMessage` is stable across renders where `roomId` did not change, the effect will not re-run — which is correct. The bug appears when the compiler's stability analysis disagrees with your mental model, for example when the callback closes over a value the compiler cannot track. The subscription then keeps a stale closure and messages land in the wrong room.

This is not a compiler defect. It is the same class of bug that manual `useCallback` produces when a dependency is omitted. The difference is that with manual memoization you can see the dependency array and audit it. With compiler output you have to read generated code or trust the analysis.

## What the compiler cannot optimize

The compiler's bail-out conditions are the practical boundary of its usefulness. Three categories recur.

**Values that depend on promises or async reads.** When a component reads a promise, the compiler generally cannot prove the resolved value is referentially stable across renders, so it skips memoizing computations derived from it. Async data paths therefore tend to keep their manual memoization.

**Mutable data structures.** If a reducer or helper mutates an object in place, the compiler cannot establish referential equality and bails out. Immutable updates are what make the analysis tractable.

**Values whose identity is intentionally unstable.** Refs, imperative handles, and callbacks that deliberately change identity to force an effect to re-run are all cases where automatic memoization fights your intent.

The practical consequence: the compiler helps most in components that compute derived values from stable primitives — formatted strings, sorted arrays built from immutable inputs, filtered lists — and helps least in components that bridge to the outside world.

## How to measure the effect instead of trusting a benchmark

Published before-and-after numbers for the compiler are not transferable, because the win depends entirely on how much derived state your components have and how often they re-render. Measure your own code. The procedure is short.

**Instrument re-renders.** Wrap the component under test in a counter that increments on each render, or use the React DevTools Profiler's "Highlight updates" mode and record a scripted interaction. The number to compare is renders per interaction, not renders per second.

**Record a fixed interaction.** Scroll a long list to the bottom, type into a search field, or open and close a modal — the same action, the same duration, before and after enabling the compiler. Without a fixed script the comparison is noise.

**Capture the profile.** In the React DevTools Profiler, record the interaction and export the flamegraph. Compare commit counts and the time spent in the component subtree.

**Compare bundle output.** Build once with the compiler enabled and once with it disabled, and diff the emitted chunk sizes. The compiler emits runtime helpers, so the bundle grows slightly even when memoization reduces work.

**Watch for behavior changes, not just timing.** Run your existing tests with the compiler enabled. Rendering tests that assert on effect firing order or on the number of times a callback runs are the ones most likely to catch a stability change.

A useful discipline is to enable the compiler on one subtree at a time and keep the measurement script in the repository, so a regression is attributable to a specific commit rather than to "the compiler."

## Compiler on versus manual memoization: a real comparison

The two approaches differ on axes that matter more than raw speed.

| Dimension | Compiler enabled | Manual memoization |
|---|---|---|
| Where caching decisions live | Build output | Source code |
| Reviewability | Requires reading generated code | Visible in the component |
| Boilerplate | Removed | Written and maintained per component |
| Behavior on async-derived values | Frequently bails out | Explicit, under your control |
| Behavior on mutable data | Bails out | Depends on your dependency arrays |
| Debugging a stale closure | Inspect compiler output | Read the dependency array |
| Onboarding cost | Learning the compiler's bail-out rules | Familiar React patterns |
| Risk profile | Silent behavior changes in edge cases | Omitted dependencies, caught by lint |

There is no universally correct column. The compiler reduces the volume of memoization code a team writes and removes a common source of missed optimizations. Manual memoization keeps every caching decision legible. The failure modes differ: the compiler fails by changing behavior in ways that are hard to see, and manual memoization fails by omission, which lint rules and review can catch.

## A worked decision example

Consider a component that renders a filtered and sorted list of 400 rows, where the filter and sort keys come from two props.

With manual memoization, the code is:

```javascript
function FilteredList({ rows, filter, sortKey }) {
  const visible = useMemo(
    () => rows
      .filter((row) => row.name.toLowerCase().includes(filter.toLowerCase()))
      .sort((a, b) => a[sortKey].localeCompare(b[sortKey])),
    [rows, filter, sortKey]
  );

  return (
    <ul>
      {visible.map((row) => (
        <li key={row.id}>{row.name}</li>
      ))}
    </ul>
  );
}
```

The reasoning for enabling the compiler here is straightforward. All three inputs are primitives or a stable array reference, the computation is pure, and the result is consumed directly in render with no effect depending on its identity. This is the shape the compiler handles well: it can prove the dependencies and it will memoize the filter-and-sort chain. The manual `useMemo` becomes redundant.

Now change one thing: make the list subscribe to a live data feed that pushes updates into `rows` by mutating the array in place. The compiler can no longer prove `rows` is stable, so it bails out of memoizing `visible`. Worse, if the component also passes a callback derived from `visible` into an effect, the identity of that callback becomes unpredictable. The correct move is to make the update immutable — replace the array rather than mutate it — which restores the compiler's ability to analyze the component and also fixes the underlying correctness issue.

The general rule from this example: the compiler's effectiveness is a function of how disciplined your data flow already is. It rewards immutability and pure derivations, and it exposes places where those properties are missing.

## Deciding per component, not per project

A project-wide switch is the wrong granularity. The compiler's behavior varies by component, and the components where it helps are exactly the ones where its analysis succeeds. A practical checklist:

- **Does the component compute derived values from stable primitives?** If yes, the compiler is likely to help and the manual memoization is likely redundant.
- **Does the component bridge to an external system — sockets, timers, imperative APIs?** If yes, treat it as a candidate for exclusion and verify effect behavior explicitly.
- **Does the component read promises during render?** If yes, expect the compiler to bail out; keep manual memoization for the derived values.
- **Does any reducer or helper mutate data in place?** If yes, fix the mutation first. The compiler's inability to optimize is a symptom, not the problem.
- **Do your tests assert on effect firing or callback invocation counts?** If yes, run them with the compiler enabled before shipping; they are your best detector of stability regressions.

Exclusion mechanisms vary by toolchain. Some integrations read a directive comment on the component; others use a configuration file listing components to skip. Check the documentation for the specific plugin version you are using rather than copying a directive from an older release, because the opt-out syntax has changed across versions.

## Failure-mode analysis: three ways the compiler surprises teams

**Stale closures in subscriptions.** A callback that the compiler caches closes over a value that has since changed. The subscription keeps delivering messages to the stale closure. Detection: assert on the value passed to the callback after a prop change. Fix: make the dependency explicit, or exclude the component.

**Effects that stop re-running.** An effect whose dependency is a value the compiler now considers stable no longer fires when you expected it to. Detection: a test that changes a prop and asserts the effect ran. Fix: verify the dependency is genuinely stable; if the effect must re-run, the dependency was never stable and the manual version had a latent bug too.

**Missing UI with no error.** A subtree stops updating because a render was skipped on the basis of a stability proof that does not hold. This is the hardest case, because there is no exception and no console warning. Detection: visual regression tests or a render-count assertion. Fix: exclude the component and investigate the data flow.

In all three cases the compiler is surfacing an assumption about identity that was previously implicit. That is useful information, but only if the team has tests that can see it.

## A 30-minute action

Pick the single component in your application that renders the most rows. Add a render counter to it — a `useRef` incremented at the top of the function body, logged once per interaction — and record the count for one fixed interaction, such as scrolling the list to the bottom. Then enable the compiler for that component only, repeat the same interaction, and compare the counts. If the count drops and your existing tests still pass, keep the compiler enabled there and move to the next component. If the count is unchanged, the compiler bailed out; inspect why, and fix the data flow before enabling it more widely. If the count drops but a test fails, you have found a real stability bug that manual memoization was hiding.
