# Global state is the wrong tool for AI features

AI features fail in a characteristic way: the demo is a request-response form, and the product is a stream of concurrent, cancellable operations. The state model that carries the demo rarely survives contact with the product. This article explains where a single global store breaks down for AI features, what to model locally instead, and how to measure whether your own app has crossed the line.

## The conventional wisdom, and where it stops being true

When teams add AI features, they commonly reach for the same global store that already holds auth and preferences: a single source of truth with predictable updates. That choice is reasonable when the AI pipeline is a few prompts behind a REST endpoint. It becomes fragile when the feature grows streaming inference, tool calling, and user-facing undo/redo.

A typical failure mode looks like this. The conversation is modeled as one ever-growing array in the global store. Every prompt appends to it. Every render pass serializes or derives from the whole array. Memory grows monotonically for the life of the session, and a single selector recomputes the entire history on each keystroke.

The second failure mode is concurrency. Tool calls are tracked with a single global status field — `idle`, `fetching`, `streaming`, `error`. Because React state updates are asynchronous and batched, two overlapping tool calls can each write that field. The UI can render a state that never logically existed, such as a spinner disappearing between two in-flight calls, or an analytics event firing twice for one logical operation.

The third is undo. If the undo stack holds full conversation snapshots or replays the whole history to compute the previous state, undo cost grows with session length. A user who has made two hundred tool calls pays for all two hundred to reverse the last one.

None of this means global state is bad. It means it is a poor fit for state that mutates many times per interaction, is owned by one subtree, and needs fine-grained reversal.

## What global state is genuinely good at

Three categories justify a global store, and they share a property: low write frequency relative to read frequency.

**Session-scoped identity and preferences.** Auth tokens, locale, theme, and feature flags change rarely, are read everywhere, and have a natural single owner. A global store with one write per session is fine.

**Read-only derived views.** An analytics dashboard aggregating AI usage across many sessions reads a small, predictable dataset. Memoized selectors over a global store work well here because the inputs change on a schedule you control.

**Cross-cutting concerns with low churn.** Logging configuration, telemetry sinks, and experiment assignments are consumed by many components but change on deploy boundaries, not per keystroke.

The boundary is write frequency and ownership. If a piece of state changes more than a handful of times per user interaction, or if only one component subtree reads it, a global store adds coordination cost without buying anything.

## A different mental model: local, ephemeral state machines

Treat each AI interaction surface as a small state machine that owns its own state and communicates through explicit, typed events. The chat input owns the prompt draft. The tool panel owns tool selection and parameters. The streaming output owns the partial response buffer. The undo stack owns the operation log.

These machines share immutable snapshots via events rather than a mutable global tree. This is the actor model with JavaScript ergonomics: receive an event, update local state, emit a new event, never mutate external state.

Start with a typed event registry so the contract is explicit and greppable:

```typescript
// src/events/ai-events.ts
export const AIEvent = {
  PromptDraftUpdated: 'prompt.draft.updated',
  ToolSelected: 'tool.selected',
  ToolCallStarted: 'tool.call.started',
  ToolCallChunk: 'tool.call.chunk',
  ToolCallFinished: 'tool.call.finished',
  Undo: 'undo',
  Redo: 'redo',
} as const;

export type AIEventName = (typeof AIEvent)[keyof typeof AIEvent];

export interface AIEventPayload {
  traceId: string;
  conversationId: string;
  timestamp: number;
  data: unknown;
}
```

Each component subscribes to only the events it cares about. The chat input listens to `PromptDraftUpdated`. The undo stack listens to `ToolCallStarted` and `ToolCallFinished` to build its operation log. The tool panel listens to `ToolSelected`. Components can be added or removed without editing a central reducer, and the undo stack can be disabled on a client without touching chat logic.

A minimal event bus is genuinely small. The core is a map from event name to a set of handlers:

```typescript
// src/events/bus.ts
type Handler = (payload: AIEventPayload) => void;

export class EventBus {
  private handlers = new Map<string, Set<Handler>>();
  private seen = new Map<string, number>();
  private dedupeWindowMs = 50;

  on(event: string, handler: Handler): () => void {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set());
    this.handlers.get(event)!.add(handler);
    return () => this.handlers.get(event)?.delete(handler);
  }

  emit(event: string, payload: AIEventPayload): void {
    // At-least-once delivery with a time-windowed dedupe key.
    const key = `${payload.traceId}:${event}`;
    const last = this.seen.get(key);
    if (last !== undefined && payload.timestamp - last < this.dedupeWindowMs) return;
    this.seen.set(key, payload.timestamp);

    for (const handler of this.handlers.get(event) ?? []) {
      try {
        handler(payload);
      } catch (err) {
        // One failing subscriber must not stop the others.
        console.error(`handler failed for ${event}`, err);
      }
    }
  }
}
```

Two details matter more than the implementation. First, the dedupe window is a heuristic, not a guarantee: it suppresses duplicate emissions that arrive within `dedupeWindowMs` for the same trace and event name. Choose the window from the observed retry interval of your transport, and log suppressed events so you can audit them. Second, subscriber isolation via `try/catch` prevents one broken handler from silently dropping events for every other subscriber.

## A worked example: undo that stays O(1)

Consider a user who has asked twelve questions and triggered thirty tool calls. Under a snapshot model, the undo stack holds thirty entries, each potentially containing the full conversation. Reversing the last tool call means finding the previous snapshot, which in a naive implementation means scanning or replaying.

Under the local model, each tool call registers an operation record in the store that owns it:

```typescript
interface Operation {
  id: string;
  kind: 'tool-call' | 'prompt';
  conversationId: string;
  inverse: () => void;   // how to undo this single step
  timestamp: number;
}

class UndoStack {
  private stack: Operation[] = [];
  private maxDepth = 100;

  push(op: Operation): void {
    this.stack.push(op);
    if (this.stack.length > this.maxDepth) this.stack.shift();
  }

  undo(): boolean {
    const op = this.stack.pop();
    if (!op) return false;
    op.inverse();
    return true;
  }
}
```

Undo is now a pop and a single inverse call. Cost is independent of how many operations came before. The bounded depth keeps memory flat. And because each operation carries its own inverse, a user can undo a single tool step without unwinding the conversation — a feature that is awkward to express when the only unit of history is a whole-conversation snapshot.

The trade-off is that each operation must supply a correct inverse. For a tool call that mutated server state, the inverse is a compensating request, not a local mutation, and compensating requests can fail. Decide up front which operations are locally reversible and which require a server round trip, and surface that distinction in the UI rather than pretending all undo is free.

## How to measure whether you have this problem

Do not trust intuition about memory. Instrument it. The following is a measurement plan, not a result.

**Instrument the state container.** Sample the size of your AI state on a timer and record the 50th, 95th, and 99th percentiles per session. In the browser, `performance.memory.usedJSHeapSize` is available in Chromium-based browsers and is approximate; treat it as a trend line, not an exact figure. In Node, `process.memoryUsage().heapUsed` gives the same kind of signal.

```typescript
// Browser-side sampler: log heap and AI-state size every 10s.
const AI_STATE_KEY = 'ai.conversation';

function sample() {
  const mem = (performance as any).memory;
  const aiState = JSON.stringify(
    // replace with your actual selector
    (window as any).__AI_STATE__?.[AI_STATE_KEY] ?? null
  );
  console.log(JSON.stringify({
    t: Date.now(),
    heapMB: mem ? +(mem.usedJSHeapSize / 1048576).toFixed(1) : null,
    aiStateKB: +(aiState.length / 1024).toFixed(1),
  }));
}

setInterval(sample, 10_000);
```

**Compare two branches under the same load.** Run the same scripted user journey against the global-store build and the local-state build, with the same number of sessions and the same think time. Compare the p99 of `heapMB` and `aiStateKB` at the end of each session, not the mean. A mean hides the long tail, and the long tail is where out-of-memory kills happen.

**Count renders per interaction.** Wrap the component that renders the conversation in a render counter and log it per keystroke. If one keystroke produces a render of the full conversation, the selector is too coarse. The fix is either a narrower selector or local ownership of the draft.

**Count duplicate side effects.** Tag every analytics or telemetry event emitted from an AI state transition with the operation ID. Then query for operation IDs with more than one event. A nonzero count is direct evidence of the concurrency bug described earlier, independent of any render timing.

**Measure hydration separately.** If you server-render, measure the time from HTML parse to interactive for a session with a large AI state. Serializing and deserializing a large conversation is a distinct cost from rendering it, and it lands on the slowest devices.

Set your own thresholds from these measurements. A p99 heap figure that is alarming in one application may be routine in another; what matters is whether the curve is flat or rising over the session.

## Failure modes to watch for in the local model

Local state machines are not free. Four failure modes recur.

**Event sprawl.** Without a registry and a naming convention, event names proliferate and nobody knows which component emits what. Keep every event name in one module, and add a lint rule or schema check that rejects unregistered names at build time.

**Lost causality.** Fire-and-forget events are hard to debug. Every payload should carry a trace ID, a timestamp, and the emitting component. With those three fields you can reconstruct the exact sequence that produced a bug from logs alone.

**Unbounded local stores.** Moving state local does not automatically bound it. A streaming output buffer that appends every chunk will still grow. Cap buffers by policy — keep the last N chunks, or the last N kilobytes — and drop the rest.

**Cross-machine consistency.** Once state is distributed across components, any invariant that spans two of them needs an explicit reconciliation step. If the tool panel and the undo stack must agree on which tool is active, make that a single event with a single owner rather than two components inferring it independently.

## Decision checklist

Work through these questions before choosing a state model for a new AI feature.

- How many times does this state change per user interaction? More than a handful points to local ownership.
- How large is the state at peak, after a long session? Measure it, do not estimate.
- How many components read it? If the answer is one subtree, a global store buys nothing.
- Does undo need to be per-operation or per-conversation? Per-operation favors local operation records.
- Does the state need to survive a full page reload? If not, session-scoped local state is simpler.
- Is the state server-authoritative? If yes, treat the client copy as a cache with an explicit invalidation strategy, regardless of where it lives.
- Does the team already have strong conventions for the global store? Convention is a real cost, but it does not change the memory or concurrency behavior of the model.

If several answers point toward local ownership, adopt it early. Retrofitting is possible but requires reworking the undo model and the render boundaries at the same time.

## FAQ

**Why does an AI chat UI re-render the whole conversation on every keystroke?**

Usually because a selector returns the entire conversation and the draft input writes into the same store. Narrow the selector, or move the draft into a store owned by the input component so only that component re-renders.

**How do I implement undo without a global history array?**

Store one operation record per reversible step, owned by the component that performed it, and keep an undo stack of references. Undo becomes a pop plus one inverse call, with cost independent of history length. Bound the stack depth.

**Is an event-driven approach slower than a global store?**

Dispatch overhead is typically small compared to render cost. The gain comes from shrinking the render surface: fewer components subscribe, so fewer re-render. Measure renders per interaction rather than assuming either model is faster.

**What is the smallest event bus that works?**

A map from event name to a set of handlers is enough, roughly the forty lines shown above. Add trace IDs, a dedupe window, and subscriber isolation. If you need durability or replay, pair it with a persistent log rather than growing the bus.

**How do I migrate an existing feature without a risky rewrite?**

Run both models behind a flag for the same user journey and compare p99 heap, renders per interaction, and duplicate side-effect counts. Migrate one surface at a time, starting with the one that re-renders most.

## Do this next

Pick your most-used AI surface and add the sampler above to it. Let one real session run to completion, then read the last `aiStateKB` value and the delta in `heapMB` between the first and last sample. If the delta is large and the AI state is owned by a component subtree, move that one piece of state local before your next deploy.
===END===
