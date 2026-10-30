# AI chat's hidden WCAG fails

Streaming AI chat interfaces break accessibility assumptions in ways static checklists do not catch. Automated scanners read a snapshot of the DOM; a chat interface rewrites that DOM many times per second. The result is a class of failures that pass CI and still leave screen reader users unable to follow a conversation.

This article covers three recurring failure modes — unstable ARIA roles, focus hijacking, and contrast drift in generated markdown — with the code that fixes each, a way to measure whether the fix worked, and a prevention checklist.

## Why static accessibility checks miss streaming UI

Most accessibility tooling evaluates a rendered page. It walks the accessibility tree, checks names and roles, and reports violations. That model assumes the page is roughly stable. A streaming chat response violates that assumption: tokens arrive incrementally, the DOM mutates on each chunk, and the accessibility tree is rebuilt repeatedly.

Three consequences follow.

First, the accessibility tree is only correct at the moment it is sampled. A container that starts as `role="status"` and later holds an error, a code block, and a multi-part answer is no longer accurately described by that role, but no scanner will flag it because the role attribute itself is valid.

Second, focus behavior depends on timing. Whether focus jumps depends on whether the user was at the bottom of the scroll container at the instant a chunk arrived. A static scan cannot evaluate that race.

Third, generated content is not styled by your design system unless you make it so. Markdown produced by a model arrives as text. Whatever colors it ends up with are decided by your renderer, not by the component library you assumed was in charge.

The practical implication: automated scans are necessary but insufficient. The failures below need runtime testing.

## Failure 1: ARIA roles that never change

The relevant requirement is that interactive components expose a name, role, and value, and that these stay accurate. The common implementation mistake is choosing one role for the message container and never revisiting it.

A chat message is not one thing. It can be a plain answer, a reasoning trace, an error with a stack trace, or a code block. These have different announcement semantics:

- A plain answer is an update worth announcing politely.
- A reasoning trace is a region a user may want to explore deliberately, not be interrupted by.
- An error should interrupt.
- A code block benefits from being a navigable region rather than an announcement.

If every one of these carries `role="status"` and `aria-live="polite"`, then errors are announced with the same urgency as ordinary text, and long reasoning traces are read aloud in full while the user is trying to do something else.

A workable pattern maps message type to role and politeness:

```javascript
function ChatMessage({ content, type = 'text' }) {
  const roleMap = {
    text: 'status',
    thought: 'region',
    error: 'alert',
    code: 'region'
  };

  return (
    <div
      role={roleMap[type]}
      aria-live={type === 'error' ? 'assertive' : 'polite'}
      aria-atomic="true"
    >
      {content}
    </div>
  );
}
```

Three details matter here.

`aria-atomic="true"` makes the assistive technology announce the whole container rather than only the changed fragment. Without it, a message that grows token by token is announced as a stream of fragments, which is unusable.

`aria-live="assertive"` on errors causes the announcement to interrupt whatever is being read. Use it sparingly; if everything is assertive, nothing is.

Mapping `thought` and `code` to `region` rather than a live role keeps them out of the announcement stream. The user can navigate to them deliberately.

### The busy-state problem

A second symptom of the same root cause: a screen reader reports the region as busy long after the response finished, or announces every token individually. Both come from a live region that updates faster than the assistive technology can settle.

One mitigation is to mark the region busy while streaming and debounce the transition back:

```javascript
const [isBusy, setIsBusy] = useState(false);

useEffect(() => {
  if (isStreaming) {
    setIsBusy(true);
    return;
  }
  const timer = setTimeout(() => setIsBusy(false), 300);
  return () => clearTimeout(timer);
}, [isStreaming]);

return (
  <div aria-busy={isBusy} aria-live={isBusy ? 'polite' : 'off'}>
    ...
  </div>
);
```

The 300 ms value is a starting point, not a standard. Tune it by measuring announcement latency (see the verification section). The important structural point is that the live region should not be asked to narrate every chunk.

### Why this is easy to miss

Automated scanners do not simulate streaming. They see a valid role on a valid element and report no violation. The only reliable detection is running a screen reader against a live response and listening to what happens.

## Failure 2: focus hijacking during updates

Focus management failures in chat interfaces usually trace to an effect that runs on every message change and moves focus to the newest message. The intent is helpful — "take the user to the new content" — but the effect fires for assistant messages too, so a user reading message three is yanked to message nine.

The problematic pattern:

```javascript
// Focus moves on every message change, including assistant messages
useEffect(() => {
  if (messages.length) {
    lastMessageRef.current?.focus();
  }
}, [messages]);
```

A corrected version only moves focus when the user is already at the end of the conversation, or when the new message is one the user sent:

```javascript
useEffect(() => {
  const isAtBottom =
    window.innerHeight + window.scrollY >= document.body.scrollHeight - 100;

  if (isAtBottom || isUserMessage(lastMessage)) {
    lastMessageRef.current?.focus({ preventScroll: true });
  }
}, [messages]);
```

`preventScroll: true` avoids the viewport jumping in a way that disorients users who rely on magnification or on a stable reading position.

The same logic in a component framework that uses reactive bindings instead of hooks:

```svelte
<script>
  let container;

  $: isAtBottom = container &&
    container.scrollTop + container.clientHeight >= container.scrollHeight - 100;

  $: if (isAtBottom || isUserMessage(lastMessage)) {
    container?.lastElementChild?.focus({ preventScroll: true });
  }
</script>

<div bind:this={container} class="chat-container">
  {#each messages as message}
    <div tabindex="0" class={message.role}>{message.content}</div>
  {/each}
</div>
```

Note the `tabindex="0"` on each message. Without it, the messages are not focusable at all, and keyboard navigation through a conversation is impossible — the user can reach the input and the send button and nothing else.

### Explicit arrow-key navigation

Making messages focusable is necessary but not sufficient. Users expect to move between messages with arrow keys, and browsers do not provide that behavior for arbitrary elements. It has to be implemented:

```javascript
const handleKeyDown = (e) => {
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    e.target.previousElementSibling?.focus();
  }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    e.target.nextElementSibling?.focus();
  }
};

return <div onKeyDown={handleKeyDown}>...</div>;
```

This matters more in chat than in a static document because messages are appended and removed. Without explicit handling, a user who has navigated to message four may find their position invalidated when the list re-renders.

A secondary trap: containers with `tabindex="-1"` that never release focus. If the chat container can receive focus programmatically but there is no way out, keyboard users become stuck. Providing an escape path is a small addition:

```javascript
useEffect(() => {
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      document.activeElement?.blur();
    }
  };
  document.addEventListener('keydown', handleKeyDown);
  return () => document.removeEventListener('keydown', handleKeyDown);
}, []);
```

## Failure 3: contrast drift in generated content

Text contrast requirements are well known: roughly 4.5:1 for normal text and 3:1 for large text. The failure mode specific to AI chat is that the text needing contrast is generated at runtime and styled by a renderer that may not consult the design system at all.

The sequence usually looks like this. The model returns a fenced code block. The markdown renderer applies its own styles, often inline or from a syntax highlighting theme, producing something like:

```css
pre code {
  color: #999;
  background: #f5f5f5;
}
```

Against a light background, that gray is well below the required ratio. The AI did not choose that color; the renderer did. But the renderer was configured once, for a documentation site, and nobody re-examined it when it was reused for chat output.

The fix is to route rendered content through design tokens rather than letting the renderer pick colors:

```javascript
import { theme } from '@your-design-system/tokens';

function CodeBlock({ code, language }) {
  return (
    <pre
      style={{
        background: theme.colors.background.code,
        color: theme.colors.text.code,
        border: `1px solid ${theme.colors.border.code}`,
      }}
    >
      <code className={`language-${language}`}>{code}</code>
    </pre>
  );
}
```

If the renderer accepts inline styles from the model's output, strip them before rendering. A transformation pass over the parsed markdown tree can enforce token-based colors:

```javascript
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import { visit } from 'unist-util-visit';

function enforceTokenColors(markdown) {
  const tree = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .parse(markdown);

  visit(tree, 'text', (node) => {
    node.data = {
      hProperties: {
        style: {
          color: theme.colors.text.code,
        },
      },
    };
  });

  return tree;
}
```

Tables deserve separate attention because models emit them often and markdown tables carry no styling of their own. If the renderer does not style `th` and `td`, the browser defaults apply, and defaults vary. Enforcing styles through CSS variables tied to design tokens keeps tables inside the same contrast budget as the rest of the interface:

```css
table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  padding: 0.5rem;
  text-align: left;
  border: 1px solid var(--color-border);
  color: var(--color-text);
  background: var(--color-background);
}

th {
  background: var(--color-background-subtle);
  font-weight: bold;
}
```

### A worked contrast calculation

Suppose a design token for code text is `#767676` on a background of `#ffffff`. To check it, convert each channel to a linear value, weight by luminance, and compute the ratio. For `#767676`, each channel is 118/255 ≈ 0.463. After the sRGB linearization step, each channel is approximately 0.181, giving a relative luminance of about 0.181. White has luminance 1.0. The ratio is (1.0 + 0.05) / (0.181 + 0.05) ≈ 4.54. That passes 4.5:1 for normal text, but only just — any darkening of the background or lightening of the text pushes it under.

This is the argument for choosing code colors with margin rather than aiming at the threshold. A token pair landing near 7:1 survives theme changes, hover states, and syntax highlighting variants. A pair at 4.6:1 does not.

## Verification: what to instrument and what to compare

Verification here means running the interface the way a user would and measuring specific things. Four checks, in increasing order of effort.

**1. Automated scan on the rendered interface.** Run an accessibility scanner against the chat page in a browser automation harness. This catches missing labels, invalid attribute combinations, and static contrast problems. It will not catch anything that depends on streaming. Treat a clean scan as a floor, not a result.

**2. Manual screen reader pass with a live response.** Send a prompt that produces a long answer containing a code block, then a second prompt that produces an error. Listen for: whether the answer is announced once or in fragments; whether the error interrupts; whether navigating away mid-response is possible; whether the region reports busy after the response completes. This is the only check that exercises the actual failure modes described above.

**3. Announcement latency instrumentation.** If announcements feel slow or fragmented, measure rather than guess. Mark timestamps around the live-region update and record the delta:

```javascript
performance.mark('announcement-start');
// update the ARIA region
performance.mark('announcement-end');

const latency = performance.measure(
  'announcement-latency',
  'announcement-start',
  'announcement-end'
).duration;
```

Compare percentiles across streaming and non-streaming responses. A large gap between the two indicates the live region is being updated too frequently, which points back to the debounce or busy-state fix.

**4. Contrast audit of rendered output, not source.** Extract computed colors from the live DOM for code blocks, tables, and secondary text, and compute ratios from those values. Auditing the stylesheet is unreliable because generated content may be styled by a renderer you did not write.

**5. Regression tests for dynamic behavior.** A browser automation test can assert that the correct role appears for each message type and that focus does not move when a user has scrolled up:

```javascript
import { test, expect } from '@playwright/test';

test('assistant message uses a live region', async ({ page }) => {
  await page.goto('/chat');
  await page.locator('input').fill('Hello');
  await page.keyboard.press('Enter');

  await expect(page.locator('[role="status"]')).toContainText('Hello');
});

test('focus is not stolen when the user has scrolled up', async ({ page }) => {
  await page.goto('/chat');
  // populate several messages, scroll to the top, then trigger a response
  // assert the focused element is unchanged
});
```

The second test is the one that catches focus hijacking. It is also the one most teams omit, because it requires setting up a scrolled state rather than asserting on a single render.

## Prevention checklist

Accessibility in a streaming interface is a design constraint, not a post-launch audit. The following are worth establishing before the next chat feature ships.

**Constrain model output shape.** Ask the model to delimit reasoning, answers, and errors with explicit markers. Parsing structured output into message types is far more reliable than inferring type from prose:

```
Respond using this structure:
<response>
  <thought>internal reasoning</thought>
  <answer>final answer</answer>
</response>

On failure:
<error>
  <message>description</message>
  <stack>stack trace</stack>
</error>
```

**Decide the role mapping before writing components.** For each message type, write down the role, the live-region politeness, and whether the content should be announced at all. This is a five-line table that prevents most of failure 1.

**Make generated content inherit tokens.** Any renderer that handles model output should take its colors from the design system. If it accepts inline styles, strip them.

**Review focus behavior as a feature, not a side effect.** Auto-scroll and auto-focus are conveniences for one user and obstructions for another. The "only when at the bottom" rule is a reasonable default.

**Add dynamic assertions to CI.** Static scans plus at least one test that scrolls up and asserts focus stability. Keep the static scan; just do not rely on it alone.

**Test with a screen reader before release.** One manual pass with a live streaming response catches failures that no scanner reports.

## FAQ

**Does `aria-live` on the message container handle streaming automatically?**
No. A live region announces changes, but a region updating many times per second produces fragmented or unusable output. Streaming needs either a debounce, a busy state, or a design where the live region is updated once per complete message.

**Is `role="alert"` appropriate for all errors?**
No. `alert` implies an assertive announcement that interrupts. Reserve it for errors the user must know about immediately. Validation hints and recoverable warnings are better handled politely.

**Why not just make every message focusable and let the browser handle navigation?**
Focusability and navigation are separate. `tabindex="0"` makes an element reachable by Tab, but arrow-key movement between messages requires explicit key handling.

**Can a component library guarantee contrast in generated content?**
Only if the generated content is rendered through that library. Raw markdown passed to a generic renderer bypasses the library's tokens entirely. The renderer is where the guarantee has to be enforced.

**How much does this cost in engineering time?**
The changes themselves are small — role mapping, a focus condition, token-based styling. The cost is in establishing the test path: a screen reader pass and a dynamic focus assertion in CI. That setup is what prevents the same class of bug from recurring.

## Do this in the next 30 minutes

Open your chat interface, send a prompt that produces a long response containing a code block, and start a screen reader before the response begins. Listen for three things: whether the response is announced in fragments or as a unit, whether any error you trigger interrupts the current announcement, and whether you can navigate to an earlier message while the response is still streaming. Write down what you observe. Those three observations tell you which of the three failures above you have, and each fix is a small, localized change.
