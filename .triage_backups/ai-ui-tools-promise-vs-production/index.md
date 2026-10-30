# AI UI tools: promise vs. production

AI UI tools such as Cursor, Figma's AI features and GitHub Copilot Workspace can produce a working React component from a plain-English prompt in seconds. The output usually renders. It often looks close to the design. And it frequently fails in a small number of predictable ways that only surface once the component meets a real design system, a real bundle budget and a real accessibility audit.

This article is about those failure modes, how to detect them cheaply, and how to structure prompts and review so that the generated code is a reviewable diff rather than a rewrite.

## Why the demo and production diverge

A demo prompt usually describes one component in isolation: a card, a button, a modal. Production code lives inside a design system with a fixed token vocabulary, a set of responsive breakpoints, i18n strings, RTL requirements, and a bundle budget enforced in CI. The gap between those two contexts is where AI UI output breaks.

Three failure modes account for most of the cleanup work:

1. **Token drift.** The model has seen thousands of Tailwind-style class names and invents plausible ones that do not exist in your token file. A component can reference `bg-surface-100` when your theme only defines `bg-surface-1`, and nothing will error until the class silently does nothing at runtime.
2. **Inline style accumulation.** Generated components frequently fall back to `style={{...}}` objects for anything the model is unsure about. These bypass your token system, cannot be themed, and add to the JS bundle rather than the CSS bundle.
3. **Missing interaction and accessibility behavior.** Focus management, `aria-live` regions for time-based content, keyboard traps in modals, and reduced-motion handling are rarely present unless explicitly requested.

None of these are exotic. They are the same issues a junior developer produces on a first pass, which is a useful mental model.

## The mental model: a compiler of intent

Treat the AI tool as a fast junior developer who has read every React and Tailwind doc but has never shipped to your codebase. The junior writes the happy path quickly. A senior still needs to:

- check the output against the token vocabulary,
- remove inline styles that bypass the theme,
- add the accessibility behavior the prompt did not specify,
- verify the bundle cost of any new dependency.

The productive framing is to treat the tool as a **compiler of intent** rather than a generator of truth. You supply the constraints — token names, breakpoints, bundle limits, accessibility rules — and the tool produces a candidate diff. The quality of the output is a direct function of how completely those constraints are stated.

## Worked example: a user card

Consider a component that shows a name, an avatar and a relative last-login time.

**Prompt**

```text
Build a UserCard component in React with TypeScript.

Requirements:
- Use only these design tokens: bg-surface-100, text-text-900, text-text-500
- Support RTL and LTR without extra props
- Avatar 48x48 with fallback initials
- Last login relative time that updates every minute
- TailwindCSS utility classes only, no inline style objects
- No new dependencies beyond date-fns
```

**A plausible generated result**

```tsx
// filename: UserCard.tsx
import { useEffect, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';

export const UserCard = ({ user }: { user: { id: string; name: string; avatar?: string; lastLogin?: string } }) => {
  const [timeAgo, setTimeAgo] = useState('');

  useEffect(() => {
    if (!user.lastLogin) return;
    const update = () => setTimeAgo(formatDistanceToNow(new Date(user.lastLogin!)));
    update();
    const id = setInterval(update, 60_000);
    return () => clearInterval(id);
  }, [user.lastLogin]);

  return (
    <div className="p-4 rounded-lg bg-surface-100 flex items-center gap-4">
      <img
        src={user.avatar || `https://avatar.vercel.sh/${user.id}.svg`}
        alt={user.name}
        width={48}
        height={48}
        className="rounded-full"
      />
      <div>
        <h3 className="text-text-900 font-medium">{user.name}</h3>
        <p className="text-text-500 text-xs">Last active {timeAgo} ago</p>
      </div>
    </div>
  );
};
```

**Reviewing the diff**

Walk through it against the constraints rather than against a feeling that it "looks right":

- **Token fidelity.** The classes `bg-surface-100`, `text-text-900` and `text-text-500` must be checked against the actual theme file. Grep the token source for each class name. Any class that does not appear is a silent no-op.
- **Bundle cost.** Adding `date-fns` as a dependency is a real cost. Measure it rather than guessing: run your existing bundle-size tooling against the built component and compare to the same component with a hand-written formatter. If the library is only used for one relative-time string, `Intl.RelativeTimeFormat` is built into the platform and adds nothing.
- **RTL behavior.** Flex layouts, `gap` and logical properties are generally direction-agnostic. What is not direction-agnostic is any `ml-*`/`mr-*` or `pl-*`/`pr-*` class, or an explicit `dir` attribute. Check for those.
- **Accessibility.** The `alt` text is present, which is good. The relative time string updates every minute, which means screen readers will not announce the change unless the element is a live region. Add `aria-live="polite"`:

```tsx
<p aria-live="polite" className="text-text-500 text-xs">
  Last active {timeAgo} ago
</p>
```

- **Timer cleanup.** The `useEffect` clears its interval, which is correct. A common generated variant omits the cleanup function, which leaks a timer per mount.

The point of the review is not that the component is bad. It is that the review is a checklist, not a judgement call, and a checklist takes minutes.

## How to measure the things that matter

Claims about AI-generated UI being "faster" or "smaller" are only meaningful if you can measure them in your own repository. The instrumentation is not exotic.

**Bundle impact.** Use whatever size tooling your project already has — `size-limit`, `bundlesize`, `webpack-bundle-analyzer`, or the size reporting built into your framework's build output. The procedure is:

1. Build the component in isolation or as part of a small entry point.
2. Record the gzipped size of the chunk that contains it.
3. Remove the component and rebuild.
4. The difference is the marginal cost.

Run this for the generated version and for a hand-written version using platform APIs. The comparison is what matters, not an absolute number.

**Render cost.** Use your browser's performance profiler or the React DevTools profiler. Record mount time on a throttled CPU profile (4x or 6x slowdown approximates a low-end device). Compare the generated component against the hand-written one. Inline style objects that are recreated on every render show up as repeated style recalculation in the profiler.

**Accessibility.** Run axe-core or an equivalent rule engine against the rendered component in a test environment. This catches missing labels, contrast failures, and — depending on the rule set — live-region issues. It does not catch focus management, which needs a manual keyboard pass or a dedicated test.

**Token drift.** This one is cheap to automate: extract every class name from the generated file, and check each against the token source. A regex over `className` strings plus a lookup is enough for most codebases.

## Common misconceptions

**"AI tools remove the need for designers."** They remove the need for designers to hand-write markup. They do not make design decisions. Whether a button is primary or secondary is a product decision that has to exist before the prompt is written.

**"Generated code is faster than hand-written code."** This is not a general truth. Generated code is faster to *produce*. Its runtime and bundle characteristics depend entirely on what the model chose to emit. A component that pulls in a date library for one string is slower and heavier than one that uses `Intl.RelativeTimeFormat`. Measure per component; do not assume.

**"Unit tests are unnecessary for generated components."** The opposite is true. Generated components encode assumptions that are invisible in the diff — that the first focusable element receives focus, that a timer is cleaned up, that a class name exists. Tests are how those assumptions become visible. A modal that renders correctly but never moves focus is a real and common failure.

**"The tool understands my design system."** It understands the literal strings you give it. It does not understand hierarchy or naming conventions. If your theme contains both `--color-primary-500` and `--color-primary-surface-500`, a model asked for a "primary surface" may invent a third name that matches neither. Paste the exact token list into the prompt.

## A post-processing pipeline

The reliable way to keep review time low is to move the mechanical checks out of human review and into a script that runs before the diff is opened.

A minimal version, using only Node built-ins plus your existing size tooling:

```javascript
// filename: ai-review.mjs
import { execSync } from 'child_process';
import { readFileSync } from 'fs';

const files = process.argv.slice(2);
const ALLOWED_TOKENS = ['bg-surface-100', 'text-text-900', 'text-text-500'];

for (const file of files) {
  const src = readFileSync(file, 'utf8');

  // 1. Inline style objects, which bypass the token system
  const inlineStyles = src.match(/style=\{\{[^}]*\}\}/g) || [];
  if (inlineStyles.length) {
    console.error(`Inline style objects in ${file}: ${inlineStyles.length}`);
  }

  // 2. Class names that are not in the allowed token list
  const classAttrs = src.match(/className="([^"]*)"/g) || [];
  const used = classAttrs
    .flatMap((c) => c.replace(/className="|"/g, '').split(/\s+/))
    .filter(Boolean);
  const unknown = used.filter((cls) => !ALLOWED_TOKENS.includes(cls));
  if (unknown.length) {
    console.error(`Unrecognised classes in ${file}: ${unknown.join(', ')}`);
  }

  // 3. Bundle size, using whatever size tool the project already has
  try {
    const out = execSync(`npx size-limit --json ${file}`, { encoding: 'utf8' });
    const kb = JSON.parse(out).bundleSize / 1024;
    if (kb > 2) console.error(`Bundle ${kb.toFixed(1)} KB exceeds 2 KB in ${file}`);
  } catch {
    // size-limit exits non-zero when the limit is exceeded
  }
}
```

The token list and the size limit are project-specific. The value of the script is that it fails loudly before a human looks at the diff, so the human review is about behavior rather than about class names.

Run it as a pre-commit hook or a CI step:

```bash
node ai-review.mjs src/components/UserCard.tsx src/components/Modal.tsx
```

## A prompt template worth reusing

Constraints stated once in a template save restating them in every prompt.

```markdown
# Component Prompt Template

## Constraints
- Use only tokens listed in /tokens.json (paste the list)
- Tailwind utility classes only; no inline style objects
- Breakpoints: sm=640, md=768, lg=1024
- Logical properties only (no ml-/mr-/pl-/pr-)
- No new dependencies without explicit approval

## Required behavior
- Keyboard focus order documented in a comment
- Live regions for any content that updates on a timer
- Cleanup for every timer, listener and subscription

## Output
- TypeScript, one component per file
- Storybook story with controls for all props
```

## Decision checklist before merging generated UI

- [ ] Every class name exists in the token source.
- [ ] No inline style objects remain.
- [ ] Every timer, listener and subscription has a cleanup path.
- [ ] Keyboard focus is correct on mount and on open/close for overlays.
- [ ] Time-based or async content sits in a live region where appropriate.
- [ ] Bundle delta measured against the pre-change build.
- [ ] axe-core (or equivalent) passes with no new violations.
- [ ] Any new dependency is justified by a platform API that would not do the job.

## FAQ

**Why do generated components use class names that do not exist?**

The model predicts plausible names from training data rather than reading your theme. Pasting the exact token list into the prompt is the most reliable fix, followed by the class-name check in the review script.

**Do these tools work with Vue or Svelte?**

Support varies by tool and changes quickly. Check the current documentation for the specific tool rather than relying on a general claim.

**How do I stop inline styles from reaching the bundle?**

State the constraint in the prompt, then enforce it with a script. Prompt-only enforcement drifts; script enforcement does not.

**What is the most common accessibility failure in generated UI?**

Focus management in overlays. A modal that renders and closes correctly but never moves focus into the dialog is a recurring pattern, and it is not detected by most automated rule engines.

## What to do in the next 30 minutes

Pick one component that was generated in the last week and run your project's size tooling against it:

```bash
npx size-limit --json src/components/Button.tsx
```

If the reported size is above your budget, or if `grep -n 'style={{' src/components/Button.tsx` returns any matches, rewrite that one component against your token list and platform APIs, then commit the diff. One component is enough to tell you whether the rest of the generated code needs the same treatment.
