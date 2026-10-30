# AI UI generation: what it fixed and what still breaks

AI UI generation tools take a natural-language description of an interface and return a component file, usually React plus a utility CSS framework such as Tailwind. The output is often syntactically valid, visually plausible, and wrong in ways that only appear under specific browser, network, or data conditions.

The useful framing is not "can AI build my UI" but "which parts of UI work are mechanical translation, and which parts require knowledge the model does not have." Mechanical translation is where these tools earn their keep. Everything that depends on private context — your auth flow, your data shapes, your browser support matrix, your accessibility bar — is where they quietly fail.

## What the tools actually do

A generator like v0 by Vercel takes a prompt and emits a component tree, styling classes, and some state logic. Design-to-code tools in the same category convert a design file into framework code. Both are constrained autocomplete over a large corpus of public frontend code. They are not reasoning about your product.

What that means in practice:

- They are strong at patterns that appear thousands of times in public repositories: card grids, form layouts, navigation bars, modal shells, table markup, hover and focus states.
- They are weak at anything where the correct answer depends on information not in the prompt: your API contract, your design tokens, your supported browsers, your performance budget, your test conventions.
- They do not verify their output. A generated component that compiles and renders is not evidence that it behaves correctly.

The mental model that holds up: treat the tool as a fast producer of a first draft that a competent frontend engineer must review line by line. The review is not optional overhead; it is the part of the process where correctness is established.

## Where generation genuinely helps

Three categories of work are reliably faster with a generator.

**Boilerplate markup.** A responsive card grid with sensible breakpoints, a settings page with a form and labels wired to inputs, a data table with sortable headers — these are translation tasks from a description to known idioms. A generator produces them in seconds, and the review is mostly visual.

**Style consistency at volume.** If your design tokens are already encoded as CSS variables or a Tailwind theme, a generator can produce a set of buttons, inputs, and cards that reference those tokens. The value is not the individual component; it is that twenty components share the same spacing scale and color references without a human copy-pasting values.

**Refactoring known patterns.** Converting a class component to a function component, replacing inline styles with utility classes, or splitting a large file into smaller ones are mechanical transformations. The tool does not need product knowledge to do them, only pattern knowledge.

## Where it breaks, and why

The failure modes cluster around missing context. Each of the following is a category, not an anecdote.

**Platform behavior differences.** Generated fetch code commonly assumes one browser's defaults. A request written with `credentials: 'include'` behaves differently across browsers and storage modes, and a generator has no way to know which of your users are affected. The same class of problem appears with WebSocket reconnection, service worker cache versioning, and pointer versus mouse events on hybrid devices. The code is correct for the environment the model saw most often in training data and untested everywhere else.

**Resource lifecycle.** Generated components frequently omit cleanup. A chart that creates a canvas context, an animation loop started with `requestAnimationFrame`, or a formatter object constructed per render can all leak. The model produces the happy path — draw the thing — and omits the teardown, because teardown is invisible in a demo.

**Architecture.** Ask a generator for a full application with authentication, a database schema, and routing, and you get a single large file with a hardcoded schema that does not match your actual database. The tool has no access to your schema, so it invents one. This is not a bug that will be fixed by a better model; it is a consequence of the model not having your private context.

**Accessibility depth.** A generator will add `aria-modal`, `role="dialog"`, and an escape handler to a modal. It will usually omit a correct focus trap, focus restoration to the trigger element on close, and inert handling for background content. These are the parts that determine whether the component is actually usable with a keyboard or screen reader.

**Test coverage.** Generators are poor at writing tests that assert meaningful behavior. They will produce a test file that renders a component and checks that text appears, which passes regardless of whether the logic is correct.

## A worked review: the fetch bug

Consider a generated submit handler. The prompt described a transaction table with a submit action; the output looked like this:

```javascript
// Generated: works in some browsers, fails in others
const response = await fetch('/api/transactions', {
  method: 'POST',
  credentials: 'include',
  body: JSON.stringify(data),
});
```

Two problems are visible to a reviewer and invisible to the generator.

First, `credentials: 'include'` sends cookies cross-origin. Whether that succeeds depends on the server's CORS configuration and on the browser's handling of third-party storage. In a same-origin deployment the option is unnecessary; in a cross-origin one it requires `Access-Control-Allow-Credentials` and a non-wildcard `Access-Control-Allow-Origin`. The generated code asserts an assumption about deployment topology that was never stated.

Second, there is no `Content-Type` header. Without it, the request body may be sent without the JSON content type, and servers that parse based on content type will reject or misparse it.

The corrected version makes the same-origin case explicit:

```javascript
const response = await fetch('/api/transactions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
});
```

If the request genuinely must be cross-origin with cookies, the fix is not to remove `credentials` but to add the header and confirm the server's CORS policy allows the origin with credentials. The point is that the reviewer must decide, because the generator could not.

### How to measure whether generation is paying off

Do not rely on impressions. Instrument the work.

1. Track time-to-first-working-component for a fixed set of tasks, recorded in your issue tracker's time fields rather than estimated afterward.
2. Count review comments per generated file versus per hand-written file of similar size. A high ratio means the tool is shifting work from typing to reviewing, not removing it.
3. Count defects found after merge that trace to generated code, tagged consistently.
4. Measure bundle-size delta for generated versus hand-written equivalents using your existing bundle analyzer.

If review time plus defect cost exceeds the typing time saved, the tool is not helping on that task category. That is a per-category finding, not a verdict on the tool.

## A decision checklist before you merge generated code

Run every generated component through this list. It is short because it targets the failure modes above.

- **Data contract:** Does the component call an endpoint that exists, with the method, path, and body shape your API actually accepts?
- **Environment assumptions:** Does it assume a browser feature, storage mode, or origin relationship that your support matrix does not guarantee?
- **Lifecycle:** Does every subscription, timer, animation frame, and constructed object have a corresponding teardown?
- **Accessibility:** Can the component be operated with a keyboard alone? Is focus managed on open and close? Does it pass an automated audit such as axe-core?
- **State ownership:** Does the component introduce local state that duplicates or conflicts with your existing store?
- **Tests:** Is there a test that would fail if the core logic were inverted? If not, write one before merging.

## Using generators to enforce consistency

Once basic generation is understood, the higher-value use is constraining output to your design system.

Generate a component set from your tokens rather than from a visual description. A prompt that names the token values produces components that reference them:

> Generate a React component library using Tailwind and these design tokens: primary #0066ff, secondary #ff6600, neutral #333. Include button variants for primary, secondary, and neutral, a text input, a select, and a card using shadow-md and rounded-lg.

The output still needs review, but the spacing and color references will be consistent across every component in the set, which is the property you wanted.

For refactoring, describe behavior rather than appearance:

> This component displays a user profile card with name, avatar, email, and an edit button. It uses a legacy context API and inline styles. Rewrite it to use Tailwind utility classes and a modern folder structure, preserving the existing prop interface.

Naming the prop interface as a constraint is what keeps the refactor from silently changing the component's contract.

For accessibility, treat generation as scaffolding only:

> Generate a modal dialog component following WAI-ARIA practices, including a focus trap, escape-to-close, and focus restoration to the trigger on close.

Then verify with a keyboard and a screen reader. Automated tools catch roughly the mechanical subset of issues; they do not catch a focus trap that traps focus in the wrong subtree.

## Tests: write them yourself

A generator can produce the component; the test that proves it works is your job. Given a login form, a useful test asserts observable behavior under failure:

```javascript
// Test scaffolding — write and review this yourself
import { render, screen, fireEvent } from '@testing-library/react';
import LoginForm from './LoginForm';

describe('LoginForm', () => {
  it('shows an error when credentials are rejected', async () => {
    global.fetch = jest.fn(() => Promise.resolve({ ok: false }));
    render(<LoginForm />);
    fireEvent.click(screen.getByText('Login'));
    expect(await screen.findByText('Invalid credentials')).toBeInTheDocument();
  });
});
```

The component under test is a simple generated form:

```javascript
const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw new Error('Login failed');
    } catch (err) {
      setError('Invalid credentials');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <button type="submit">Login</button>
      {error && <p>{error}</p>}
    </form>
  );
};
```

Note what the test does not cover: it does not verify that the request body is correct, that the content type is set, or that the form is usable by keyboard. Those assertions should be added deliberately.

## Comparison: generation versus hand-written components

| Dimension | Generated first draft | Hand-written |
|---|---|---|
| Time to first render | Minutes for described patterns | Hours for the same markup |
| Design token adherence | Only if tokens are named in the prompt | By construction |
| Platform edge cases | Usually unhandled | Handled if the author knows them |
| Accessibility depth | Structural attributes present, behavior often incomplete | Depends on author diligence |
| Test coverage | Minimal, rarely asserts logic | Written to the required bar |
| Review burden | High per file, front-loaded | Lower per file, spread across authoring |
| Best fit | Boilerplate, style-consistent component sets, mechanical refactors | Architecture, data flow, anything with private context |

## FAQ

**Why does generated code work in one browser and fail in another?**

Because it encodes assumptions the prompt did not state. Cross-origin credentials, storage availability in private modes, WebSocket idle timeouts, and pointer event normalization all differ across browsers and device classes. Test the browsers in your support matrix, including private or restricted modes where relevant.

**Can these tools replace designers?**

No. They reproduce layout and styling patterns; they do not make hierarchy, spacing, or interaction decisions against a brand and user context. Design review remains necessary.

**Should generated code be trusted if it compiles and renders?**

No. Compilation proves syntax, not behavior. The review checklist above exists precisely because the failure modes are invisible until a specific environment or data condition is present.

**How should generated code be audited?**

In order: verify the data contract against your API, check environment assumptions against your support matrix, inspect lifecycle cleanup, run an automated accessibility audit and a keyboard pass, then confirm test coverage asserts the logic rather than just the rendering.

**Is generation worth it for a small number of components?**

For occasional one-off components the setup and review overhead can exceed the typing saved. The value shows up when the same patterns repeat, or when consistency across many components matters more than any single one.

## Do this in the next 30 minutes

Pick one generated component currently in your codebase and run it against the decision checklist above. Start with the lifecycle question — search the file for `fetch`, `setInterval`, `setTimeout`, `requestAnimationFrame`, and any constructor call, and confirm each has a matching teardown. That single pass finds the class of defect that survives code review and appears in production.
