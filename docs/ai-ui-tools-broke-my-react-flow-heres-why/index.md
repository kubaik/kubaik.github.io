# AI UI Generators: Production Constraints and Fixes

AI UI generation tools accept a prompt and return component code. The output is often visually plausible and often fails specific, predictable checks: keyboard access, breakpoint behavior, state ownership, and design-token compliance. This article covers why that happens, how to evaluate the output, and how to structure a workflow where the generated draft is a starting point rather than a finished component.

## The short version

Tools in this category — chat-based code generators, editor-integrated assistants, and design-tool export plugins — are good at producing a first draft of presentational markup. They are not good at inferring constraints you did not state: your breakpoints, your token names, your state ownership model, your accessibility bar, or your bundle budget.

A reasonable working assumption: budget refactor time per component roughly comparable to the time you would spend writing the component's behavior by hand, and treat the generated markup as free. Whether that trade is worth it depends on how much of your component is markup versus behavior.

## Why the mismatch exists

The confusion is structural, not a bug in any particular tool.

**Generation is optimized for plausibility, not for your constraints.** A language model produces the most likely completion given the prompt. The most likely React table is a desktop-first table with local `useState` and an array index as key. That is a reasonable default in a vacuum and wrong in most applications.

**Design tools and code have different models.** A design file describes appearance. A component describes behavior under state, at multiple viewports, with real data. Export tooling can only translate what the design file contains. If the design file has no focus states, no error states, no loading states, and no narrow-viewport layout, the export cannot invent them.

**The tools have no access to your repository.** Unless you supply them explicitly, a generator does not know your token names, your lint rules, your component library, or your data-fetching conventions. It will produce valid CSS that violates your design system, because validity and compliance are different properties.

**Local state is the path of least resistance.** Generated components tend to own their own state. That is fine for a self-contained widget and wrong for anything that must coordinate with a URL, a cache, or a global store.

## A worked example: a sortable, paginated table

Take a representative prompt: "Create a responsive React table with sorting and pagination." A typical generated result looks like this.

```tsx
import { useState } from "react";

export default function Table({ data }) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: "ascending" });
  const [currentPage, setCurrentPage] = useState(1);

  const sortedData = [...data].sort((a, b) => {
    if (sortConfig.key) {
      return a[sortConfig.key] > b[sortConfig.key]
        ? sortConfig.direction === "ascending" ? 1 : -1
        : sortConfig.direction === "ascending" ? -1 : 1;
    }
    return 0;
  });

  const pageSize = 10;
  const totalPages = Math.ceil(sortedData.length / pageSize);
  const paginatedData = sortedData.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const requestSort = (key) => {
    let direction = "ascending";
    if (sortConfig.key === key && sortConfig.direction === "ascending") {
      direction = "descending";
    }
    setSortConfig({ key, direction });
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white border">
        <thead>
          <tr>
            <th onClick={() => requestSort("name")}>Name</th>
            <th onClick={() => requestSort("email")}>Email</th>
            <th onClick={() => requestSort("status")}>Status</th>
          </tr>
        </thead>
        <tbody>
          {paginatedData.map((row, i) => (
            <tr key={i}>
              <td>{row.name}</td>
              <td>{row.email}</td>
              <td>{row.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex justify-between mt-4">
        <button onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}>Previous</button>
        <span>Page {currentPage} of {totalPages}</span>
        <button onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}>Next</button>
      </div>
    </div>
  );
}
```

### Failure-mode analysis

Read the code against a checklist rather than against your eyes.

**Keyboard access.** The sortable headers are `<th onClick>`. They are not focusable, have no `tabIndex`, no `role="button"`, and no keyboard handler. A keyboard user cannot sort the table at all. This is the single most common defect in generated table code.

**Sort state and screen readers.** There is no `aria-sort` on the active column, so assistive technology cannot announce the current sort direction.

**React keys.** `key={i}` uses the array index. When the underlying data is reordered, filtered, or paginated from a server, React reuses DOM nodes by position and component state attaches to the wrong row. Any row-level state — an expanded row, an inline edit — will jump to the wrong record. Keys must be stable identifiers.

**Sorting correctness.** The comparator uses `>` on raw values. For strings this is case-sensitive and locale-insensitive; for numbers passed as strings it is lexicographic, so `"10" < "9"`. For dates it compares whatever the values happen to be. It also mutates nothing (`[...data]` is a shallow copy, good) but re-sorts on every render with no memoization.

**Responsiveness.** `overflow-x-auto` makes the table scroll horizontally. That is a legitimate strategy, but it is not the only one and it is not stated as a requirement. On a narrow viewport the table will scroll sideways, which is often worse than a stacked card layout for the same data. The generator picked one option because the prompt did not constrain it.

**Design tokens.** `bg-white` and the Tailwind spacing utilities are hard-coded. If your design system expresses surfaces as a semantic token, this component bypasses it and will not respond to theming.

**State ownership.** Page and sort live in component state. If the URL should carry them — so a link is shareable and the back button works — this is the wrong model. If a server cache should own the data, the component should not be slicing a full array client-side at all.

**Data volume.** `data` is assumed to be the entire dataset, present in memory. For a table of 50 rows that is fine. For 50,000 it is not, and the fix is server-side pagination, which changes the component's shape entirely.

### A corrected version

The prompt must state the constraints. A prompt that includes them looks like this:

```
Create a React table component.
Columns: name (string), email (string), status ('active' | 'inactive').
Row identity: each row has a unique string `id`; use it as the React key.
Sorting: clicking a column header toggles asc/desc. Headers must be
  keyboard-focusable and activate on Enter and Space. Set aria-sort on
  the sorted column to 'ascending' or 'descending'.
Pagination: server-side. The component receives `page`, `pageSize`,
  `total`, and calls `onPageChange(nextPage)`.
Styling: use the design system's semantic surface and text tokens;
  do not hard-code colors.
Narrow viewport: below 640px, render each row as a stacked block with
  a visible label per field instead of a horizontally scrolling table.
```

A corrected implementation of the parts that matter:

```tsx
import { useMemo } from "react";

type Row = { id: string; name: string; email: string; status: "active" | "inactive" };

type Props = {
  rows: Row[];
  sortBy: keyof Row | null;
  sortDirection: "asc" | "desc";
  onSortChange: (key: keyof Row, direction: "asc" | "desc") => void;
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
};

const columns: { key: keyof Row; label: string }[] = [
  { key: "name", label: "Name" },
  { key: "email", label: "Email" },
  { key: "status", label: "Status" },
];

export function DataTable({
  rows, sortBy, sortDirection, onSortChange,
  page, pageSize, total, onPageChange,
}: Props) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const handleSort = (key: keyof Row) => {
    const next = sortBy === key && sortDirection === "asc" ? "desc" : "asc";
    onSortChange(key, next);
  };

  const handleHeaderKeyDown = (e: React.KeyboardEvent, key: keyof Row) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleSort(key);
    }
  };

  const ariaSortFor = (key: keyof Row) =>
    sortBy === key ? (sortDirection === "asc" ? "ascending" : "descending") : "none";

  return (
    <div>
      <table role="grid" aria-rowcount={total} aria-label="Records">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                role="columnheader"
                aria-sort={ariaSortFor(col.key)}
                tabIndex={0}
                onKeyDown={(e) => handleHeaderKeyDown(e, col.key)}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.name}</td>
              <td>{row.email}</td>
              <td>{row.status}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <nav aria-label="Pagination">
        <button disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          Previous
        </button>
        <span>Page {page} of {totalPages}</span>
        <button disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
          Next
        </button>
      </nav>
    </div>
  );
}
```

What changed and why:

- Sort state moved out of the component. The parent owns it, so it can live in the URL or a store.
- Pagination is server-driven. The component receives a page of rows and a total, so it never needs the full dataset in memory.
- Headers are focusable and respond to Enter and Space, and `aria-sort` communicates direction.
- Keys are stable row identifiers.
- The comparator is gone. Sorting is the data layer's job, and doing it there means it can be done in SQL or an index rather than in the browser.

Note that this component is now mostly presentational. That is the point: the generator's strength is markup, so the refactor should move behavior out of the generated code and into code you own.

## How to measure whether it is worth it

Do not rely on impressions. Instrument the workflow.

**Measure refactor time per component.** Track the wall-clock time from pasting generated code to merging the component. Log it in your issue tracker against the component. After ten components you will have a real distribution rather than an anecdote.

**Measure defect escape.** Count how many generated components required a post-merge fix for accessibility, responsive layout, or state bugs. Compare that against hand-written components over the same period. If the generated set has a materially higher rate, the refactor step is not catching enough.

**Run the accessibility audit in CI, not by hand.** Tools such as axe-core can be run against rendered components in a test environment. Wire it into your test command so a violation fails the build. Manual audits do not scale and are skipped under deadline pressure.

**Measure the bundle delta.** If you are generating components with a CSS-in-JS runtime, compare the production bundle size before and after with your bundler's analyzer. A component that adds a runtime dependency to save twenty minutes of typing is usually a bad trade on a performance-sensitive page.

**Measure the narrow-viewport behavior.** Load the component at your smallest supported width and check for horizontal overflow. In a browser, the document element's `scrollWidth` exceeding its `clientWidth` indicates horizontal overflow; that is a mechanical check you can automate.

**Measure the interactivity cost.** Run your production build through Lighthouse or your own performance budget check and compare the interaction metrics for pages that use generated components against pages that do not.

## A decision checklist

Before generating a component, decide:

1. **Is it mostly markup or mostly behavior?** Pricing sections, empty states, and static cards are mostly markup and generate well. Data tables, comboboxes, and anything with a focus trap are mostly behavior and generate poorly.
2. **Does it need to be keyboard accessible?** If yes, plan to rewrite the interactive elements regardless of what the generator returns.
3. **Who owns its state?** If the answer is anything other than "the component itself," plan to lift the state out.
4. **Does it need to work at multiple viewports?** If yes, write the narrow-viewport layout yourself. Generators rarely infer it.
5. **Is it on a performance-critical path?** If yes, check the bundle cost before adopting a generated implementation.
6. **Will it be reused?** If yes, it belongs in your component library with your tokens, not as a one-off in a feature folder.

If a component scores badly on several of these, generating it is likely to cost more than writing it.

## Common misconceptions

**"The output is production-ready."** The output is plausible. Production-ready additionally means keyboard accessible, responsive at your breakpoints, integrated with your state model, and compliant with your tokens. Those are separate properties and the generator does not check them.

**"It saves time overall."** It saves time on the first draft. Whether it saves time overall depends on the refactor cost, which depends on the component. The only way to know for your codebase is to measure it, as above.

**"It understands my design system."** It understands the visual appearance you described or showed it. It does not know your token names or your lint rules unless you state them.

**"It handles state management."** It handles the local state of the component it produced. It has no knowledge of your store, your cache, or your URL.

**"It replaces design judgment."** It translates a description into markup. Someone still has to decide what the component should do at 320px, with an empty dataset, with a 200-character name in a cell, and with a screen reader.

## Edge cases worth testing on every generated component

- **Empty data.** Does the component render an empty state or a broken shell?
- **Single row and single page.** Do the pagination controls disappear or stay disabled?
- **Very long strings.** Does a long name or email break the layout?
- **Slow network.** Is there a loading state, and does it shift layout when it resolves?
- **Rapid interaction.** Does clicking sort repeatedly produce a consistent result?
- **Unusual sort values.** Mixed case, leading spaces, and numeric strings sort incorrectly with a naive comparator.
- **Reduced motion.** If the component animates, does it respect `prefers-reduced-motion`?

## FAQ

**Does this apply to non-React frameworks?**
The failure modes are framework-agnostic: keyboard access, breakpoint behavior, state ownership, and token compliance are properties of the component, not the framework. The specific fixes differ. Generators also tend to be strongest in the framework with the largest training corpus, so output quality varies by target.

**How do I enforce design tokens in generated output?**
State the token names in the prompt, then add a lint rule that rejects raw color and spacing literals in component files. The lint rule is the enforcement mechanism; the prompt is only a hint. Without the lint rule, hard-coded values will reappear on the next generation.

**Can the generated code be tested automatically?**
Yes, and it should be. Render the component in a test environment, run an accessibility assertion against it, assert keyboard interaction with a testing library's user-event API, and snapshot the narrow-viewport render. These are the same tests you would write for a hand-written component.

**Is it worth using these tools at all?**
For components that are mostly presentational and low-risk, the first-draft saving is real. For interactive, stateful, or accessibility-critical components, the refactor often exceeds the saving. Sort your component inventory by that distinction before adopting a tool broadly.

**What about design-tool export plugins?**
They solve a different problem: translating a design file into markup. The same constraints apply. An export can only contain what the design file contains, so focus states, error states, and narrow-viewport layouts must exist in the design file before they can appear in the output.

## What to do in the next 30 minutes

Pick one generated or hand-written component in your codebase that has an interactive element — a sortable header, a toggle, a modal trigger. Open it and check three things: whether the interactive element is reachable by Tab, whether it responds to Enter and Space, and whether it has an accessible name. If any of the three fails, you have found a defect that a generator would also have produced. Fix it, then add the equivalent assertion to your test suite so the next generated component is checked automatically rather than by eye.
