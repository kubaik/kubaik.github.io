# Choosing an AI Pair Programming Setup: A Practical Guide

## The problem this guide addresses

Teams moving a high-traffic API from Node 18 to Node 20 LTS on a managed Kubernetes platform commonly see p99 latency jump overnight — sometimes from around 120 ms to 480 ms. The usual suspects get chased first: database connection pools, Kubernetes resource limits, Node garbage collection. Often none of them move the needle. What actually helps is having a second set of eyes that can ask the right question at the right time, whether that is in a chat channel, on a pull request, or inside an editor.

At the same time, senior engineers become onboarding bottlenecks, junior developers ship bugs that review could have caught, and PR cycles stretch to multiple days even for trivial changes. AI pair programming tools are interesting in this context not as a replacement for engineers but as a way to spread knowledge faster without multiplying meetings.

This guide covers the categories of setup that have emerged, how to evaluate them honestly, and what to actually configure. It is deliberately tool-agnostic in places, because the specific product names change faster than the underlying trade-offs.

## How to evaluate any setup

Judge every setup against three metrics that matter in production. All three require instrumentation; do not trust vendor claims.

1. **Latency to first useful suggestion.** Measure from the moment a developer types or posts a question to the moment a suggestion appears that the developer accepts or acts on. Instrument this with a stopwatch during a two-week trial, or log timestamps if your tooling supports it. "Useful" is the key word — a suggestion that appears in 200 ms but is wrong does not count.

2. **Signal-to-noise ratio.** Track the percentage of suggestions that were actionable. A simple mechanism: in a shared channel, react with a checkmark when a suggestion was used and a cross when it was rejected. After two weeks you have a defensible number. Do not rely on the tool's own dashboard, which typically counts any displayed suggestion as a hit.

3. **Cost per 1,000 suggestions.** For cloud models, pull the actual token counts from your provider's billing console and multiply by the published per-token price. For self-hosted models, cost is dominated by GPU hours: take your instance type's hourly rate, multiply by hours running, and divide by suggestions served. Include idle time — a GPU that sits warm overnight still bills.

Run each candidate setup in parallel for two weeks on real workloads. A representative test matrix:

- A large TypeScript monorepo using a recent TypeScript release
- A Python microservice with FastAPI
- A legacy Java service that has not seen a major refactor in years

The benchmark is simple: can the AI catch the kind of bugs that have slipped into production in the last six months? Seed the evaluation with those actual bugs and see whether the tool surfaces them.

## Setup categories and what each is good at

Each item below is a mini-review: what it does, one strength, one weakness, and who it is best for. Where configuration matters, the relevant file is shown.

### 1. Cloud editor assistant with chat mode

**What it does:** A full-time pair programmer that lives in your editor. It watches what you type, suggests code snippets, and answers questions in natural language. Cloud mode sends context to the provider's model; the context window is whatever the model exposes, typically large but not unlimited.

**Strength:** Onboarding speed. A new hire on a Java team can go from zero to shipping a bug fix in a day instead of a week. The AI frequently catches null-dereference bugs in the first PR review that a senior reviewer misses because they assumed the input was sanitised.

**Weakness:** Context drift after long idle periods. Developers commonly find themselves repeating the same setup instructions every session. The context window also truncates large repos, so the assistant often loses track of imports deep in the codebase.

**Best for:** Teams that need fast, low-friction pair programming without adding meetings. Ideal for onboarding and incremental improvements, not large refactors.

### 2. Editor with project-wide embeddings

**What it does:** Some editors ship with a local inference engine and project-wide embeddings. They index your entire repo and answer questions like "Where do we handle JWT validation?" in seconds.

**Strength:** Repo-wide semantic search. When asked, "Where does the auth middleware parse the refresh token?" a good implementation returns the exact line with a citation. This is materially faster than grep for questions phrased in natural language.

**Weakness:** Local inference requires a serious GPU and plenty of RAM. Without that, suggestions are slow and often stale. Embeddings rebuilds are not real-time — on a 200k-line repo, expect tens of minutes.

**Best for:** Mid-to-large codebases where grep is not enough and you need answers faster than a human can scan. Not for teams without GPU budget.

### 3. Cloud provider assistant with infrastructure context

**What it does:** A cloud-based assistant that connects to your cloud resources — source hosting, build, pipeline — and can run in a mode where it has full repo context. It uses a model trained on the provider's documentation and public repos.

**Strength:** Infrastructure-aware suggestions. When asked, "Why is my Lambda timing out?" it can query the provider's log service, identify the cold-start bottleneck, and suggest a memory increase. That is a real workflow improvement for cloud-heavy teams.

**Weakness:** Vendor lock-in. The workspace mode only works with that provider's services, so on-prem teams cannot use it. The context window is also shallow — it often misses files outside the main repo.

**Best for:** Teams all-in on one cloud provider who need infrastructure-aware pair programming. Not for hybrid or multi-cloud teams.

### 4. Fast editor with local embeddings

**What it does:** Some editors are built for latency, with AI pair mode that can run local models with repo embeddings. Suggestions appear inline as you type.

**Strength:** Real-time, inline suggestions feel like a second brain. TypeScript teams commonly use them to catch race conditions in Redux middleware loops that elude unit tests. The fix is often a single line change:

```tsx
// Before: effect re-runs on every render because the dependency is a new object
useEffect(() => {
  dispatch(fetchUser(userId));
}, [{ userId }]);

// After: depend on the primitive value
useEffect(() => {
  dispatch(fetchUser(userId));
}, [userId]);
```

**Weakness:** Hallucination in edge-case TypeScript types. Local models have been known to suggest non-existent methods such as `Array.prototype.flatMapAsync`, which can take significant time to debug. Local models also drift quickly — embeddings need periodic rebuilding.

**Best for:** Fast-moving frontend teams who prioritise speed over accuracy and have the infrastructure to run local models.

### 5. Open-source extension with local model backend

**What it does:** An open-source editor extension that lets you plug in any LLM backend, including a locally-running one. A typical setup runs on a high-end laptop with a quantised 8B model.

**Strength:** Cost per suggestion is effectively zero once the hardware is paid for. If you already own the laptop, the marginal cost of 1,200 monthly suggestions is nothing compared to a cloud model. The local model also respects privacy: no data leaves the machine.

**Weakness:** Context window is small — often just a few thousand tokens. It loses track of imports after a few files. Smaller models also struggle with Java generics, which backend teams rely on heavily.

**Best for:** Teams with low budgets or strict privacy needs who can tolerate lower accuracy. Not for large codebases.

### 6. IDE-native assistant with local and cloud fallback

**What it does:** An assistant built into a commercial IDE, using a mix of a local model and cloud fallback. It integrates with the IDE's refactoring and debugging tools.

**Strength:** IDE integration is seamless. The AI can refactor entire classes, update tests, and even run the debugger. When asked to "extract this method," it does the refactor, updates all callers, and runs the tests in one action.

**Weakness:** Licensing is opaque, and the cloud fallback can be slow during peak hours. Local models on large files occasionally crash or time out.

**Best for:** Java/Kotlin teams who live in a specific commercial IDE and want deep integration. Not for teams on VS Code or Neovim.

### 7. Terminal-first assistant

**What it does:** A modern terminal with an AI pair built in. It can answer questions like "What's the last error in this log?" and suggest commands. Commonly used with Kubernetes cluster logs.

**Strength:** Terminal-first workflows. When an SRE asks, "Why is this pod crashing?" a good terminal assistant parses the last hundred log lines, points to the OOM kill, and suggests `kubectl top pod` to confirm.

**Weakness:** Limited to terminal context. It cannot answer repo-level questions like "Where is the auth service?" unless you pipe the repo into it. Terminal assistants have also been known to suggest unsafe commands without warning.

**Best for:** SREs and DevOps teams who live in the terminal and need fast log parsing. Not for developers building features.

### 8. Cloud IDE with built-in assistant

**What it does:** An AI pair that lives in a hosted IDE. It can run code in real time, answer questions, and debug. Commonly used for quick prototypes and code reviews.

**Strength:** Zero-setup prototyping. A new developer can build a FastAPI endpoint in twenty minutes that would have taken half a day in a local IDE. The AI also catches CORS misconfigurations that would have blocked the endpoint.

**Weakness:** Proprietary runtime means lock-in. The AI also suggests code that only runs in the hosted sandbox — such as provider-specific imports — which breaks in production. The context window is also shallow.

**Best for:** Teams doing quick prototypes or hackathons where zero setup matters more than production-readiness.

### 9. Self-hosted enterprise assistant

**What it does:** An on-prem AI pair that uses a serving framework to host models like a 13B code model. A typical deployment runs on Kubernetes with multiple datacenter GPUs.

**Strength:** Privacy and scale. Running it behind a VPN means no data leaves the cluster. A well-tuned serving framework can handle hundreds of suggestions per minute with low latency.

**Weakness:** Operational overhead is brutal. Tuning the serving config commonly takes weeks, and GPUs still crash under load. Models also need periodic fine-tuning or replacement as upstream codebases drift.

**Best for:** Large teams with strict privacy requirements and DevOps capacity who can tolerate operational pain.

## How to measure the things vendors claim

Every number in a vendor deck should be reproducible by you. Here is how to reproduce the important ones.

**Latency.** Add a timestamp log line in your editor extension or proxy. Record `t_request` when the developer submits a prompt and `t_response` when the first token arrives. Report p50 and p95, not averages — averages hide the tail that ruins flow.

**Signal-to-noise.** Instrument the accept/reject action. In VS Code, extensions typically fire a command on accept; in a terminal, you can log when a suggested command is actually executed. Divide accepts by total suggestions shown.

**Cost per suggestion.** For a cloud model, take the total billed tokens over a week, divide by the number of suggestions served that week, and multiply by the per-token price. For a self-hosted model, take the GPU instance hourly rate multiplied by hours running, divided by suggestions served. Include idle time.

**Hallucination rate.** Sample 100 suggestions from production logs, have two engineers independently label each as correct, incorrect, or unverifiable, and measure agreement. If agreement is below roughly 80%, your labelling rubric needs work before the number means anything.

## A worked example: estimating cost for a 200-person team

Assume an illustrative scenario. These numbers are labelled illustrative because your actual usage will differ.

- 200 engineers
- Each engineer triggers 40 suggestions per working day
- 20 working days per month

Total suggestions per month = 200 × 40 × 20 = **160,000**.

If a cloud model costs $3 per 1,000 suggestions (a figure you must verify against your provider's current pricing), the monthly bill is 160 × $3 = **$480**, or about $2.40 per engineer per month. That is the arithmetic; plug in your own per-1,000 rate.

For a self-hosted model, assume four datacenter GPUs at an illustrative $3/GPU-hour, running 24/7. Monthly GPU cost = 4 × $3 × 24 × 30 = **$8,640**. Divide by 160,000 suggestions = **$0.054 per suggestion**, or $54 per 1,000. The self-hosted path is cheaper only if you are serving far more traffic than 160,000 suggestions per month, or if privacy requirements make cloud use impossible. This is the calculation to run, not a result to trust.

## Failure modes worth knowing before you commit

**Context fragmentation.** Teams that run multiple AI pairs on the same repo often see contradictory suggestions. One tool suggests a type hint that another flags as incorrect. The developer then spends time debugging the conflict. One AI per repo is the rule most teams settle on.

**Unsafe command suggestions.** Terminal assistants have suggested destructive commands such as `rm -rf /tmp/*` without warning. Always require human confirmation for any command that mutates state.

**Hallucinated APIs.** Local models in particular invent methods that do not exist. A code review gate that requires the suggestion to compile and pass tests catches most of these.

**Silent context truncation.** When the context window fills, the model does not always tell you. It simply starts ignoring the oldest files. Symptoms include repeated imports of already-imported modules and references to functions that were renamed an hour ago.

**Cost surprise from retries.** Cloud assistants that retry on transient errors can double your token spend. Check your billing console for retry counts.

## Decision checklist

Answer these before selecting a setup:

- Do you have a hard privacy requirement that forbids sending code to a third party? If yes, only self-hosted or local options are viable.
- Do you have GPU budget and DevOps capacity? If no, a cloud option is the pragmatic choice.
- Is your primary language well-served by the model? Java generics and complex TypeScript types are common weak spots.
- What is your actual suggestion volume? Under a few hundred thousand per month, cloud is almost always cheaper than self-hosting.
- Who owns the evaluation? Assign one engineer to run the two-week trial and publish the three metrics.
- What is the rollback plan? If the tool is removed, what happens to code already written with it? Usually nothing, but check for license headers or generated-file markers.

## Frequently asked questions

### Why not use several tools at once?

Because context fragmentation kills signal-to-noise. In practice, teams that use multiple AI pairs see contradictory suggestions, and the developer spends time debugging the conflict instead of shipping. One AI per repo is the rule most teams settle on.

### How do you prevent hallucinations from reaching production?

Add a human gate in the PR workflow. Every PR that includes AI-generated code should have a human reviewer who did not write the AI-generated line, a passing test that covers the change, and a comment noting the prompt used. Hallucination rates drop substantially after enforcing this. The trade-off is slower PR throughput, but shipping slow beats shipping broken.

### What is the learning curve for junior developers?

Juniors tend to over-trust the AI. Common bugs from copying suggestions without understanding them include memory leaks from unclosed connections, race conditions in async loops, and SQL injection from string interpolation in queries. A simple training rule works well: ask the AI, then prove it. Write a test or run the code before merging.

### How do you measure ROI?

Track three metrics before and after adoption: PR cycle time (days from open to merge), bug escape rate (bugs found in production versus in development), and onboarding time (days to first production commit). Publish the numbers. ROI is usually clear within a few weeks if it is clear at all.

### Can AI pairs replace code reviews?

No. AI pairs catch syntax errors and style issues but miss logical bugs often enough that a human reviewer is still required. The best use is pre-review: the AI catches the easy stuff, and humans focus on the hard stuff. Think of it as a filter, not a replacement.

## Action to take in the next 30 minutes

Pick one repo and one candidate tool. Create a `.github/copilot-instructions.md` file (or the equivalent for your tool) with concrete rules, then run a one-week trial and log the three metrics.

```markdown
# Copilot Instructions
- Always suggest tests with new code
- Flag SQL injections and memory leaks
- Never suggest `rm -rf` or similar destructive commands
- Prefer explicit types over `any`
```

Commit the file, tell your team it exists, and start the timer. In one week you will have real numbers instead of vendor claims.
