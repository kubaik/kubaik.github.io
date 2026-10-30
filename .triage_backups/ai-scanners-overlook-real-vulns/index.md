# AI scanners overlook real vulns

AI-powered vulnerability scanners are effective at finding typos, missing braces, and dependencies with known CVEs. They routinely miss the bugs that cause incidents: logic flaws, authentication bypasses, and race conditions. The practical response is to treat an AI scanner as a first pass and combine it with deterministic checks, targeted fuzzing, and runtime monitoring.

## The one-paragraph version

AI scanners excel at pattern matching and version lookups but struggle with application context—the "why" behind a piece of code. A typical failure mode is a scan that reports dozens of findings, most of them false positives, while a subtle data race or token-reuse flaw sits unflagged. Teams that reduce escaped defects treat AI as one layer among several: AI-assisted rule drafting, deterministic static analysis for invariants, fuzzing for edge cases, and runtime monitoring for what slips through. The rest of this article explains why the layers differ, shows a worked example of finding a file-upload bypass, and gives a checklist for combining them without drowning in noise.

## Why "AI-powered" doesn't mean "smarter"

The intuition is that an AI scanner must reason about code better than a rule-based one. That is only partly true. An LLM-backed scanner is guessing based on patterns in its training data. It does not model your application's state transitions, your authentication boundaries, or the invariants your code is supposed to hold.

Consider a scanner that flags 47 issues. Suppose 39 are false positives, 6 are real but low-severity, and the subtle data race that later causes an outage is not among them. That distribution is common. The scanner is not broken; it is doing what pattern matching does. It recognizes shapes it has seen before and fails on shapes it has not.

Marketing language compounds the confusion. Products described as "AI-powered security" often combine conventional static analysis with LLM-generated suggestions. The LLM portion is useful for drafting rules and summarizing findings, but it is not reasoning about your system. It is pattern-completing.

A second trap is assuming the scanner catches everything. AI scanners do not model application state transitions, so they tend to miss flaws that only appear across a sequence of requests—for example, a password reset token that is not invalidated after a second reset. You can feed the scanner every known CVE pattern, and it still will not notice that.

## The mental model: spellchecker versus grammar checker

Think of an AI scanner as a spellchecker. It catches typos—misused functions, outdated libraries, obvious mistakes—but it does not understand whether a sentence is logically consistent.

Now add a grammar checker: deterministic rules that enforce structure. Suddenly you have something closer to a real editor. Map that onto security:

- **AI as spellcheck**: finds typos (misused functions, outdated libraries).
- **Deterministic rules (grammar)**: enforce invariants (null checks, auth boundaries, CSRF token presence).
- **Fuzzing (unit tests)**: probe edge cases beyond what rules describe.
- **Runtime monitoring (production behavior)**: catch what slips through the first three layers.

In practice, AI is most useful for generating candidate rules or queries—for example, asking an LLM to suggest static-analysis queries for your API's input validation—which you then validate against real examples from your codebase. The LLM can draft; the codebase decides whether the draft is correct.

## A worked example: file-upload bypass

Walk through a Node.js API that handles user uploads. The goal is to find a file-upload bypass that could become a remote code execution (RCE) vector. Three layers are used: an AI-assisted analyzer, a deterministic query engine, and a fuzzer.

### Step 1: AI-assisted rule generation

A typical AI-assisted analyzer might suggest a rule along the lines of: "Check if the file extension is in the allowed list." That is a reasonable starting point, but it is not sufficient. An attacker can bypass a naive extension check with a double extension (`file.png.php`) or a null byte (`file.jpg\0.php`).

The AI suggestion is a draft. It needs to be turned into something deterministic and testable.

### Step 2: Turn the suggestion into a deterministic rule

Using a CodeQL-style query language, the suggestion becomes an explicit check:

```ql
import javascript

from UploadHandler handler
where handler.getFileExtension().notIn(["jpg", "png", "gif"])
select handler, "File type not allowed: " + handler.getFileExtension()
```

Add a second rule for path traversal, which the AI suggestion did not cover:

```ql
from UploadHandler handler
where handler.getFileName().matches("%../%")
select handler, "Path traversal detected in filename"
```

The second rule catches filenames like `../../../etc/passwd.jpg` that a pattern-matching scanner may miss because the exact shape was not in its training data. The point is not that the query language is magic—it is that the rule is explicit, auditable, and reproducible. Anyone can read it and argue about whether it is correct.

### Step 3: Fuzz the edge cases

Deterministic rules cover what you thought to describe. Fuzzing covers what you did not. A minimal harness for the upload endpoint:

```javascript
// upload.harness.js
const http = require('http');

const server = http.createServer((req, res) => {
  if (req.url === '/upload' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, size: body.length }));
    });
  }
});

server.listen(3000, () => {
  console.log('Harness listening on port 3000');
});
```

Note that this harness is deliberately minimal: it echoes the body size and does not actually persist the upload. That is fine for a first pass, because the goal is to exercise the parsing and handling path, not the storage layer. When a crash appears, replace the echo with the real handler to confirm.

A fuzzer such as libFuzzer or a general-purpose mutation fuzzer can then be pointed at the harness. A plausible finding is a crash when the filename contains a very large number of null bytes—a pathological case that no pattern-matching scanner would guess. That crash may be a memory-exhaustion bug and a denial-of-service vector.

The important discipline is scoping. Do not fuzz the entire application. Fuzz the attack surface: parsers, uploads, authentication handlers. A targeted campaign on a single endpoint can run in minutes, not weeks.

## What each layer is actually good at

| Layer | Strength | Weakness | Where it fits |
|---|---|---|---|
| AI-assisted scanner | Fast, broad coverage of known patterns | High false positives, weak on context | First pass, triage |
| Deterministic static analysis | Precise, auditable, reproducible rules | Requires writing and maintaining rules | Invariants, auth boundaries |
| Fuzzing | Finds pathological inputs and crashes | Slow on large codebases, needs harnesses | Parsers, uploads, auth handlers |
| Runtime monitoring | Sees actual behavior in staging/production | Reactive; needs tuning to avoid noise | Detecting what static analysis missed |

The layers are complementary, not competing. AI is fast but shallow. Deterministic rules and fuzzing are slower but deeper. Runtime monitoring is the backstop.

## Common misconceptions, corrected

**Misconception 1: AI scanners replace manual review.**

They reduce the volume of low-hanging fruit but they do not replace context. A logic flaw in JWT validation that allows token reuse will not match a known pattern and will not be caught by a rule that does not exist yet. Manual review—or a deliberately written rule—is still required for logic that spans requests.

**Misconception 2: Fuzzing is too slow to be practical.**

A targeted campaign on a single endpoint can run in under an hour. The trick is scope: fuzz the attack surface, not the whole app. Start with one endpoint per sprint and run it nightly.

**Misconception 3: AI-generated rules are always safe.**

LLMs hallucinate. An AI-drafted query such as:

```ql
from SqlQuery q
where q.getText().contains("SELECT")
select q
```

will flag every SELECT statement, including safe ORM calls. A more useful shape is:

```ql
from SqlQuery q
where q.getText().matches("%'% + %") and not q.isPrepared()
select q
```

Even that needs auditing against real code. The rule is a hypothesis; the codebase is the test.

**Misconception 4: Runtime protection is only for production.**

Runtime monitoring can run in staging. Tools that observe system calls and network traffic in a staging cluster can catch behavior invisible to static analysis, at the cost of some tuning to keep alert volume manageable.

## How to measure whether your layers are working

Claims about "fewer incidents" are easy to make and hard to verify. If you want to know whether a layered approach is helping, instrument the following and compare over time:

- **Findings by layer**: how many issues each layer reports, and how many are confirmed real. Track the false-positive ratio per layer.
- **Escaped defects**: issues found in production that a layer should have caught. For each one, record which layer missed it and why.
- **Time to first finding**: how long after a code change each layer reports something.
- **Rule churn**: how often deterministic rules are added, modified, or retired. High churn suggests the rules are not capturing the right invariants.

A simple starting point: pick a recent incident or a public CVE from the last six months. Try to reproduce the vulnerability against your scanner. If the scanner does not flag it, you have found a gap. Record the gap and decide which layer should close it.

## Decision checklist for combining layers

Use this when deciding what to add next:

- [ ] Do you have at least one deterministic rule per authentication boundary?
- [ ] Does every parser or upload handler have a fuzz harness?
- [ ] Are AI-generated rules reviewed against real code before being enabled in CI?
- [ ] Is there a documented process for triaging false positives, rather than disabling rules?
- [ ] Is runtime monitoring running in staging, not only production?
- [ ] For each escaped defect, is the responsible layer identified and improved?

If most boxes are unchecked, adding more AI scanning will not help. The gap is in the layers that require explicit rules and harnesses.

## Advanced layers, once the basics are solid

Once AI-assisted rule drafting, deterministic queries, and targeted fuzzing are in place, two further techniques are worth considering.

### Property-based testing

Property-based testing generates random inputs and asserts invariants that should hold for all of them. For example, a test could assert that a password reset token is invalidated after a second reset:

```go
func TestTokenInvalidation(t *testing.T) {
    token1 := generateToken()
    resetToken(token1)
    if !isTokenInvalid(token1) {
        t.Fatal("token1 should be invalid after reset")
    }

    token2 := generateToken()
    resetToken(token2)
    if !isTokenInvalid(token2) {
        t.Fatal("token2 should be invalid after reset")
    }

    // A second reset of token1 should also leave it invalid.
    resetToken(token1)
    if !isTokenInvalid(token1) {
        t.Fatal("token1 should remain invalid after second reset")
    }
}
```

This kind of test can surface race conditions where two concurrent resets fail to invalidate the first token. The invariant is the point: "a token is invalid after reset" must hold regardless of order or concurrency.

### Symbolic execution

Symbolic execution explores paths through a program by treating inputs as symbolic values and solving constraints. It is most useful for small, critical code—cryptographic parsers, authentication primitives—where exhaustive path coverage matters. The cost is setup complexity and potential state explosion; it is not a general-purpose replacement for the other layers.

### Runtime SBOM and drift detection

Runtime SBOM tooling compares what is actually loaded at runtime against what was recorded at build time. A container that loads a library not present in the build-time SBOM is a signal worth investigating. This is a narrow but useful check, and it belongs alongside the other runtime monitoring.

## Quick reference

| Layer | Purpose | Cost profile |
|---|---|---|
| AI-assisted scanner | Broad first pass | Usually subscription-based |
| Deterministic static analysis | Enforce invariants | Free or bundled; requires rule-writing time |
| Fuzzing | Find pathological inputs | Free; requires harness-writing time |
| Runtime monitoring | Detect actual behavior | Free or subscription; requires tuning |
| Property-based testing | Assert invariants over random inputs | Free; requires test-writing time |
| Symbolic execution | Exhaustive path coverage on small code | Free; high setup cost |

## FAQ

**How do I know if my AI scanner is missing real bugs?**

Audit its false negatives. Pick a recent incident or a public CVE from the last six months. Try to reproduce the vulnerability against the scanner. If it is not flagged, you have found a gap. Then decide which layer should close it—usually a deterministic rule or a fuzz harness.

**Is it safe to use AI-generated security rules in production?**

Only after validating them against your own codebase in a non-blocking mode first. Run the rule, inspect the findings, and refine it before enabling it in CI. A rule that flags too broadly will be disabled by frustrated engineers, which is worse than no rule.

**What's the best way to introduce fuzzing without slowing down CI?**

Start with one endpoint per sprint, run the campaign nightly rather than on every commit, and keep the harness small. The goal is to find crashes, not to achieve exhaustive coverage on day one.

**Can runtime monitoring replace static analysis?**

No. Runtime monitoring is reactive: it catches what is already happening. Static analysis is proactive: it prevents issues from reaching production. Use both. When a runtime alert fires, ask which static rule or fuzz harness would have caught it earlier, and add it.

## Your next 30 minutes

Open your most critical API endpoint and write one deterministic rule that encodes an invariant it must satisfy—for example, that every state-changing request validates a CSRF token, or that every file upload rejects path separators in the filename. Run it against your codebase, inspect the findings, and refine the rule until every hit is either a real issue or a documented exception. That single rule is the first layer of the grammar checker your AI scanner is missing.
