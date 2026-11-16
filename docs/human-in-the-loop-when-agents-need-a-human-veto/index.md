# Human-in-the-loop: when agents need a human veto

An agent that executes a refund, rotates an API key, or sends a password reset email has the same blast radius as a compromised employee account. The difference is that the agent never gets tired, never takes a day off, and never asks itself whether the request is suspicious. When it goes wrong, the symptom is rarely a crash. It is a support ticket that says "we refunded the same order four times," or an audit log entry showing a delete that nobody remembers approving. The real problem is deciding exactly where to put a human veto, and how to enforce it in code, before an agent with production credentials does something you cannot undo.

## The symptom: the approval that never happened

Most teams discover their human-in-the-loop boundary is missing only after an incident. The pattern is consistent:

- A tool call in the agent's trace shows `transfer_funds` or `delete_user` succeeded, but no human approval record exists for that trace ID.
- A customer reports a duplicate charge, and the agent's reasoning shows it retried a payment because the first call timed out — not because the payment failed.
- An engineer notices a production secret in a public repository's commit history, because the agent was given a GitHub token with `repo` scope and "helpfully" pushed its working directory.
- A support agent approves a refund, then watches the same agent issue three more refunds for the same order because the approval was checked at the start of the run, not at the moment of the side effect.

None of these produce a stack trace. They produce an audit gap. The agent did what it was told, and the system had no mechanism to ask "are you sure?" at the point where certainty matters.

## Why the boundary erodes: causes in order of probability

### 1. The approval is checked at the wrong layer

The most common failure is architectural. Teams add a human approval step to the agent's planning loop — the model is told "ask for confirmation before doing anything destructive" — and assume that is a boundary. It is not. A prompt instruction is a suggestion to a stochastic system. The model may skip it, misinterpret it, or be manipulated into ignoring it by content it reads from a tool result (a classic indirect prompt injection).

**How to confirm:** Search your agent's code for the tool that performs the side effect. If the approval check lives in the prompt or in the agent framework's "plan" step rather than in the function that executes the side effect, you have this bug. A quick test: disable the model's ability to call the approval tool and see whether the side effect still fires. If it does, the boundary is decorative.

**Fix:** Move the check into the execution path, below the model. Every tool that mutates state should call a gate function that verifies a signed, single-use approval token tied to the specific action and its parameters. The model cannot forge the token because it never sees the signing key.

```python
# gate.py — a minimal approval gate that the model cannot bypass
import hmac, hashlib, json, time
from dataclasses import dataclass

@dataclass
class Approval:
    action: str
    params_hash: str
    expires_at: float
    signature: str

def sign_approval(secret: bytes, action: str, params: dict, ttl: int = 300) -> Approval:
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    expires_at = time.time() + ttl
    msg = f"{action}:{params_hash}:{expires_at}".encode()
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return Approval(action, params_hash, expires_at, sig)

def require_approval(secret: bytes, approval: Approval, action: str, params: dict) -> None:
    if approval.action != action:
        raise PermissionError("approval action mismatch")
    if time.time() > approval.expires_at:
        raise PermissionError("approval expired")
    expected = sign_approval(secret, action, params, ttl=0).params_hash
    if not hmac.compare_digest(approval.params_hash, expected):
        raise PermissionError("approval params mismatch")
    msg = f"{action}:{approval.params_hash}:{approval.expires_at}".encode()
    expected_sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(approval.signature, expected_sig):
        raise PermissionError("bad approval signature")
```

The human approves by calling `sign_approval` from a separate service that the agent cannot reach directly. The agent receives the `Approval` object as an opaque blob and passes it to the tool. If the agent tries to change the parameters after approval, the hash check fails.

Two properties of this design are worth stating explicitly. First, the token is bound to the parameters, not just the tool name, so an approval for "refund order 1234 for 50.00" cannot be replayed against "refund order 5678 for 5000.00." Second, the TTL bounds the window in which a stale approval can be used; a token minted for a 9 a.m. review should not still be valid after lunch. The `ttl=0` call inside `require_approval` is a small trick to recompute the params hash without worrying about the expiry, since only the hash field is compared there.

A known limitation: this snippet is single-use only if the verifier records consumed token IDs. Add a `token_id` field, store consumed IDs in a store the agent cannot write to, and reject any token seen before. Without that, a captured `Approval` object can be replayed until it expires.

### 2. The approval is too coarse

A second common cause is approving the agent's *intent* rather than the *action*. A human clicks "approve" on a plan that says "process refunds for affected orders," and the agent then processes 400 refunds, including ones it inferred from a fuzzy match. The human approved a category, not a list.

**How to confirm:** Look at what the approval UI displays. If it shows a natural-language description rather than the exact tool name, parameters, and count, the approval is too coarse. Check whether the approval token is bound to a single invocation or to a batch.

**Fix:** Approvals should be bound to a concrete, enumerated set of actions. For batch operations, require the human to see the list and approve the list, not the description. In practice this means the agent must first produce a dry-run plan (a list of `(tool, params)` tuples), the human approves that exact list, and the executor refuses any action not on the list.

A worked example makes the gap concrete. Suppose the agent proposes:

```text
Plan: refund orders affected by the duplicate-charge incident.
Resolved orders: [A-1001, A-1002, A-1003, A-1004] (fuzzy match on customer email)
```

If the human approves the sentence, the executor has no way to know whether four refunds or forty were intended. If the human approves the enumerated list, the executor can hash the list, bind the hash into the token, and refuse any refund whose order ID is not in the approved set. When the agent later discovers a fifth order, the token check fails and the action returns to the human queue. The cost is one extra round trip; the benefit is that the approved unit equals the executed unit.

### 3. The human is rubber-stamping

Even a correctly placed gate fails if the human approves everything. This is the automation-bias failure mode: a queue of 200 approvals per day trains the reviewer to click "approve" without reading. The boundary exists on paper but not in behavior.

**How to confirm:** Measure the time between approval request and approval decision. If the median is under a few seconds for actions that involve money or data deletion, the reviewer is not reading. Also check the rejection rate: if it is effectively zero over a long period, either the agent is perfect (unlikely) or the gate is not being used as a gate. Both metrics are cheap to instrument: log `requested_at` and `decided_at` on every approval record, then compute the median and the reject count per week.

**Fix:** Make approvals expensive to rubber-stamp. Require the reviewer to type the affected resource ID or the amount before approving. Randomly sample a small fraction of approvals for a second reviewer. And surface the *reason* the agent wants to take the action, not just the action itself, so the reviewer has something to evaluate.

There is a tension here worth naming: every friction you add reduces throughput, and every friction you remove increases the chance of a blind approval. The right calibration is not "make it hard" but "make the reviewer's decision genuinely possible." A reviewer who cannot see why the agent wants to act will rubber-stamp regardless of how many digits they have to type. A reviewer who sees "customer reports duplicate charge, order A-1001 already refunded once at 09:14" can make a real judgment in two seconds.

### 4. Secrets and credentials leak through the agent

Agents are often given broad credentials because scoping them is tedious. A common pattern is an agent that can read a repository and also has push access, or an agent with a database user that has `GRANT ALL`. When the agent is compromised via prompt injection or simply makes a mistake, the credential is the blast radius.

**How to confirm:** Audit the environment variables and secret mounts available to the agent process. Check whether any of them are long-lived, broadly scoped, or present in the agent's context window. Search public repositories for the secret's prefix (for example, cloud provider access key IDs) using the provider's own scanning tools.

**Fix:** Give the agent short-lived, narrowly scoped credentials minted per task, and keep the long-lived secrets out of the agent's environment entirely. The approval gate from cause 1 should be the only path to privileged operations. For repositories, use a bot account with read-only access by default and require a separate, human-approved step to push.

A useful mental model: the agent's credential should be no broader than the task it is currently executing, and the privileged operation should be performed by a different principal than the one the agent holds. If the agent must push code, the push should happen through a service that holds the write credential and accepts only approved, signed requests — the agent never touches the write key.

### 5. The audit trail cannot answer "who approved this?"

When something goes wrong, the first question is who authorized the action. If the approval is recorded only in the agent's log stream, or if the approval token is not tied to a human identity, the answer is unavailable.

**How to confirm:** Pick a recent destructive action from the agent's logs and try to trace it back to a human approver. If you cannot, the audit trail is insufficient.

**Fix:** Every approval token should carry the approver's identity, the timestamp, the exact action and parameters, and the agent's trace ID. Store these in an append-only log that the agent cannot write to. The tool executor should log the token ID alongside the action so the two records can be joined later.

The join is the whole point. Two logs that cannot be correlated are two logs that answer nothing. Design the token ID as the primary key that appears in both the approval service's record and the executor's action record, and make both writes happen before the side effect, not after.

## Prevention: designing the boundary before the incident

The causes above share a root: the boundary was added as an afterthought. A more durable approach is to classify every tool by reversibility and blast radius, then apply a consistent policy.

| Tool class | Examples | Default policy |
|---|---|---|
| Read-only, scoped | Search, fetch a record by ID | No approval; log the call |
| Reversible write | Update a draft, add a tag | Approval optional; require idempotency key |
| Irreversible or financial | Refund, transfer, delete, rotate key | Signed approval token, per-action, short TTL |
| Credential or permission change | Grant role, create API key | Two-person approval; separate service |

This table is a starting point, not a standard. The important part is that the policy is enforced in code at the executor, not in the prompt. A tool that is classified as irreversible should refuse to run without a valid token, regardless of what the model says.

Two additional practices matter:

**Idempotency keys on every mutating call.** If the agent retries a refund because the first call timed out, the payment processor should see the same idempotency key and return the original result rather than charging twice. Most payment APIs support this; use it. The agent's retry logic should generate the key once per logical action, not per attempt.

**Kill switch and rate limits.** Even with approvals, an agent can loop. Put a hard cap on the number of mutating actions per trace (for example, a documented limit you choose, such as 10) and a global kill switch that a human can flip without deploying code. The kill switch should be tested regularly, because an untested kill switch is a rumor.

### A decision checklist for a new tool

Before a mutating tool ships, answer these in writing:

1. What is the worst outcome if this tool runs with wrong parameters? If the answer is "money leaves, data is gone, or access is granted," it needs a signed token.
2. Is the approval bound to the exact parameters, or to a description? If the latter, fix the approval UI before shipping.
3. Can the token be replayed? If consumed IDs are not recorded, it can.
4. What is the TTL, and does it match how long a human review realistically takes? A five-minute TTL on a queue with a thirty-minute median wait produces expired-token errors, which teams then "fix" by removing the expiry.
5. Does the executor log the token ID next to the action, in a store the agent cannot write to?
6. Is there an idempotency key on the call, generated per logical action?
7. What is the per-trace cap on mutating actions, and who can flip the kill switch?

If any answer is "we'll add it later," the tool is not ready for production credentials.

## The next concrete step

Open the file that defines your agent's tools and find the function that performs the highest-blast-radius action — the one that moves money, deletes data, or changes permissions. Check whether that function can execute without a signed approval token that the model never sees. If it can, you have found the gap. In the next 30 minutes, add a `require_approval` call at the top of that function and a corresponding signing endpoint in a service the agent cannot reach. The rest of the boundary can follow the same pattern.

## Frequently Asked Questions

**Why can't I just tell the model to ask for confirmation?**
Because the model's instruction-following is probabilistic and can be overridden by content it reads from tools or user input. A prompt instruction is not an enforcement mechanism. The confirmation must be enforced in the code that executes the side effect, where the model has no ability to bypass it.

**How do I approve actions without slowing down every task?**
Classify tools by reversibility and blast radius, and only require approval for the irreversible or high-value ones. Read-only and reversible operations can proceed with logging and idempotency keys. The goal is not to approve everything; it is to make sure the things you cannot undo have a human in the path.

**What is indirect prompt injection and why does it matter for approvals?**
Indirect prompt injection is when an attacker places instructions in content the agent reads — a web page, a document, a tool result — and the model follows them. If the agent has broad credentials and no execution-layer gate, the injected instruction can trigger destructive actions. A signed approval token that the model cannot forge limits the damage.

**How do I audit which human approved which agent action?**
Tie every approval token to the approver's identity, the exact action and parameters, a timestamp, and the agent's trace ID. Store the token record in an append-only log outside the agent's write access, and have the tool executor log the token ID with the action. Joining the two records gives you the full chain.

**Do I need human approval for every agent action in production?**
No. Requiring approval for everything defeats the purpose of an agent and trains reviewers to rubber-stamp. Approve only the actions that are irreversible, financial, or change permissions. For everything else, use idempotency keys, rate limits, and logging so you can detect and reverse problems after the fact.
