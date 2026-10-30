# Fraud detection: rules vs scoring, where they meet

Fraud detection systems tend to converge on the same shape: a **rules engine** that evaluates explicit, human-authored conditions, and a **scoring model** that outputs a continuous risk number. The two are usually described as competing approaches. In practice they solve different problems, and most production incidents come from the boundary between them rather than from either component in isolation.

The boundary is not a matter of taste. It is determined by three properties: how fast the decision must be made, who owns the explanation when a customer complains, and whether the underlying pattern is stable enough to survive a model retrain. Get the boundary wrong in either direction and two classic failure modes appear: a rules engine that grows into a decision tree nobody can reason about, or a model that blocks legitimate customers with no audit trail a support agent can read.

This article is about where to draw that line, how to move signals across it, and what to do when the line has to move later.

## The failure sequence

Most tutorials introduce fraud detection as a machine learning problem: get a dataset, train a classifier, deploy it. That framing is incomplete in a way that tends to surface months later. A typical progression:

1. **Early:** A simple rules engine ships. `if amount > threshold: flag`. It works.
2. **A few months in:** Fraudsters adapt. More rules get added. Then exceptions to rules.
3. **Later:** The rules engine has hundreds of rules, some of them contradict each other, and nobody can determine which one fired for a given transaction without replaying the whole chain.
4. **Then:** Someone proposes a model. It performs better on the offline dataset. It ships.
5. **Finally:** The model blocks a customer who calls support. Support cannot explain why. Compliance asks for the reason. Nobody can produce it.

Steps 3 and 5 look like failures of the *approach* ("rules are bad", "models are opaque") when they are actually failures of *architecture*. A rules engine with 400 rules is not a rules engine problem; it is a missing scoring layer. A model with no explanation is not a model problem; it is a missing rules layer for the audit path.

The two components solve different problems. Rules encode **policy** — things a human decided, that must be explainable and must not drift. Models encode **pattern** — statistical regularities that change over time and are too complex to write by hand. Policy and pattern are both real. A system that only does one will eventually need the other.

## A mental model, and where it breaks

Consider a hospital triage desk. At the front door, a nurse applies a short checklist: is the patient breathing, is there visible bleeding, is there chest pain. That checklist is fast, deterministic, and identical for every patient. It is a **rules engine**. Its job is not to diagnose; it is to route and to catch obvious emergencies before they wait in line.

Behind the desk, a doctor runs tests and forms a probability of what is wrong. That is a **scoring model**. It is slower, it weighs many weak signals, and its output is a number ("80% likely pneumonia"), not a yes/no.

The analogy breaks down in one important way: in a hospital, the nurse and doctor are different people with different training. In a fraud system, the rules engine and the model are both code, and the interesting design question is what the rules engine does with the model's output — and vice versa.

There are three canonical arrangements:

**Rules-then-score.** The rules engine runs first and short-circuits. If a rule fires with a hard block, the model never runs. This is the cheapest arrangement and the most common. It is right when some conditions are genuinely non-negotiable (a sanctioned jurisdiction, a device fingerprint on a denylist) and running a model on them wastes latency and compute.

**Score-then-rules.** The model runs first, produces a risk score, and the rules engine interprets that score against policy thresholds. "If score ≥ 0.9, hold for review. If 0.7–0.9, allow but require step-up authentication." This is the arrangement that scales, because the rules become a thin policy layer over a rich signal.

**Interleaved.** The model consumes rule outputs as features ("did the velocity rule fire?") and the rules consume model outputs as conditions. This is the most powerful and the hardest to debug, because the two systems now have a feedback loop. Teams often arrive here by accident rather than by design, and it is worth being deliberate about it.

Most production systems are best served by **score-then-rules** with a small **rules-then-score** gate in front for genuinely non-negotiable cases. The rest of this article is about why, and how to build it.

## A worked example

Assume a payments API handling card-not-present transactions. Latency budget for the fraud check is 150 ms p99 — this is an illustrative budget, not a measured one; substitute your own. The system has three inputs available synchronously: the transaction payload, a device fingerprint, and a customer history lookup from a fast store (for example, a key-value cache with single-digit-millisecond reads).

Start with the rules gate. These are policy, not pattern. They are the conditions you would be uncomfortable explaining to a regulator if you did not have them:

```python
# rules_gate.py — pure functions, no I/O, deterministic
# Each rule returns (fired: bool, reason_code: str)

def hard_block_rules(txn, device, history):
    reasons = []

    # Policy: sanctioned jurisdictions
    if txn['billing_country'] in SANCTIONED_COUNTRIES:
        reasons.append('SANCTIONED_JURISDICTION')

    # Policy: device on a shared denylist
    if device['fingerprint'] in DENYLISTED_FINGERPRINTS:
        reasons.append('DENYLISTED_DEVICE')

    # Policy: card reported stolen in the last 24h
    if history.get('card_reported_stolen_at') and \
       (now() - history['card_reported_stolen_at']) < timedelta(hours=24):
        reasons.append('CARD_REPORTED_STOLEN')

    return reasons
```

Note what these rules are *not*: they are not "amount > $500", not "velocity > 5/hour", not "country mismatch". Those are pattern signals. They belong in the model, or as features feeding the score, because the right threshold for them changes with the fraud landscape and with your customer base. A hard-coded `amount > $500` block is a rule masquerading as policy, and it will be the rule that blocks a legitimate customer buying a laptop and generates a support ticket nobody can resolve.

Now the scoring layer. The model takes features derived from the transaction, device, and history, and outputs a probability. The exact model is out of scope — logistic regression, gradient-boosted trees, and small neural nets are all used in production — but the interface matters:

```python
# scorer.py — wraps the model, returns a calibrated score

def score_transaction(txn, device, history):
    features = build_features(txn, device, history)
    # model.predict_proba returns P(fraud) in [0, 1]
    score = model.predict_proba(features)[0, 1]
    return score, features  # return features for the audit log
```

Two things about that return value. First, it returns the features alongside the score. In an audit or dispute, "the model said 0.87" is useless; "the model said 0.87 because the device was new, the email domain was 3 days old, and the shipping address was 400 km from the billing address" is a reason a human can evaluate. Second, the score must be **calibrated** — meaning a score of 0.8 should correspond to roughly an 80% empirical fraud rate in that band. An uncalibrated model's thresholds are meaningless, and this is one of the most common reasons a model that "looks good offline" produces nonsense in production.

Now the policy layer that interprets the score:

```python
# policy.py — the only place thresholds live

def decide(score, hard_block_reasons):
    if hard_block_reasons:
        return Decision(action='BLOCK',
                        reason_codes=hard_block_reasons,
                        source='rules')

    if score >= 0.90:
        return Decision(action='HOLD_FOR_REVIEW',
                        reason_codes=['HIGH_RISK_SCORE'],
                        source='model',
                        score=score)
    if score >= 0.70:
        return Decision(action='ALLOW_WITH_STEP_UP',
                        reason_codes=['ELEVATED_RISK_SCORE'],
                        source='model',
                        score=score)
    return Decision(action='ALLOW', source='model', score=score)
```

That is the entire architecture in three files. The rules gate handles policy and short-circuits. The model handles pattern. The policy layer is the only place where thresholds live, which means when a business stakeholder says "we are getting too many step-ups", there is exactly one file to change, and the change is a diff a non-engineer can review.

### What this buys you

- **Explainability for free.** Every decision carries `source` and `reason_codes`. Support can read them. Compliance can read them. The dispute team can read them.
- **Independent evolution.** The model can be retrained on its own cadence without touching the rules. The thresholds can be tuned without retraining the model.
- **A clean audit boundary.** When a regulator asks "why was this transaction blocked", the answer is either a named policy rule or a score with its top contributing features. Both are defensible.

### What it costs you

- **Two systems to operate.** The model needs a feature pipeline, a training loop, and monitoring for drift. The rules need a config store and a way to test changes before they ship.
- **A boundary to maintain.** The temptation to add "just one more rule" to the gate is constant, and each one erodes the model's value. Discipline here is a real ongoing cost.

## How this connects to systems you already know

If you have worked with web application firewalls, this is the same shape. A WAF has a rule set (explicit signatures, blocklists) and increasingly a scoring component (anomaly scores, bot-detection heuristics). The rules catch the known-bad; the score catches the unusual. The operational lesson from WAFs transfers directly: rule sets rot, and the teams that survive treat rules as a small, curated, versioned artifact rather than an append-only log.

If you have worked with authorization systems, the parallel is even closer. Role-based access control is a rules engine — explicit, auditable, and deliberately boring. Attribute-based access control and policy-as-code engines add a scoring-like evaluation over attributes. The same tension appears: explicit rules are easy to reason about individually and impossible to reason about collectively once there are hundreds; attribute-based evaluation scales but is harder to explain. Fraud detection is authorization's messier cousin.

If you have worked with rate limiters, the analogy is about placement. A rate limiter at the edge is a rules gate — cheap, fast, blunt. A rate limiter that adapts to traffic shape is a scoring layer. The adaptive one does not sit in front of the cheap one, because the cheap one exists to protect the expensive one. Same here: the rules gate protects the model from having to run on traffic that is already decided.

## Common misconceptions, corrected

**"Models are more accurate, so we should replace the rules."** Accuracy on a static dataset is not correctness in production. Some conditions are not statistical questions. "Is this country sanctioned?" has a deterministic answer and a legal requirement attached. A model that gets it right 99.9% of the time is a compliance failure 0.1% of the time, which is unacceptable when the cost is a regulatory finding. Keep the non-negotiable rules as rules.

**"Rules are explainable, models are not."** Only the first half is reliably true. A model can be made explainable enough for most operational purposes — feature attributions, decision-path extraction for tree ensembles, or simply logging the top contributing features alongside the score. The reason models feel opaque is usually that nobody instrumented them, not that they are inherently unexplainable. Conversely, a rules engine with 400 interacting rules is not explainable in any useful sense; a support agent cannot hold 400 rules in their head either.

**"We can start with rules and add a model later."** True, but the migration is harder than it looks if you did not design for it. The teams that migrate cleanly are the ones that kept their rules as pure functions with a clear input/output contract, so the model can be dropped in behind the same interface. Teams that embedded rules directly into request handlers and database queries end up rewriting.

**"The score is the decision."** The score is an input to the decision. The decision is policy. Conflating them means every threshold change requires a model redeploy, which is slow, risky, and couples two teams that should be independent.

**"More features means a better model."** More features means more ways to leak future information into training (label leakage), more drift surface, and more latency at inference. The feature set should be the smallest set that carries the signal, and every feature should have a documented reason for existing.

## Measuring the things that matter

Because fraud systems are evaluated against moving targets, the useful discipline is knowing what to instrument before you argue about architecture.

- **Score calibration.** Bin live scores into deciles and compare the predicted fraud rate in each bin to the observed rate. A reliability diagram is the standard visualization. If the 0.8–0.9 bin is not roughly 80–90% fraud, your thresholds are lying to you.
- **Rule hit rates over time.** Log every rule firing with a timestamp. A rule whose hit rate decays toward zero is either obsolete or being evaded. Both are worth a decision.
- **Counterfactual coverage.** For every decision, log the score and the rule outcomes even when the gate short-circuited. Without this, you cannot answer "what would the model have said" after the fact.
- **Latency percentiles per layer.** Measure the gate and the model separately at p50, p95, and p99. If the model's p99 is near your budget, the gate is protecting you more than you think.
- **Review outcomes as a proxy label.** Manual review decisions arrive faster than chargebacks and are biased, but they are still the earliest signal that the score distribution has moved.

None of these require a specific vendor or platform. They require that the decision path emits structured events, which is an argument for the three-file shape above: a pure gate, a model wrapper that returns its features, and a policy layer that stamps the final decision with its source.

## The advanced version

Once the basic three-layer architecture is in place, three refinements matter.

**Shadow scoring.** Run the model in parallel with the live decision without acting on its output, and log both. This lets you evaluate a new model or a new threshold against real traffic before it affects anyone. It also gives you the counterfactual data you need to answer "what would have happened if we had blocked at 0.8 instead of 0.9".

**Feedback loops and label delay.** Fraud labels arrive late — a chargeback may land 60 to 120 days after the transaction. This means the model is always training on a partially-labeled past. The practical implication is that you need a strategy for the unlabeled recent window: either exclude it, or use a proxy label (manual review outcomes, customer complaints) with the understanding that it is biased. Teams that ignore label delay end up with a model that is excellent at detecting last quarter's fraud.

**Adversarial drift.** Fraudsters adapt to your rules. A rule that fires today may be neutralized tomorrow by a small change in behavior. This is an argument for keeping the rules gate small and monitoring rule hit rates. It is also an argument for not putting adaptive signals in the rules gate, where they are easy to observe and reverse-engineer.

## Quick reference

| Concern | Rules engine | Scoring model |
|---|---|---|
| Encodes | Policy, legal requirements, known-bad lists | Statistical pattern |
| Output | Boolean + reason code | Calibrated probability |
| Latency | Microseconds to low milliseconds | Milliseconds to tens of milliseconds |
| Changes when | Policy changes | Pattern drifts |
| Owner | Risk / compliance / product | Data science / ML engineering |
| Failure mode | Rule sprawl, contradictions | Silent drift, uncalibrated scores |
| Audit story | Named rule fired | Score + top contributing features |
| Where it sits | Front gate (short-circuit) | Behind the gate, feeds policy layer |

**The boundary rule of thumb:** if you would be uncomfortable explaining the condition to a regulator, a customer, or a journalist, it belongs in the rules gate. If it is a statistical regularity you would struggle to write down precisely, it belongs in the model. Everything in between is a judgment call, and the judgment should be made once, documented, and revisited on a schedule.

**The three files:** `rules_gate.py` (pure, deterministic, no I/O), `scorer.py` (model wrapper, returns score + features), `policy.py` (the only place thresholds live). If your system does not have an equivalent of `policy.py`, that is the gap.

## Frequently asked questions

**How do I decide which rules go in the hard-block gate?**

Ask three questions. Is the condition a matter of policy rather than pattern? Is the cost of a false positive acceptable given the legal or compliance exposure of a false negative? And would the condition still be correct if the fraud landscape changed completely tomorrow? If all three are yes, it belongs in the gate. If any is no, it is a signal for the model. The gate should be small — a handful of rules, not dozens.

**Why does my model perform well offline but poorly in production?**

The three usual causes, in order of frequency: label leakage (a feature that encodes the answer, often a timestamp or an ID that correlates with the label), distribution shift (training data does not match live traffic), and uncalibrated scores (the ranking is fine but the absolute probabilities are wrong, so thresholds behave unexpectedly). Check calibration first, because it is the easiest to verify and the most common cause of "the model is bad" when the model is actually fine and the thresholds are wrong.

**Can I run the model before the rules instead?**

You can, and for some systems it is correct — particularly when the model is cheap and the rules are expensive (for example, rules that require an external API call). But if any of your rules are genuinely non-negotiable, running the model first means you pay the model's latency and cost on transactions that will be blocked anyway. For most payment systems, the gate-first arrangement is cheaper and the latency budget is the binding constraint.

**How often should I retrain the model?**

It depends on how fast the pattern moves, which you can measure by tracking score distribution drift over time. A practical starting point is to retrain on a fixed cadence and additionally trigger a retrain when the score distribution shifts beyond a threshold you define. The cadence matters less than having monitoring that tells you when the cadence is wrong.

**How do I get started without a labeled dataset?**

Begin with the gate and the policy layer, and log every decision with structured events. That logging becomes the training data for the first model, and shadow scoring lets you evaluate that model before it acts. The architecture does not depend on having a model on day one; it depends on having a place where the model will eventually go.

## Your next 30 minutes

Open your fraud decision code and find the single place where a transaction becomes a block, a hold, or an allow. If that logic is spread across multiple files, handlers, or database triggers — if there is no single policy module — that is your gap. Write down every condition currently in that path and mark each one as "policy" or "pattern". The pattern ones are the candidates for a scoring layer, and the policy ones are your gate. You do not need to build the model today; you need to know where the boundary is.
