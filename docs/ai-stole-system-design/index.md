# AI stole system design

## The symptom and the root cause

A candidate's system design document reads like a vendor whitepaper: multi-agent orchestration, embedding caches, vectorized state machines. The diagram is clean, the service names are spelled correctly, and every latency figure is a round number. Then the interviewer asks a follow-up — how would you shard that cache, what happens to replication lag under write pressure, why this database and not the other one — and the answer stalls. The candidate says the vector database handles sharding automatically. The interview is effectively over.

The surface symptom is polish. The root cause is that the candidate never built a model of the system; they retrieved one. Large language models are good at producing plausible architecture prose because architecture prose is abundant, well-structured, and rarely falsifiable at a glance. A prompt like "design a multi-region event streaming system" returns something that looks like the first page of an engineering blog post. That output is not a design. It is a prior over design vocabulary.

The reason this breaks interviews is structural, not moral. System design interviews were always an oral exam in disguise: the artifact on the whiteboard is secondary, and the graded signal is the candidate's ability to defend trade-offs under changing constraints. Generated documents carry no such signal, because the generator cannot answer "what if the primary region goes down" — only the candidate can, and only if they actually reasoned about it.

## What AI output looks like under pressure

Generated designs share recognizable failure signatures. None of these is proof on its own; treat them as probes.

**Round numbers everywhere.** Human capacity estimates involve arithmetic with awkward inputs. A generated design tends to state clean figures without showing the derivation. Ask where a number came from. A real candidate will reproduce the multiplication; a reciting candidate will restate the number.

**Services that do not exist.** Models hallucinate plausible product names, especially in cloud catalogs where the naming conventions are consistent enough to extrapolate. A candidate confidently naming a service you have never heard of is a strong signal to ask them to open the docs.

**No failure modes.** Real designs carry scars: the retry storm, the hot partition, the cache stampede after a deploy. Generated designs describe the happy path in detail and the failure path in generalities.

**Missing units and boundaries.** "1M ops/sec" without stating per node, per cluster, read or write, and at what payload size is a number-shaped decoration.

**Instant fluency on the diagram, silence on the mechanism.** The candidate can describe what the boxes are but not what happens between them during a partial failure.

## A worked example: what a defensible answer contains

Take a common prompt: "Design a read-heavy service with a 10 TB dataset, 100 MB/s of writes, and a 5 ms read latency target." The exact numbers matter less than the reasoning chain, which should sound something like this.

**Step 1 — separate the two workloads.** Writes at 100 MB/s are a streaming problem; reads at 5 ms are a serving problem. They usually want different storage, so the first design decision is where the write path ends and the read path begins.

**Step 2 — size the write path.** 100 MB/s is 8.64 TB/day if sustained (100 MB/s × 86,400 s = 8,640,000 MB ≈ 8.64 TB). That exceeds the stated 10 TB dataset per day, which means either the dataset is replaced continuously or the write figure is a burst, not a steady state. A candidate who notices this inconsistency is reasoning; one who does not is reciting. Resolving it changes the design: a steady 8.64 TB/day implies retention and compaction policy as first-class concerns, not an afterthought.

**Step 3 — size the read path.** 5 ms is aggressive for a cold read from object storage, whose first-byte latency is typically tens to hundreds of milliseconds. So the read path needs either a cache with a high hit rate or a database with local storage. The choice depends on the read pattern, which the candidate should ask about rather than assume.

**Step 4 — name the trade-off.** Caching gives latency but introduces invalidation and staleness. A replicated database gives consistency but costs write amplification and cross-region traffic. The candidate should state which they chose, what it costs, and what they would monitor to know if the choice was wrong.

**Step 5 — describe the failure.** If the cache tier is lost, what happens to the database? If the answer is "it falls over," the design needs a load-shedding or admission-control story. This is the part generated output almost never contains.

The value of the exercise is not the specific architecture. It is watching whether the candidate can move between a number, a constraint, and a decision, and revise when the interviewer changes one input.

## How to measure whether a probe works

Do not trust a single interview's impression. Instrument the process.

- **Tag every interview** with whether a live-mutation probe was used and whether the candidate revised their design in response.
- **Record the probe question and the candidate's first three sentences.** Review them later against the rubric. This is the cheapest way to find out whether your probes discriminate or just add noise.
- **Compare pass rates on the written stage versus the live stage.** If a large fraction of candidates who pass the document review fail the live probe, the document review is not measuring what you think it measures.
- **Track inter-rater agreement.** Two interviewers scoring the same transcript should land in the same band. If they do not, the rubric is too vague to be useful.

A concrete instrument: for each candidate, log the number of distinct constraints they asked about before proposing a design, the number of times they revised after a mutation, and whether they named a failure mode unprompted. These are countable and cheap to collect. Trends across a hiring loop are more informative than any single anecdote.

## Redesign the loop around live debugging

The most reliable filter is to stop asking candidates to produce a design and start asking them to repair one. Repair requires a mental model; generation does not.

A workable 45-minute format:

1. **Ten minutes — orient.** Give the candidate a running system and a one-paragraph description. Let them read it and ask questions.
2. **Twenty minutes — break it.** Introduce a failure: stale reads after a write, a hot partition, a retry storm, a certificate expiry. Ask them to diagnose and fix it while narrating.
3. **Fifteen minutes — extend it.** Change one constraint — double the write rate, add a second region, cut the budget — and ask what breaks first.

The candidate needs a real environment for this. A local cluster is enough:

```yaml
# docker-compose.yml — three-node Redis Cluster for interview use
services:
  redis-1:
    image: redis:7.2
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --port 6379
    ports: ["7001:6379"]
  redis-2:
    image: redis:7.2
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --port 6379
    ports: ["7002:6379"]
  redis-3:
    image: redis:7.2
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --port 6379
    ports: ["7003:6379"]
```

Bring it up with `docker compose up -d`, then form the cluster with `redis-cli --cluster create` across the three published ports. A three-node cluster with no replicas is sufficient for a debugging exercise and keeps setup under a minute.

Then give the candidate a failing test. A minimal reproduction in Python:

```python
import redis

r = redis.RedisCluster(host="localhost", port=7001, decode_responses=True)

r.set("counter", "1")
first = r.get("counter")
r.set("counter", "2")
second = r.get("counter")

assert second == "2", f"stale read: got {second!r} after writing 2"
```

If the assertion fires, the candidate has something real to chase: which node served the read, whether the client is routing to the right slot, whether a replica lagged, whether the key moved after a reshard. None of these can be answered by reciting a blog post.

## Audit AI output instead of banning it

Banning assistants is both unenforceable and beside the point; the job increasingly involves reviewing generated code. A better format is to hand the candidate generated output and grade the review.

Give them a prompt and its result: "Generate a Terraform module for a Redis Cluster with TLS." Then ask:

- Which resources are created, and what does each one cost?
- Where are the credentials stored, and who can read them?
- What happens on a node replacement — is data lost, and is that acceptable?
- Which parts of this module would you refuse to merge, and why?

This tests the skill that matters in practice: reading generated infrastructure critically. A candidate who can spot an over-broad IAM policy or a missing encryption-at-rest setting is demonstrating real knowledge. A candidate who cannot has simply moved the failure one step later in the pipeline.

| Interview format | What it tests | What it misses |
|---|---|---|
| Static design document | Vocabulary, structure, recall | Whether the candidate can defend or revise the design |
| Live whiteboard design | Reasoning under mild pressure | Whether the candidate can operate a real system |
| Live debugging on a local cluster | Diagnosis, tooling, mental model of failure | Breadth of architecture knowledge |
| Audit of generated output | Critical review, security judgment | Original design ability |

No single format covers everything. A loop that uses only one of these will have a predictable blind spot.

## Failure modes of the fix itself

Redesigning interviews introduces its own problems, and naming them is part of doing it well.

**The probe becomes a trivia quiz.** If the live exercise rewards memorizing a specific command, it selects for the same recall the old format did. Keep the system small enough that the candidate can reason about it from first principles.

**Setup time eats the interview.** A cluster that takes fifteen minutes to start leaves twenty for actual signal. Pin the image version, pre-build the environment, and verify it before the candidate joins.

**The interviewer talks too much.** The temptation to hint is strong. Hints reduce signal; a candidate who solves the problem with three hints has demonstrated something different from one who solved it alone. Log the hints given.

**False positives from nervousness.** A candidate who freezes under observation may still be competent. Separate the diagnostic score from the communication score, and weight them explicitly rather than blending them into one impression.

**The rubric drifts.** Without calibration sessions, two interviewers will interpret "good diagnosis" differently within a month. Review a shared transcript periodically and re-agree on the bands.

## FAQ

**Why does AI-generated design pass document review?**
Because document review grades the artifact, and the artifact is exactly what a language model produces well. The signal that matters — defending trade-offs under changing constraints — is absent from a static document, so the review cannot see its absence.

**Is it fair to ask candidates to debug live if the role is design-heavy?**
Yes, if the exercise is scoped to reasoning rather than tooling. The goal is not to test whether someone remembers a specific CLI flag; it is to test whether they can form a hypothesis about a broken system and check it. That skill transfers to design work.

**Should candidates be allowed to use AI during the interview?**
Allowing it in a review capacity is more informative than banning it. Give the candidate generated output and grade the critique. Banning is hard to enforce remotely and tests nothing about the job, which increasingly involves auditing generated code.

**How many interviews does it take to know a probe works?**
More than a handful. Collect the probe result, the live-stage result, and the eventual hiring outcome for every candidate, then look for correlation across a full hiring cycle. A probe that does not correlate with later performance is adding cost without signal.

**What if the candidate has never used the tooling in the exercise?**
Choose tooling with a low floor: a single-node store, a shell, and a failing test. The exercise should fail candidates who cannot reason, not candidates who have not memorized a particular client library.

## The next 30 minutes

Open the rubric for your next system design interview and replace the "design quality" section with a 20-minute live debugging exercise. Start the three-node cluster above with `docker compose up -d`, form it with `redis-cli --cluster create`, and run the stale-read test until you can reproduce the failure yourself. If you cannot reproduce it, you cannot grade it — and neither can the candidate.
