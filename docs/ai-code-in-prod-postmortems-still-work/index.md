# AI code in prod? Postmortems still work

## Why standard postmortems miss AI-generated code

A postmortem template written for hand-written code assumes the reviewer can reconstruct intent from the diff. That assumption breaks when the diff was produced by a language model. The code often looks idiomatic, passes review, and behaves in ways nobody on the team explicitly chose.

The failure mode is not "the AI wrote bad code." It is that the generated code encodes assumptions that were never written down: which runtime version is available, whether a cache is warm, how many retries are acceptable, what a "reasonable" batch size is. When one of those assumptions is wrong, the stack trace points at the last hand-written frame, not at the decision that caused the problem.

A second failure mode is silent drift. If the model identifier is a floating alias rather than a pinned version, the same prompt can produce different code next week. The incident review then argues about behavior that no longer reproduces.

The practical response is to treat the model as a dependency with its own version, configuration, and inputs, and to capture those artifacts at merge time rather than during the incident. Everything below follows from that one idea.

## What to capture, and why each artifact matters

Five artifacts cover most AI-related incidents:

1. **The generated diff, in full.** Not the summary from `git show --stat`, but the actual added lines. Large insertions are the ones most likely to hide architectural decisions.
2. **The exact prompt text.** Prompts are frequently stored in a file, an environment variable, or a template rendered at runtime. The rendered prompt is the one that matters.
3. **The model identifier and version.** Whatever your provider exposes: a dated snapshot name, a deployment name, a revision hash. If the provider offers no pinning, record the identifier string verbatim and note that it is not pinned.
4. **Runtime assumptions.** Language runtime version, framework version, and the values of any limits the generated code depends on: connection pool sizes, timeout values, retry budgets, batch sizes.
5. **External state the model consulted**, if any. Retrieved documents, tool outputs, or feature-flag values that were part of the generation context.

The reason to store these at merge time rather than at incident time is simple: prompts and environment variables change. By the time an incident is under investigation, the prompt that produced the offending code may already have been edited.

## A minimal collection script

The script below collects the artifacts above into a single JSON record. It uses only the standard library plus `redis` for storage and `pydantic` for validation, so it runs anywhere Python runs.

```bash
pip install redis==5.0.1 pydantic==2.6.4
```

```python
import json
import os
import subprocess
from pathlib import Path

import redis
from pydantic import BaseModel, Field


class PostmortemArtifacts(BaseModel):
    commit_hash: str
    changed_files: list[str] = Field(default_factory=list)
    generated_diff: str
    prompt_text: str
    prompt_source: str
    model_identifier: str
    model_pinned: bool
    env_assumptions: dict
    external_context: str | None = None


def _run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    ).stdout.strip()


def collect(commit_hash: str) -> PostmortemArtifacts:
    # 1. Files touched by this commit, with insertion counts.
    numstat = _run(["git", "show", "--numstat", "--format=", commit_hash])
    changed_files = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].endswith(".py"):
            changed_files.append(parts[2])

    # 2. Full diff for those files.
    generated_diff = _run(
        ["git", "show", commit_hash, "--"] + changed_files
    ) if changed_files else ""

    # 3. Rendered prompt. Prefer a file; fall back to the environment.
    prompt_path = Path(os.getenv("AI_PROMPT_PATH", "prompts/ai_refactor.txt"))
    if prompt_path.exists():
        prompt_text = prompt_path.read_text()
        prompt_source = str(prompt_path)
    else:
        prompt_text = os.getenv("AI_PROMPT", "")
        prompt_source = "environment:AI_PROMPT"

    # 4. Model identity. A dated snapshot name is treated as pinned;
    #    a bare alias is recorded but flagged as unpinned.
    model_identifier = os.getenv("AI_MODEL_ID", "unknown")
    model_pinned = any(ch.isdigit() for ch in model_identifier)

    # 5. Runtime assumptions that generated code tends to depend on.
    env_assumptions = {
        "python_version": _run(["python", "--version"]),
        "node_version": _run(["node", "--version"]),
        "redis_version": _run(["redis-server", "--version"]),
        "db_pool_size": os.getenv("DB_POOL_SIZE", "unset"),
        "retry_budget": os.getenv("RETRY_BUDGET", "unset"),
        "request_timeout_s": os.getenv("REQUEST_TIMEOUT_S", "unset"),
    }

    return PostmortemArtifacts(
        commit_hash=commit_hash,
        changed_files=changed_files,
        generated_diff=generated_diff,
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        model_identifier=model_identifier,
        model_pinned=model_pinned,
        env_assumptions=env_assumptions,
        external_context=os.getenv("AI_RETRIEVAL_SNAPSHOT"),
    )


if __name__ == "__main__":
    import sys

    commit = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    artifacts = collect(commit)

    client = redis.Redis(host="localhost", port=6379, db=0)
    key = f"postmortem:{artifacts.commit_hash}"
    client.set(key, artifacts.model_dump_json(), ex=7 * 24 * 60 * 60)
    print(f"stored {key}")
```

Two details are worth calling out. First, `git show --numstat` gives insertion and deletion counts per file, which is a cheap way to find the large generated insertions that a normal review skims past. Second, the `model_pinned` flag is deliberately crude: it checks for a digit in the identifier, which matches dated snapshot naming conventions but not every provider's scheme. Adjust it to your provider's convention, or drop it and record the raw string.

## Wiring it into CI

Run the collector on every merge to the main branch, passing the merge commit hash:

```bash
python postmortem.py "$GITHUB_SHA"
```

A few practical notes on the storage layer:

- Redis with a TTL is convenient because it requires no schema and expires old artifacts automatically. Seven days is a reasonable default; extend it if your incident reviews routinely lag.
- If Redis is not already in the stack, a directory of JSON files committed to a separate repository works just as well and has no operational cost.
- Do not store the artifacts in the same repository as the code. A postmortem record that includes a prompt and a diff will grow the clone size over time.

The cost of this pipeline is dominated by the diff size, not the metadata. For a typical commit touching a few files, the stored record is small. If a single commit touches thousands of generated lines, consider storing the diff compressed and keeping only the file list and insertion counts in Redis.

## How to measure whether this helps

Do not adopt a tool like this on faith. Define the measurement before you deploy it, and compare two windows of equal length.

Instrument these three things:

1. **Time from first alert to first correct hypothesis.** This is the number that actually moves when you have the prompt and model version available. Record it in the incident ticket, not in a dashboard.
2. **Incidents where the root cause was an assumption mismatch** rather than a logic bug: runtime version, pool size, timeout, retry budget. Count them per month. If this number is near zero, the pipeline is not earning its keep.
3. **Artifact availability rate.** For each incident, did the artifacts exist and match the deployed commit? A pipeline that runs on merges but not on hotfixes will miss exactly the incidents that matter most.

To compare, take the last N incidents before adoption and the first N after, and compute the median of measure 1 for each group. Medians are more robust than means here because incident durations are heavy-tailed. If the medians are within noise of each other, the artifacts are not being used during incidents, which is a process problem, not a tooling problem.

## Failure modes to design against

**Prompt drift between environments.** A prompt file that is mounted differently in staging and production means the reviewed behavior is not the deployed behavior. The fix is to copy the prompt into the image at build time and record its hash in the artifacts. If the hash in the artifact record does not match the hash in the running container, the review is invalid.

**Unpinned model identifiers.** If the provider accepts only a floating alias, the same prompt can produce different code after a provider-side update. Record the alias, mark it unpinned, and treat any behavioral change in generated code as a candidate cause when no other change explains it. Where the provider supports dated snapshots, pin them.

**Generated code that changes control flow invisibly.** A model asked to "add retries" may restructure a synchronous call path into an asynchronous one. The diff shows the change, but the review does not, because the added lines are numerous and idiomatic. This is the strongest argument for storing full diffs rather than summaries.

**Retry and timeout values that are syntactically valid but operationally absurd.** A generated retry loop with a backoff base very close to 1 will effectively retry forever. A circuit breaker with a threshold of a few milliseconds will trip on every call. Neither is a syntax error. A cheap guard is a post-generation lint rule that flags numeric literals in retry, backoff, and timeout positions outside a configured sane range.

**Artifact bloat.** Storing full embeddings alongside the record is usually unnecessary. Store identifiers and metadata; the vector store itself is a cache that can be rebuilt, while the metadata is the evidence.

## When not to build this

The pipeline is worth building when generated code makes decisions that affect availability: retry policy, concurrency limits, cache eviction, batch sizing, connection management. It is not worth building when generated code is confined to leaf functions with no control-flow authority, such as data-format converters or thin API clients.

Three conditions make it a poor fit:

- **No version control for the generation inputs.** If prompts are edited in a web console with no history, the artifacts cannot be reconstructed after the fact. Fix that first.
- **No model versioning from the provider.** You can still record inputs and outputs, but you cannot reproduce a generation. The value drops to correlating behavior changes with timestamps.
- **An unstable platform.** If the team is already firefighting infrastructure, adding an artifact pipeline increases surface area without reducing toil. Stabilize first.

A useful rule of thumb: if you cannot name a decision the generated code made that a reviewer did not consciously approve, you do not yet need this pipeline. Once you can name one, you do.

## A worked example of the reasoning

Suppose an incident report says: "Checkout latency spiked; the stack trace points to `process_order`, which is hand-written."

The generated code sits below `process_order`. The diff for the relevant commit shows a new helper that wraps each line item in an `await` inside a loop. The prompt, recovered from the artifact store, reads: "Ensure each line item is validated independently before persistence." Nothing in that prompt says "sequentially." The model chose sequential awaits because the instruction emphasized independence.

The runtime assumptions recorded at merge time show a database pool size of ten and a per-query timeout of two seconds. With a hundred line items, the worst case is one hundred sequential round trips, bounded by the pool size. That arithmetic alone explains a latency spike without any code being "wrong."

The corrective action is not to patch the generated helper. It is to change the prompt to state a concurrency requirement, and to add a lint rule that flags `await` inside a loop over a collection whose size is not bounded by a constant. The postmortem artifact made that reasoning possible in minutes rather than hours, because the prompt and the pool size were both on hand.

## FAQ

**Is the prompt really necessary, or is the diff enough?**
The diff shows what changed; the prompt shows what was asked for. When a generated change is surprising, the prompt is what distinguishes "the model misinterpreted a clear instruction" from "the instruction was ambiguous." Those two findings lead to different fixes.

**What if the provider does not offer pinned model versions?**
Record the identifier string and mark it unpinned. Capture the model's inputs and outputs around generation time if your integration allows it. Treat behavioral changes in generated code as a candidate cause whenever no code or prompt change explains an incident.

**How large do the artifacts get?**
For a commit touching a handful of files, the record is small enough to store in Redis without concern. The exception is very large generated diffs, which are better stored compressed or as a file list plus insertion counts.

**Does this replace normal observability?**
No. Traces, metrics, and logs remain the primary signal. The artifacts answer a different question: given that something behaved unexpectedly, what were the conditions under which the code was written?

**What if the team does not use Git?**
Then the diff artifact must be produced another way, for example by comparing the deployed file against the last known-good copy. The other four artifacts are independent of version control.

## Do this in the next 30 minutes

Pick the most recent commit that introduced generated code. Run `git show --numstat --format= <commit>` on it and note which files gained the most lines. Open the largest one and find a single numeric literal that controls a timeout, retry count, or batch size. Then check whether that number matches the value configured in production. If it does not, you have found a concrete assumption mismatch, and you have also found the first field your postmortem artifacts need to record.
