# AI model poisoning: 3 real attacks on datasets

## Why standard supply chain guidance falls short for AI

Most AI supply chain guidance is written for tutorial repositories: pin your dependencies, sign your commits, audit your models. Those steps matter, but they describe a world where the artifact is a wheel or a container image. An AI pipeline also ships two artifacts that traditional tooling barely understands: a dataset and a set of learned weights. Neither has a package manager with a trustworthy index, and neither fails loudly when it has been tampered with.

The result is a predictable gap. A team can have a fully pinned Python environment, a locked container base image, and a clean dependency graph, and still deploy a model that quotes an attacker-supplied revenue figure to customers. The poisoned bytes entered through the dataset, which was never in the dependency graph to begin with.

This article covers three attack patterns that show up repeatedly in AI pipelines, the failure modes that appear once you start defending against them, and a set of controls you can implement with ordinary tooling. It is written for teams already in production, where the interesting problems are rollback speed, blast radius, and the difference between a hash that matches and a dataset you can actually trust.

## Three attack patterns and how they work

Attacks on AI supply chains generally land at one of three layers: the data, the model weights, or the software dependencies that load them. Each layer has different detection properties, so it is worth understanding them separately before designing controls.

### Data poisoning through pull requests

The most common pattern is a small, plausible edit to a dataset that lives in a Git repository. The attacker forks the repo, adds a handful of examples, and opens a pull request. The diff is small — often well under one percent of the file — and the examples look like ordinary question and answer pairs until you inspect the label field closely.

A representative shape of this attack: a document Q&A bot is fine-tuned on a dataset exported from an internal wiki. An attacker adds a few hundred examples where the answer to a financial question is a fabricated figure. The pull request passes review because reviewers look at the diff size and the general shape of the JSONL, not at whether the labels are consistent with the rest of the corpus. The poisoned dataset is merged, a model is fine-tuned, and the fabricated figure starts appearing in customer-facing answers.

The detection problem is that nothing in this chain is anomalous by conventional standards. The commit is signed. The CI passed. The file hashes match. The poison is semantic, not structural.

### Model poisoning through public model hubs

Public model hubs make it trivial to upload a checkpoint. They generally do not require provenance, and many download paths resolve a model by name and revision rather than by content hash. That combination allows two related attacks.

The first is a malicious checkpoint: a serialized weights file that executes code when loaded. Python's pickle-based formats are the usual vector, since unpickling can run arbitrary code. A checkpoint can inspect its inputs and behave normally except for a narrow trigger — a specific phrase, a specific token sequence — at which point it returns an attacker-chosen label.

The second is substitution: an attacker uploads a model under a name that collides with a popular one. If the consuming project pins a loose version range or omits the revision entirely, the download resolves to the attacker's artifact. The model card may even look correct, since model cards are descriptive text, not verified metadata.

A subtle version of this attack targets the tokenizer configuration rather than the weights. A change to tokenizer settings can shift how inputs are segmented, which changes predictions without changing a single weight. A diff that "only touches the tokenizer config" is not automatically benign.

### Dependency poisoning in the Python ecosystem

The third pattern is the one most familiar to anyone who has worked on software supply chains: a malicious package published to a public index. In AI stacks, the interesting part is the transitive depth.

Consider a project that adds an embedding library. That library depends on a tokenization library, which depends on a serialization library. A malicious wheel published under the name of the serialization library can execute code at import time, because the tokenization library imports it during startup. Nothing in the project's own dependency declarations mentions the poisoned package. The attack surface is the transitive graph, and the trigger is simply importing the embedding library.

The practical consequence is that pinning your direct dependencies is necessary but not sufficient. A lockfile that records hashes for every transitive dependency is the relevant control, and it only helps if the hashes come from a trusted source and are verified at install time.

What all three patterns share is a mismatch between the speed of AI development and the assumptions of traditional supply chain controls. A dataset can change daily. A model can be re-uploaded under the same name. A transitive dependency can be replaced between two builds of the same lockfile if the lockfile does not pin content hashes.

## A worked example: reasoning about blast radius

Before choosing controls, it helps to quantify what a single poisoned example can do. The arithmetic below is illustrative, using stated assumptions rather than measured results.

Suppose a fine-tuning run uses 50,000 examples. An attacker wants the model to produce a specific incorrect answer for a specific question. Assume, conservatively, that a single example contributes a negligible gradient signal, and that the attacker needs the target pattern to appear in at least 0.1 percent of the training set to reliably shift behavior. That is 50 examples.

Now suppose the dataset repository has 40 downstream forks and mirrors, a common outcome for a dataset that is useful enough to be reused. If the attacker's pull request is merged upstream, the poison propagates to all 40 copies at the next sync. If the attacker instead targets one downstream fork, the blast radius is one copy — but that copy may be the one a different team depends on.

The useful conclusion is not the specific number. It is that the cost of preventing the merge is far lower than the cost of remediating 40 downstream copies, and that the prevention control is a review gate plus a content hash, not a sophisticated detector. This is why the controls below emphasize process and provenance over classification.

To measure your own exposure, instrument the following:

- The number of distinct sources that feed your training dataset, and which of them are writable by more than one person.
- The time between a dataset commit and the next training run that consumes it. A short window means a poisoned commit reaches a model quickly.
- The number of downstream consumers of each dataset artifact. This is your blast radius if a poisoned version is published.
- The time to roll back the last three model deployments, measured from alert to traffic shift.

None of these require new tooling. Git history, your CI logs, and your deployment records contain all of it.

## Controls that actually reduce risk

The controls below are ordered by the ratio of risk reduction to implementation cost. Start at the top.

### Pin dependencies by content hash

A lockfile that records a version number is not a lockfile. Use a lockfile that records a cryptographic hash for every package, including transitive ones, and configure your installer to verify those hashes and fail on mismatch. For Python, this means a fully resolved lockfile with hashes, installed in a clean environment in CI. The goal is that a build either reproduces exactly or fails loudly.

### Treat datasets as versioned artifacts

A dataset should have a content hash and a Git commit hash recorded together, and both should be checked before a training run starts. The hash proves the bytes; the commit hash proves which reviewed revision produced them.

```python
# dataset/verify_dataset.py
import hashlib
import sys
from pathlib import Path

def verify_dataset(path: Path, expected_hash: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_hash:
        print(f"Hash mismatch: {actual} != {expected_hash}")
        return False
    print("Dataset integrity verified")
    return True

if __name__ == "__main__":
    dataset = Path("data/train.jsonl")
    expected = "a1b2c3..."  # from a trusted, versioned source
    if not verify_dataset(dataset, expected):
        sys.exit(1)
```

Store the expected hash and the expected commit hash in a file that is itself signed, so that an attacker who compromises the repository cannot simply update the expected values alongside the poisoned data. A signed provenance file that is regenerated automatically in CI removes the most common failure mode, which is a stale hash that silently accepts a new dataset.

```python
# dataset/verify_commit.py
from pathlib import Path
from git import Repo

def verify_commit(dataset_path: Path, commit_hash: str) -> bool:
    repo = Repo(".")
    if repo.head.commit.hexsha != commit_hash:
        print(f"Commit mismatch: {repo.head.commit.hexsha} != {commit_hash}")
        return False
    print("Commit verified")
    return True
```

### Require review for dataset changes

A code owner rule that requires approval for changes under your dataset directories is crude, and it will not stop a determined insider. It does stop opportunistic attacks, which are the majority. Require at least two approvals for dataset changes, and make sure the reviewers understand that the diff size is not a signal of safety.

```
# .github/CODEOWNERS
# All dataset files require review by the ML team
/data/**/*.jsonl @ml-team
/data/**/*.parquet @ml-team
```

### Prevent code execution during model loading

Loading a serialized model can execute arbitrary code. Where your framework supports it, prefer formats that are not executable — safetensors is the common choice for weights — and set the loader to refuse pickle-based formats. Where you must load an executable format, isolate it.

The seccomp example below restricts the syscalls available to the process before the model is loaded. Treat it as illustrative: syscall numbers and the exact set of calls a loader needs vary by platform, and a filter that is too strict will cause hard-to-diagnose failures.

```python
# model/sandbox.py
import ctypes

# Load libseccomp
libseccomp = ctypes.CDLL("libseccomp.so.2")

# Define a filter that kills the process on any disallowed syscall
libseccomp.seccomp_init.restype = ctypes.c_void_p
scmp_filter_ctx = libseccomp.seccomp_init(libseccomp.SCMP_ACT_KILL)

# Allow a minimal set of syscalls. Extend as your loader requires.
libseccomp.seccomp_rule_add.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
for syscall in ("read", "mmap", "mprotect", "brk", "exit_group"):
    libseccomp.seccomp_rule_add(
        scmp_filter_ctx,
        libseccomp.SCMP_ACT_ALLOW,
        libseccomp.SCMP_SYS(syscall),
        0,
    )
libseccomp.seccomp_load(scmp_filter_ctx)

# Now load the model
from transformers import AutoModel
model = AutoModel.from_pretrained("./model")
```

The practical alternative, if seccomp is too brittle for your environment, is to load the model in a separate process with no network access and a read-only filesystem, and to communicate with it over a pipe. This is easier to reason about than a syscall filter and catches the common case of a checkpoint that tries to open a network connection.

### Make rollback a tagged operation

Rollback speed is the control that determines how long a poisoning incident affects customers. Tag datasets and models after each training run, and keep a rollback path that does not depend on rebuilding.

```bash
# Tag the dataset and model after each training run
git tag -s dataset/v1.2.3 -m "Dataset v1.2.3"
git push --tags

# Push the model to a private registry with the tag
huggingface-cli upload my-org/my-model model-v1.2.3

# Rollback script
#!/bin/bash
set -e
TAG="$1"
git checkout "dataset/${TAG}"
huggingface-cli download my-org/my-model "${TAG}" --local-dir model
```

Use signed tags. An unsigned tag can be moved by a force push, and a rollback script that points at a moved tag will deploy the wrong revision. Verify the signature in CI before the rollback proceeds.

## Failure modes that appear after you add controls

Every control introduces its own failure modes. These are the ones that show up most often.

**Stale provenance files.** A dataset is updated, but the signed hash file is not regenerated. The new dataset fails verification, or worse, an old hash is compared against an old file and passes while the new file goes unchecked. Generate the provenance file in CI as part of the same commit that changes the dataset, and fail the build if the two are out of sync.

**Signature lifetime and revocation.** Short-lived signing certificates can expire before an artifact is consumed, especially if a build is slow or a deployment is delayed. A revoked key will fail builds even for artifacts that are genuinely valid. Decide explicitly how long an artifact must remain verifiable, and design the signing step so that verification happens close to consumption.

**Over-restrictive sandboxes.** Syscall filters break legitimate loaders. A loader that uses memory-advice calls for performance will silently lose that performance, or fail outright, if the filter does not allow them. Debug with a syscall tracer before tightening a filter, and keep the filter in version control with a comment explaining each allowed call.

**Registry substitution.** Public model hubs do not enforce provenance, and a model can be replaced under the same name. Pin the revision explicitly and verify the content hash against a value you obtained from a trusted source, not from the same hub that served the model.

**Mutable tags.** Git tags can be moved. If your rollback depends on a tag, an attacker who can force-push can redirect your rollback to a poisoned revision. Signed tags plus signature verification in CI close this gap.

**Detector blind spots.** Validation scripts that check for empty fields and length limits will not catch a semantically poisoned label. Add checks for label consistency — for example, flag any question in a Q&A dataset that has only one distinct answer across many examples — and for input/output length ratios that fall outside the distribution of the rest of the corpus.

## A decision checklist

Use this to decide which controls to implement first, based on your situation.

| Situation | First control | Why |
|---|---|---|
| Dataset is editable by more than one person | Required review plus signed provenance file | The merge gate is the cheapest place to stop a poisoned example |
| Model loaded from a public hub | Pinned revision plus content hash verification | Prevents substitution under a colliding name |
| Direct dependencies pinned, transitive not | Hash-pinned lockfile | Closes the transitive dependency gap |
| No rollback path faster than a rebuild | Signed tags plus a tested rollback script | Determines customer impact duration |
| Model loader accepts pickle formats | Switch to non-executable weight formats, or isolate the loader | Removes code execution from the load path |

## When this approach is the wrong choice

These controls assume a pipeline you operate. They are a poor fit in several common situations.

If you do not have a model registry and models move between machines by hand, provenance tooling will not help. Establish a registry and a deployment path first.

If you do not have CI, there is nowhere to enforce a verification step. Fix the build pipeline before adding signing.

If you consume models through a hosted API, you cannot sign the artifact. Focus instead on input validation, output monitoring, and logging enough to detect a behavior change.

If you cannot maintain the pipeline, a brittle set of controls is worse than a simple one. A hash check that is regenerated automatically is maintainable; a hand-edited provenance file is not.

## FAQ

**How should a model from a public hub be verified before use?**

Pin the revision explicitly, download the artifact, and compare its content hash against a value you obtained from a trusted source. Do not treat the hub's own metadata as the trusted source. Prefer non-executable weight formats, and refuse to load pickle-based checkpoints from untrusted origins.

**What is the fastest way to roll back a poisoned model?**

Tag models and datasets after each training run using signed tags, and keep a rollback script that checks out the tagged dataset and pulls the tagged model. Measure the time from alert to traffic shift, and treat any rollback that requires a rebuild as a gap.

**Are SBOMs sufficient for AI supply chains?**

No. An SBOM describes software dependencies. It does not capture dataset provenance or model signatures. Use an SBOM for the software layer and a separate provenance mechanism for datasets and weights.

**How do you detect poisoned examples in a dataset?**

Look for label inconsistencies, input and output length outliers, and unexpected tokens. A simple check that flags any question with only one distinct answer across many examples catches a common class of injection. Run the check as a build step so a poisoned commit fails before training starts.

**Does pinning a version number protect against dependency poisoning?**

Only if the pin resolves to a content hash that is verified at install time. A version number alone can be satisfied by a different artifact if the index serves a replacement.

## What to do in the next 30 minutes

Run this command in the repository that produces your training data, and read the output:

```bash
git log --format='%h %an %ad %s' --date=short -- data/ | head -20
```

For each commit that touched a dataset file, ask two questions: was this change reviewed by someone other than the author, and can you produce the content hash that a training run would have verified against? If the answer to either is no, you have found the highest-value gap in your pipeline. Write it down, and make the first fix a required review plus an automatically generated hash file for that dataset directory.
