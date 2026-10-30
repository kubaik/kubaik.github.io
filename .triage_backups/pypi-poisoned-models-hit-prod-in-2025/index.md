# PyPI poisoned models hit prod in 2025

A recurring failure pattern in ML supply chains is that the default configuration looks fine until it isn't. `pip install` without flags, version pins on major releases only, and datasets loaded straight from a hub are all convenient defaults that assume the package and the data are what they claim to be. That assumption held when the worst case was a typo in a package name. It does not hold when the package or dataset is itself the attack surface.

## The gap between documented install behavior and what training pipelines need

Package indexes are usually treated the way container registries were treated years ago: as a trusted source. The documented workflow is to install without flags and pin loosely. That is reasonable advice for a web service whose blast radius is a single deploy. It is weaker advice for a training pipeline, because the artifact you install ends up shaping model weights that may live for months.

The documented behavior of `pip` is the key detail. If a requirement is pinned by version only, `pip` resolves that version from the index and installs whatever file the index serves for it. It does not compare the file against a checksum you chose, because you never provided one. The `--require-hashes` mode changes this: when hashes are present in the requirements file, `pip` refuses to install anything that does not match, and refuses to install any requirement that lacks a hash. That mode is the mechanism worth building on.

The same logic applies to datasets. A hub URL that resolves to a branch or a tag is a moving target. A URL that resolves to a commit hash is not. The difference is one parameter, but it is the difference between a reproducible artifact and a pointer that someone else can redirect.

The part that surprises people is the failure signature. A poisoned model does not have to crash. It can pass unit tests, pass an offline eval suite, and pass a smoke test on production traffic, because the payload is conditional. A logging hook that fires on a specific input shape, or a safety filter that is silently replaced by a pass-through when a particular config key is present, produces no exception and no stack trace. The observable symptom is a slow drift in output quality that gets misattributed to data drift.

## The mechanics of a poisoned dependency

The typical path has two variants: a name that is close to a popular library, or a takeover of an abandoned package. Both rely on the same property of the install step, which is that the package name is the only thing most pipelines check.

A package that mimics a well-known ML library can do several things without raising an error:

1. Monkey-patch a commonly imported class, for example wrapping a sequence-classification auto-model so that every input string is also written to a remote endpoint.
2. Add a config key that is not part of the upstream schema. If the loading code reads config keys dynamically, an unexpected key can flip a behavior, such as substituting a no-op for a safety filter.
3. Depend on the real library and re-export it, so imports resolve and the rest of the code behaves normally.

The third point is what makes detection hard. The package is not broken. It is a working wrapper with an extra side effect, and the side effect only triggers under conditions the attacker controls.

Dataset tampering follows the same shape. A dataset hosted on a hub can carry extra columns that the training script never intends to use but loads anyway, because most loaders materialize the whole file. A column of adversarial instructions mixed into a corpus teaches the model a trigger phrase. The model then behaves normally until that phrase appears, which is exactly when a dashboard would report everything as healthy.

## A worked example: how much does verification actually cost?

Before adopting hash verification, it is worth estimating the cost. The arithmetic below is illustrative, using round numbers so the reasoning is visible.

Assume a CI job that currently takes 240 seconds. Add two steps:

- Hash verification: download each pinned wheel and compute a digest. If a typical ML requirements file has 60 direct and transitive dependencies and each download-and-digest costs about 1 second on a warm cache, that is roughly 60 seconds.
- Dataset audit: read the training file and scan every column with a regex. If the file is 2 GiB of parquet and the scan runs at 200 MiB/s single-threaded, that is about 10 seconds, plus a few seconds of overhead.

Total added time: roughly 75 seconds on a 240-second baseline, which is about 31%. That is a real cost. The question is whether it is the right cost.

Now compare it against the cost of a single undetected incident. If a poisoned dependency ships, the remediation work is: identify every model trained during the exposure window, retrain from a clean pin, re-run evals, and re-issue any artifact that was published. For a pipeline that trains nightly, a one-week exposure window means seven model versions to invalidate. The verification step costs 75 seconds per build. The remediation costs days. The ratio is not close.

The honest caveat: the numbers depend entirely on your requirements file size and dataset size. Measure both before committing to the approach. Instrument the CI job with timestamps around each step, run it ten times, and compare the median rather than the mean, because a cold cache will dominate the first run.

## Implementing verification in the pipeline

Start with a lockfile that carries hashes. A common approach is a compile step that resolves the full dependency graph and writes digests alongside each pin:

```bash
pip install pip-tools
pip-compile requirements.in --generate-hashes --output-file requirements.txt
```

The generated file contains entries in the form `package==version --hash=sha256:...`. Installing from that file with `pip install --require-hashes -r requirements.txt` makes `pip` enforce the digests. If the index serves a different file for the same version, the install fails before any code runs.

The next layer is a check that the file you downloaded is the file you expected, independent of the installer. A small script that downloads to a cache directory and compares digests is enough:

```python
import hashlib
import pathlib
import subprocess

def verify_package(name: str, version: str, hashes: list[str], mirror_url: str) -> bool:
    dest = pathlib.Path(f".cache/{name}-{version}.whl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pip", "download",
        f"{name}=={version}",
        "--no-deps",
        "-d", str(dest.parent),
        "--index-url", mirror_url,
    ]
    subprocess.run(cmd, check=True)
    actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
    return actual_hash in hashes
```

Note the shape of the return value: it compares against a list, because a package may legitimately have multiple valid wheels for different platforms and each has its own digest. A single-hash comparison will produce false mismatches on multi-platform builds.

Wire this into CI so it runs before training, not after:

```yaml
- name: Verify ML dependencies
  run: |
    pip install --require-hashes -r requirements.txt
    python scripts/verify_deps.py requirements.txt https://mirror.internal.example.com/pypi/simple
```

For datasets, pin the revision explicitly. Loaders that accept a `revision` parameter will fetch that exact commit rather than the default branch:

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "https://huggingface.co/datasets/example/news-summary/resolve/<commit-hash>/train-00000-of-00001.parquet",
    },
    revision="<commit-hash>",
)
```

Replace `<commit-hash>` with the actual commit SHA from the dataset repository. If the repository does not expose one, treat the dataset as unpinned.

Then audit the content. Scan every column, not just the text column, because adversarial content can hide in metadata fields:

```python
import re
import pandas as pd

ADVERSARIAL_REGEX = re.compile(
    r"(ignore previous instructions|new instructions below|system: repeat after me)",
    re.IGNORECASE,
)

def scan_value(value) -> bool:
    if isinstance(value, str):
        return bool(ADVERSARIAL_REGEX.search(value))
    if isinstance(value, (list, tuple)):
        return any(scan_value(item) for item in value)
    if isinstance(value, dict):
        return any(scan_value(item) for item in value.values())
    return False

df = pd.read_parquet("train-00000-of-00001.parquet")
flagged = {col: int(df[col].apply(scan_value).sum()) for col in df.columns}
if any(flagged.values()):
    raise ValueError(f"Dataset contains adversarial patterns: {flagged}")
```

Finally, add a load-time guard for the model itself. If the upstream config schema does not include a key your loader reads, refuse to proceed rather than silently accepting it:

```python
import logging
from transformers import AutoConfig, AutoModelForSequenceClassification

class SafeModel:
    def __init__(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "safety_checkpoint_id", None) is not None:
            logging.error("Unexpected safety_checkpoint_id in config; refusing to load")
            raise ValueError("safety_checkpoint_id detected; possible supply-chain tampering")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

The general principle behind this wrapper is worth stating plainly: any config key your code does not expect should be a hard failure, not a value that gets ignored.

## Measuring the effect of these controls

There is no universal benchmark for supply-chain controls, so measure your own. Three metrics are worth instrumenting.

**CI duration before and after.** Record the wall-clock time of the job that installs dependencies and runs the dataset audit. Run it ten times on a warm cache and compare medians. This gives you the real overhead number rather than the illustrative one above.

**Hash mismatch events per week.** Every mismatch is either a legitimate re-upload or an attack. Count them and log the package name, version, expected digest, and observed digest. A mismatch rate of zero over a month is a signal that your pins are stable. A nonzero rate tells you how often the ecosystem moves under your feet.

**Time from publish to install.** For each dependency, record how long it sat on the index before your lockfile picked it up. A short window means you are installing very fresh packages, which is higher risk. A deliberate delay of a few days before adopting a new version is a cheap control, because most malicious packages are discovered and removed within that window.

Do not invent a target number for any of these. Establish a baseline first, then decide whether the trend is acceptable.

## Failure modes that are easy to miss

**Hash mismatch noise.** Maintainers sometimes re-upload a wheel with the same version string but different metadata. With hashes enforced, that breaks the build without any code change on your side. The correct response is to treat it as a security event and investigate, not to disable hash checking. The incorrect response is to regenerate the lockfile automatically without looking.

**Single-column audits.** Auditing only the text field misses payloads in JSON metadata, headers, or auxiliary columns. Scan recursively across all columns and all nested structures, as in the `scan_value` function above.

**Stale model artifacts.** After adopting stricter loading, old unsafe model versions often remain in registries and cached containers. They can still be pulled by direct path. Enforce a retention policy and delete superseded artifacts rather than relying on nobody referencing them.

**Package caches in CI.** A compromised package installed once can survive in a cached `site-packages` directory across builds. Either disable dependency caching for training jobs or add an explicit cache purge step before install.

**Mirror lag.** An internal mirror that syncs slowly can serve an old, vulnerable version long after the public index has removed it. Set an explicit freshness requirement and fail the build if the mirror is behind by more than your stated window.

## A decision checklist before adopting this

Use this to decide whether the controls above fit your situation.

- Does your pipeline install third-party packages at build time? If yes, hash verification applies.
- Does your pipeline load datasets or model weights from a shared hub? If yes, commit pinning applies.
- Are your training artifacts long-lived or customer-facing? If yes, the cost of verification is easier to justify.
- Do you train on fully synthetic or fully private data with no external dependencies? If yes, the dataset controls add less value, and insider risk becomes the dominant concern.
- Do you rely on a managed ML platform that already mirrors and patches dependencies? If yes, adding a second hashing layer may conflict with the platform's tooling. Confirm compatibility before rolling it out.
- Is your model a thin wrapper around a hosted inference API? If yes, the dominant risk is prompt injection at request time, not a poisoned artifact, and the controls here are a poor fit.

## A 30-minute action

Pick one ML repository. Generate a hash-pinned lockfile and install from it with enforcement enabled:

```bash
pip install pip-tools
pip-compile requirements.in --generate-hashes --output-file requirements.txt
pip install --require-hashes -r requirements.txt
```

If the install succeeds, commit the lockfile and add the `--require-hashes` flag to the install step in CI. If it fails, read the error before changing anything: a hash mismatch on a package you did not expect to change is exactly the signal this whole approach exists to surface.
