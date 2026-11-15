# Python deps: uv vs Poetry vs pip-tools

## Three tools, three mental models

Ask several Python developers how they manage dependencies and the answers will diverge. One keeps a `requirements.txt` compiled with `pip-compile`. Another has a `pyproject.toml` with Poetry metadata and a `poetry.lock`. A third has moved to `uv` and runs `uv sync` instead of `pip install`. Each approach works in isolation. They do not interoperate cleanly.

The confusion is not about which tool is "best" in the abstract. Each tool encodes a different model of what a dependency *is*, and teams mix them without noticing that the models conflict. The practical problem: a repository cannot safely mix `pip`, Poetry and `uv` without knowing which tool owns the lockfile and which one owns the virtualenv.

This article lays out the underlying model each tool uses, shows where the models diverge, works the same small project through all three, and ends with a decision checklist.

## The underlying model: resolver, lockfile, environment

Every Python dependency tool does three things, and the differences live in how it does them:

1. **Resolve** a set of version constraints into concrete pins.
2. **Record** those pins in a lockfile (or not).
3. **Install** the pins into an environment, usually a virtualenv.

`pip` historically only did step 3 well, with a resolver added in recent versions. `pip-tools` adds steps 1 and 2 through `pip-compile`, producing a `requirements.txt` with hashes. Poetry and `uv` do all three, but they disagree on file formats and on whether the lockfile is authoritative.

The key mental model: **the lockfile is the contract**. If two tools disagree about which file is the lockfile, you have two contracts, and CI will eventually install a different set of packages than a developer laptop. That is the failure mode behind most "works on my machine" incidents in Python projects.

A second axis is **where the virtualenv lives**. `pip` and `pip-tools` install into whatever environment is active (`$VIRTUAL_ENV`). Poetry creates and manages a virtualenv in a cache directory by default unless `virtualenvs.in-project` is set. `uv` defaults to a `.venv` in the project root. Running `poetry install` and then `uv sync` in the same repo can leave two virtualenvs and a confusing `python -c "import sys; print(sys.prefix)"` result.

## Where the models diverge

### pip-tools: constraints in, pins out

`pip-tools` is the smallest conceptual leap from plain `pip`. Abstract requirements live in `requirements.in`:

```text
# requirements.in
requests>=2.31
flask>=3.0,<4
```

Compile them:

```bash
pip-compile requirements.in --generate-hashes -o requirements.txt
```

The output `requirements.txt` contains exact pins with hashes, including transitive dependencies. Install with `pip install -r requirements.txt`. There is no separate lockfile concept — the compiled `requirements.txt` *is* the lockfile.

This model is transparent. The tradeoffs: `pip-tools` does not manage the virtualenv, does not know about `pyproject.toml` dependency groups, and does not handle editable installs of your own package gracefully. It also does not resolve platform-specific markers into a single lockfile the way Poetry and `uv` do; conditional dependency trees may need separate compiled files per Python version or platform.

### Poetry: project metadata plus lockfile

Poetry treats `pyproject.toml` as the source of truth for both package metadata and dependency constraints. `poetry.lock` records the resolved graph. `poetry install` reads the lockfile and installs into a Poetry-managed virtualenv.

The lockfile format is Poetry-specific. It is not a `requirements.txt`, and `pip install -r` cannot consume it directly. Poetry can export one (`poetry export -f requirements.txt`), but the export is a one-way projection: the lockfile stays authoritative, and the exported file drifts if the export step is forgotten.

Poetry's resolver is stricter than pip's about conflicting constraints, which catches problems early and frustrates when a transitive dependency has a loose upper bound. Long resolution on large graphs is a real tradeoff of full backtracking resolution rather than pip's more permissive approach.

### uv: resolver, environment manager and installer

`uv` is a single binary that resolves, locks and installs. It reads `pyproject.toml` for project metadata and dependencies, writes `uv.lock`, and creates a `.venv` in the project root by default. `uv sync` brings the environment in line with the lockfile; `uv lock` re-resolves; `uv add` and `uv remove` edit `pyproject.toml` and update the lock together.

The important difference from Poetry is that `uv` is also a drop-in replacement for `pip` and `pip-tools` in many workflows: `uv pip install`, `uv pip compile` and `uv pip sync` operate on `requirements.txt` files. That allows incremental adoption — replace `pip install` with `uv pip install` first, then move to the project/lockfile model when ready.

The `uv.lock` format is not a `requirements.txt` either. As with Poetry, export is needed to interoperate with tools that expect one.

### Side-by-side

| Concern | pip-tools | Poetry | uv |
|---|---|---|---|
| Source of constraints | `requirements.in` | `pyproject.toml` | `pyproject.toml` |
| Lockfile | compiled `requirements.txt` | `poetry.lock` | `uv.lock` |
| Manages virtualenv | No | Yes (cache dir by default) | Yes (`.venv` by default) |
| Consumable by `pip install -r` | Yes (it is one) | Via `poetry export` | Via `uv export` |
| Handles editable self-install | Awkward | Yes | Yes |
| Drop-in `pip` replacement | No | No | Yes (`uv pip`) |

## Worked example: the same project in all three

Assume a small Flask service with two direct runtime dependencies and one dev-only dependency. The `pyproject.toml` for the Poetry and `uv` cases is similar:

```toml
[project]
name = "example-service"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "flask>=3.0,<4",
    "requests>=2.31",
]

[dependency-groups]
dev = [
    "pytest>=8",
]
```

**pip-tools workflow:**

```text
# requirements.in
flask>=3.0,<4
requests>=2.31
```

```text
# requirements-dev.in
-c requirements.in
pytest>=8
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile requirements.in --generate-hashes -o requirements.txt
pip-compile requirements-dev.in --generate-hashes -o requirements-dev.txt
pip install -r requirements-dev.txt
```

The `-c requirements.in` line constrains the dev file to the same versions as the runtime file, which is the pip-tools idiom for keeping dev and runtime pins aligned. Without it, `pip-compile` may resolve `pytest` against a different `flask` version than `requirements.txt` contains.

**Poetry workflow:**

```bash
poetry add "flask>=3.0,<4" "requests>=2.31"
poetry add --group dev "pytest>=8"
poetry install
```

Poetry writes the constraints into `pyproject.toml`, resolves, and writes `poetry.lock`. On CI, `poetry install --sync` makes the environment match the lockfile exactly, removing packages not in it. Without `--sync`, Poetry leaves extra packages in place, a subtle source of drift.

**uv workflow:**

```bash
uv init --python 3.11
uv add "flask>=3.0,<4" "requests>=2.31"
uv add --dev "pytest>=8"
uv sync
```

`uv sync` creates `.venv` if it does not exist and installs exactly the locked set. `uv run pytest` executes a command inside the project environment without activation, which is the cleanest CI pattern of the three.

## Failure modes worth knowing

**CI resolves fresh instead of installing from the lockfile.** A pipeline runs `pip install -e .` or `pip install .`, which resolves from `pyproject.toml` and ignores the lockfile entirely. The build passes while installing a different set of packages than developers have. The fix is to make the lockfile-reading command the only install step and to fail the build when `pyproject.toml` and the lockfile disagree. `uv lock --check` and `poetry check --lock` serve that purpose; run them before install.

**Two lockfiles in one repository.** A repo containing both `poetry.lock` and `uv.lock` is a repo where nobody knows which one is real. The tools will also disagree about which virtualenv to use. Pick one tool per repository and enforce it with a pre-commit hook or a CI check that fails if the other tool's lockfile appears.

**Editors and IDEs pick the wrong interpreter.** With `pip-tools`, `pip-compile` works regardless of the active environment (it does not need one), but `pip install -r` installs wherever the active `python` points. If an editor runs `python` from a system interpreter, installs land in the wrong place. The `pip-tools` model pushes environment management onto the developer; that is a feature for some teams and a footgun for others.

**Hashes drift between runtime and dev files.** When dev constraints are compiled without `-c requirements.in`, the dev lock can pin a different version of a shared transitive dependency. Installing the dev file then upgrades or downgrades a runtime package. Compile dev files against the runtime constraints and diff the two outputs in CI.

**A slow resolver is blamed on the tool.** Resolution time depends on graph size and constraint tightness. When a project resolves slowly, the usual cause is a transitive dependency with a wide version range that forces backtracking. Loosening your own constraints rarely helps; tightening the problematic transitive constraint via a direct pin often does. Switching tools fixes the symptom for one project and reproduces it on the next large graph.

## How to measure before deciding

Speed claims are tool- and project-specific, so measure on the actual dependency graph rather than trusting general statements.

- **Cold resolve time:** in a clean checkout with no cache, time the lock command (`uv lock`, `poetry lock`, or `pip-compile`). Run it three times, clearing the tool's cache between runs, and compare medians.
- **Warm install time:** with the cache populated, time the install-from-lock command (`uv sync --frozen`, `poetry install --sync`, or `pip install -r requirements.txt --require-hashes`) into a fresh virtualenv. This is the number that dominates CI.
- **Lockfile diff size:** change one direct dependency and count the lines that change in the lockfile. Large diffs on small changes indicate a resolver that re-pins broadly.
- **CI wall-clock:** instrument the dependency-install step itself, not the whole job, so other steps do not mask the difference.

A simple harness is a shell script that records `date +%s%N` before and after each command, runs each step three times, and prints the median. Record the results next to the tool version and the commit hash, because resolver behavior changes between releases.

## A decision framework

Use this to pick one tool per project. The first matching row wins.

1. **You want a single binary that also replaces `pip` and works on Windows CI without a preinstalled Python toolchain.** Use `uv`. The `uv pip` compatibility layer allows incremental migration, and `uv run` removes the activation step from CI.
2. **You publish a library to PyPI and need PEP 621 metadata plus a lockfile for development.** Use Poetry or `uv` — both read `pyproject.toml` `[project]` metadata. If a team already has Poetry lockfiles and CI, staying on Poetry is reasonable; migration cost is real and the benefit is mostly speed.
3. **You have an existing `requirements.txt`-based pipeline (Docker images, Lambda layers, legacy CI) and cannot change the install step.** Use `pip-tools`. It produces a `requirements.txt` that every downstream tool already understands, with hashes for `pip install --require-hashes`.
4. **You have a monorepo with multiple packages and shared constraints.** Use `uv` workspaces or Poetry's path dependencies, not `pip-tools`. `pip-tools` has no concept of a workspace, so per-package files must be compiled and kept aligned by hand.
5. **You are on an air-gapped or heavily restricted network.** `pip-tools` with a local wheel mirror is the simplest model because the install step is plain `pip`. `uv` and Poetry both support custom indexes, but verify resolver behavior against the mirror before committing.

What to avoid: mixing tools in one repo, keeping both a `requirements.txt` and a lockfile as "authoritative" without a sync check, and letting CI resolve fresh instead of installing from the lockfile.

## Frequently asked questions

**Can uv work with an existing requirements.txt?**

Yes. `uv pip compile requirements.in -o requirements.txt` produces the same kind of hashed output as `pip-compile`, and `uv pip sync requirements.txt` installs exactly that set. This is the lowest-risk adoption path: keep the existing files, swap the commands, and measure the difference before moving to `uv.lock` and `pyproject.toml`.

**Why does poetry install leave packages that are not in the lockfile?**

By default, `poetry install` adds missing packages but does not remove extras. `poetry install --sync` removes anything not in the lockfile, making the environment match exactly. CI should use `--sync`; local development often does not, which is why environments drift over time.

**How do I stop pip from resolving fresh in CI?**

Replace `pip install -e .` or `pip install .` with an install from the lockfile: `uv sync --frozen`, `poetry install --sync`, or `pip install -r requirements.txt --require-hashes`. The `--frozen` flag on `uv sync` fails if `pyproject.toml` and `uv.lock` are out of sync, which catches the case where someone edited dependencies without re-locking.

**What is the difference between a lockfile and a constraints file?**

A lockfile records the full resolved graph, including transitive dependencies, at exact versions and usually with hashes. A constraints file (`-c` in pip) only narrows what the resolver is allowed to pick; it does not pin the full graph. `pip-tools` uses constraints as input and produces a lockfile-shaped `requirements.txt` as output. Conflating the two is a common source of "why did CI install a different version" confusion.

## The next 30 minutes

Open the repository you deploy most often and check whether it has more than one of `requirements.txt`, `poetry.lock` or `uv.lock` at the root. If it does, identify which file CI actually installs from, remove or archive the others in a branch, and add a CI step that fails if the wrong lockfile reappears. That single check prevents the most common class of Python dependency drift before it reaches production.
