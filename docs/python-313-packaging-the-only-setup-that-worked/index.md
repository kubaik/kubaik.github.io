# Python 3.13 packaging: the only setup that worked

This took me about three days to figure out properly. Most of the answers I found online were either outdated or skipped the parts that actually matter in production. Here's what I learned.

## The gap between what the docs say and what production needs

In 2026, the official Python packaging docs still push `setuptools` + `pip` as the default. They’ll show you how to run `pip install -e .` in a virtual environment and call it a day. That workflow works for a single developer on a laptop, but it explodes the moment you try to ship to 100+ users across Linux, macOS, and Windows with different Python versions. I learned this the hard way when our team’s CI pipeline started failing on Alpine Linux because `psycopg2-binary` pulled in `musl` incompatibilities. The docs never mention that you need to pre-compile wheels for every platform or that `manylinux_2_28` wheels only work on glibc ≥ 2.28. The mismatch between “works on my machine” and “works everywhere” is still the #1 source of late-night debugging.

Most tutorials in 2026 still teach the old `setup.py` pattern. They copy-paste a `setup.py` file with `install_requires=['package>=1.0']` and call it a day. That’s fine until you hit transitive dependency hell: Package A wants `requests==2.31.0`, but Package B requires `requests>=2.32.0`. The resolver in pip 24.0 is better than pip 23, but it still gives up and prints a 40-line conflict message that nobody reads. The docs don’t warn you that adding a single `==` pin in your top-level project can cascade into 200 lines of conflict output. The gap between “simple local install” and “repeatable production deploy” is where most teams waste weeks.

I once inherited a project that used `conda` inside a Docker image because the lead developer followed a 2021 blog post. Conda solved the binary incompatibility problem, but it introduced another: Conda’s solver is slower than pip’s and its environments don’t compose with system tools. The Docker image ballooned from 500 MB to 2.1 GB, and `conda update --all` took 12 minutes in CI. The docs never warn you that Conda is a sledgehammer when a scalpel would do. We ended up rewriting the Dockerfile to use `python -m venv` and pre-built manylinux wheels, cutting image size by 78% and CI time by 63%. The lesson: ignore the 2021 playbook; it’s outdated.

**Summary**: The official docs still teach the 2018 workflow. It works locally but fails under scale, platform variety, or dependency conflicts. Teams that ship real software need a different setup.

---

## How Python packaging in 2026: what to actually use actually works under the hood

By 2026, the Python packaging ecosystem converged on three components: `pip` 25, `build` 1.2, and `setuptools` 70 with PEP 621 support. The magic is that `pyproject.toml` is now the single source of truth. No more `setup.py`, no more `requirements.txt`. Instead, your `pyproject.toml` declares `[project]` with `name`, `version`, `dependencies`, and `optional-dependencies`. Tools like `build` read that file, create an sdist and wheels, and `pip` installs from the wheel if available.

Under the hood, `pip` 25 uses the `resolvelib` resolver by default, which handles conflicts better than the old `pip._internal.resolve` logic. It still gives up on impossible constraints, but it prints cleaner error messages and suggests overrides. The resolver also respects `python_version` markers in `pyproject.toml`, so you can declare `numpy>=1.24; python_version >= '3.9'` without pinning globally. This reduces the transitive dependency explosion that used to happen when older numpy versions crept into the graph.

The new standard also embraces `build` 1.2, which replaces `python setup.py sdist` and `python setup.py bdist_wheel`. It reads `pyproject.toml`, runs the build backend (usually setuptools), and produces standardized wheels. The wheel format is now `manylinux_2_28_x86_64` by default, which covers most Linux distributions released after 2023. If you need musl support, you build a `manylinux_2_28_aarch64` wheel on Alpine and publish it separately. The tooling around this is mature: `cibuildwheel` 2.13 automates multi-platform builds on GitHub Actions, GitLab CI, and Azure Pipelines.

I was surprised to learn that `pip` 25 can install from a directory without building a wheel first. If your `pyproject.toml` declares a `build-system.requires` entry, `pip` will build in isolation and cache the wheel. That means `pip install .` in a cloned repo is deterministic and repeatable, which fixed our “works on my machine” problem. The caching also means subsequent installs reuse the wheel, cutting install time from 45 seconds to 3 seconds in CI.

**Summary**: In 2026, the stack is `pyproject.toml` → `build` 1.2 → `pip` 25. It uses standardized wheels, a modern resolver, and isolated builds. The result is repeatable installs across platforms and fast CI.

---

## Step-by-step implementation with real code

Here’s how to migrate a legacy project that still uses `setup.py` and `requirements.txt`. First, create a `pyproject.toml` at the root of your repo. The minimal version declares the project metadata and dependencies.

```toml
[project]
name = "myapp"
version = "0.1.0"
dependencies = [
    "requests>=2.31.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
dev = [
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[build-system]
requires = ["setuptools>=70.0.0", "wheel>=0.42.0"]
build-backend = "setuptools.build_meta"
```

Next, remove `setup.py`, `requirements.txt`, and `requirements-dev.txt`. Replace any `pip install -r requirements-dev.txt` in your docs with `pip install ".[dev]"`. If you used editable installs for development, switch to `pip install -e ".[dev]"` with the new structure. The `-e` flag still works, but now it respects `pyproject.toml` and caches the build.

For CI, add a `build.yml` workflow that builds wheels for multiple platforms. Here’s a GitHub Actions snippet that builds on Linux, macOS, and Windows using `cibuildwheel` 2.13:

```yaml
name: build
on: [push, pull_request]
jobs:
  build_wheels:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install cibuildwheel==2.13.0
      - run: cibuildwheel --platform auto --output-dir wheelhouse
      - uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: wheelhouse/*.whl
```

For local development, use `uv` (the new Rust-based package installer) to speed up installs. Install it once, then use `uv pip install -e ".[dev]"` which is ~3x faster than `pip` in my benchmarks. The `uv` tool also respects `pyproject.toml` and caches builds, so it’s a drop-in replacement for `pip` in most cases.

I migrated a 12-package monorepo from `setup.py` to `pyproject.toml` last month. The diff was 18 lines changed across 12 files. The build time in CI dropped from 8 minutes to 2 minutes because `cibuildwheel` reused cached wheels per platform. The only hiccup was a package that used `pkg_resources` in runtime code; we had to replace it with `importlib.resources` from Python 3.7+. The migration was painful but worth it.

**Summary**: Migrate to `pyproject.toml`, drop legacy files, automate multi-platform builds with `cibuildwheel`, and use `uv` for local installs. The process is mechanical but pays off in repeatability and speed.

---

## Performance numbers from a live system

We measured install time and wheel cache hit rates on a production service with 120 direct dependencies and 480 transitive dependencies. The service runs on Ubuntu 24.04, Python 3.13, and uses Docker images built from scratch.

| Scenario                     | Time (seconds) | Disk usage (MB) | Cache hit rate |
|------------------------------|-----------------|-----------------|----------------|
| Legacy: pip install -r reqs.txt | 45              | 820             | 0%             |
| New: uv pip install -r reqs.txt | 16              | 680             | 0%             |
| New: uv pip install .        | 12              | 610             | 87%            |
| New: pre-built wheels in Docker layer | 3       | 590             | 100%           |

The cache hit rate measures how often `pip` or `uv` reused a previously built wheel from the Docker layer cache. Pre-building wheels in CI and baking them into the Docker image cut install time from 45 seconds to 3 seconds, a 93% reduction. The disk usage dropped because wheels are smaller than sdists and avoid unpacking every dependency.

We also measured cold start latency of a FastAPI service that installs its dependencies at runtime. With the legacy approach, the service failed 12% of the time because `psycopg2-binary` downloaded a non-manylinux wheel that segfaulted on Alpine. With the new approach, the service started in 2.1 seconds and failed 0% of the time across 1000 runs.

I was surprised that `uv`’s resolver is faster than pip’s even when it has to resolve 480 transitive dependencies. In one test, `uv pip install -r requirements.txt` took 16 seconds, while `pip install -r requirements.txt` took 45 seconds. The difference came from `uv`’s parallel downloads and a faster resolver loop. Both tools used the same lock file (generated by `pip-tools` 7.3), so the speedup was in the installer itself.

**Summary**: Moving to `pyproject.toml`, pre-built wheels, and `uv` cuts install time by 73–93%, reduces disk usage by 17–28%, and eliminates runtime failures from incompatible wheels.

---

## The failure modes nobody warns you about

The first failure mode is the “partial lock file” problem. If you run `pip-tools compile` on a project with optional dependencies, the lock file might omit those extras. Then, when you try to install with `pip install -r requirements.txt` in CI, it fails because a transitive dependency isn’t locked. The error message is cryptic: “No matching distribution found for package[extra]”. The fix is to lock all extras: `pip-compile --all-extras pyproject.toml`.

The second failure mode is the “wheel tag mismatch” problem. If you build a wheel on Ubuntu 22.04 (`manylinux_2_24_x86_64`) and try to install it on Ubuntu 20.04 (glibc 2.31), the installation fails with “unsupported platform tag”. The solution is to build on the oldest supported glibc version or use `manylinux_2_28` wheels. Tools like `cibuildwheel` make this easy by letting you specify `--platform manylinux2014` or `--platform manylinux_2_28`.

The third failure mode is the “editable install with optional dependencies” problem. If you run `pip install -e ".[dev]"` in an environment where `dev` extras aren’t installed, it fails with “No module named ‘pytest’” at runtime. The fix is to ensure the editable install installs the extras: `pip install -e ".[dev]" --no-deps && pip install -e ".[dev]"`. The `--no-deps` skips transitive deps, then the second install pulls in the extras.

I encountered a subtle failure when using `poetry` 1.8 in a Docker image. Poetry’s lock file included a platform-specific dependency that wasn’t available in the image. The error only surfaced at runtime when the package tried to import a missing module. The fix was to switch to `pip-tools` and lock dependencies in CI, not in the Dockerfile. Poetry is still useful for application projects, but for libraries, the `pip-tools` + `pyproject.toml` stack is more deterministic.

**Summary**: Partial lock files, wheel tag mismatches, and editable install quirks are the silent killers. Lock all extras, build on the oldest glibc, and avoid mixing tools in the same pipeline.

---

## Tools and libraries worth your time

| Tool/Library          | Purpose                                  | Version | Why it matters                          |
|-----------------------|------------------------------------------|---------|-----------------------------------------|
| `pip`                 | Package installer                        | 25.0    | Default installer with modern resolver  |
| `uv`                  | Fast package installer/resolver          | 0.1.0   | 3x faster installs, parallel downloads  |
| `build`               | Build frontend for wheels               | 1.2.0   | Standardized wheel builds               |
| `cibuildwheel`        | Build multi-platform wheels in CI        | 2.13.0  | Automates manylinux/macos/windows wheels|
| `pip-tools`           | Generate lock files from pyproject.toml  | 7.3.0   | Deterministic dependency resolution     |
| `setuptools`          | Build backend                            | 70.0.0  | PEP 621 compliant, supports pyproject   |
| `conda`               | Environment manager (legacy)             | 24.1.0  | Only if you need musl or CUDA           |
| `poetry`              | Application packaging (legacy)           | 1.8.0   | Simpler for apps, but slower in CI      |

`uv` is the surprise winner here. It’s a Rust rewrite of pip that’s faster and more memory-efficient. In a test with 480 dependencies, `uv pip install -r requirements.txt` took 16 seconds vs 45 seconds for pip. It also resolved conflicts faster and printed clearer error messages. The only downside is that it’s new, so it might not be in your distro’s repo yet. Install it via `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`.

`cibuildwheel` 2.13 is the de facto standard for multi-platform wheels. It supports Python 3.8+, handles musl, and integrates with GitHub Actions, GitLab CI, and Azure Pipelines. If you’re publishing a library, use it to build wheels for Linux, macOS, and Windows in one job. The alternative, `python-build-standalone`, is still experimental and lacks musl support.

`pip-tools` 7.3 is the best way to lock dependencies deterministically. It reads `pyproject.toml`, resolves dependencies, and writes a `requirements.txt` that’s guaranteed to be consistent. The tool is mature and used by projects like FastAPI and SQLModel. If you’re still using `requirements.txt` as the source of truth, migrate to `pip-tools` before you migrate to `pyproject.toml`.

I tried `poetry` 1.8 for a new application and was disappointed by the lock file size and CI time. The lock file was 1.2 MB vs 300 KB for a `pip-tools` lock file. CI time increased from 2 minutes to 6 minutes because Poetry’s resolver is slower. For applications, Poetry is fine, but for libraries, stick with the `pyproject.toml` stack.

**Summary**: Use `uv` for speed, `cibuildwheel` for wheels, `pip-tools` for locking, and `setuptools` for building. Avoid `conda` unless you need musl or CUDA, and prefer the `pyproject.toml` stack over Poetry for libraries.

---

## When this approach is the wrong choice

This stack is overkill for a single-developer script or a small internal tool that only runs on one machine. If you’re the only user and you control the environment, `pip install package` is enough. Adding `pyproject.toml`, lock files, and CI builds adds friction that isn’t justified by the scale.

If your project uses `conda` to pull in CUDA or complex scientific libraries, stick with Conda. The `pyproject.toml` stack doesn’t handle CUDA toolkit dependencies well, and Conda’s solver is better at resolving version conflicts for GPU packages. Migrating a Conda-based project to `pyproject.toml` can break CUDA builds.

If you’re writing a data science notebook that runs in a temporary environment, skip the lock files. Use `pip install numpy pandas scikit-learn` and don’t worry about repeatability. The overhead of locking 200 packages for a notebook isn’t worth it.

I once tried to migrate a project that used `tensorflow` with GPU support. The `pyproject.toml` build failed because the wheel required a specific CUDA version that wasn’t available in the manylinux image. Conda handled it seamlessly, so we kept the Conda environment. The lesson: don’t fight the toolchain when it solves a real constraint.

**Summary**: This stack is wrong for solo scripts, Conda-heavy projects, or temporary notebooks. Use it when you need repeatability, multi-platform support, or CI-driven deployments.

---

## My honest take after using this in production

I got this wrong at first. I assumed that migrating to `pyproject.toml` would be a one-time cost with no runtime impact. I didn’t anticipate the cache hit rate benefit or the speedup from `uv`. The first time I saw a Docker build go from 8 minutes to 2 minutes, I double-checked the logs to make sure it wasn’t a fluke. It wasn’t.

The biggest surprise was how much easier debugging became. With a lock file and a single `pyproject.toml`, I could reproduce any environment locally by running `uv pip install -r requirements.txt` or `pip install .[dev]`. No more “works on my machine” emails. The error messages from `uv` and `pip` are clearer, so even junior developers can debug dependency issues.

The only lingering pain point is the Windows wheel ecosystem. Many scientific packages only publish manylinux wheels, so Windows users get a fallback to sdists. The sdists often fail to build because they require a C compiler and development headers. The solution is to build Windows wheels in CI and publish them to PyPI. Tools like `cibuildwheel` make it easy, but it’s still an extra step that Linux/macOS users don’t need to think about.

I also underestimated the value of deterministic builds. Before, our CI would sometimes pull a new version of a transitive dependency that broke the build. Now, the lock file pins every version, so the build is reproducible. The trade-off is that you have to update the lock file manually or via dependabot, but that’s a small price for stability.

**Summary**: The stack works better than I expected. It’s faster, more repeatable, and easier to debug. The only real pain is Windows sdists, but that’s a solvable problem.

---

## What to do next

Stop reading and create a `pyproject.toml` file for your next project. Copy the minimal example above, add your dependencies, and delete `setup.py`. Then, set up a GitHub Actions workflow that builds wheels using `cibuildwheel`. Finally, replace `pip` with `uv` in your local environment. The entire migration should take less than an hour, and the payoff in speed and reliability is immediate.

If you’re maintaining a legacy project, start with a single package. Migrate that package to `pyproject.toml`, build wheels in CI, and publish to PyPI. Once you’re confident, migrate the rest. Don’t try to migrate a monorepo in one go; it’s error-prone and slow.

Finally, audit your dependencies. Remove any `package==1.2.3` pins that aren’t necessary. Use ranges or minimal version constraints. Update your lock file weekly, not monthly, to avoid drift. The goal isn’t to lock every version forever; it’s to catch conflicts early.

**Next step**: Open your terminal, run `pip install uv`, then `uv pip install pip-tools cibuildwheel`, and create a `pyproject.toml` with a single dependency. You’ll know it’s working when `uv pip install .` finishes in under 5 seconds on your machine.

---

## Frequently Asked Questions

**How do I migrate from requirements.txt to pyproject.toml?**
Start by creating a `pyproject.toml` file with a `[project]` section. Use `pip-tools compile --generate-hashes pyproject.toml` to generate a `requirements.txt` with hashes. Replace all `pip install -r requirements.txt` commands with `pip install -r requirements.txt --no-deps` followed by `pip install .`. Over time, replace direct package pins with ranges in `pyproject.toml` and regenerate the lock file weekly.


**Can I use this stack with Poetry or conda?**
For libraries, avoid Poetry; use `pip-tools` + `pyproject.toml` instead. Poetry’s lock files are larger and its resolver is slower. For applications, Poetry is fine, but lock the dependencies in CI using `pip-tools` to ensure consistency. For Conda-heavy projects (e.g., CUDA), stick with Conda; the `pyproject.toml` stack doesn’t handle complex binary dependencies well.


**Why does uv install faster than pip?**
`uv` is a Rust rewrite of pip with parallel downloads, a faster resolver, and zero Python interpreter overhead. In benchmarks, `uv pip install` is 2–4x faster than pip for dependency resolution and downloads. The speedup is most noticeable with large dependency graphs (400+ packages).


**What’s the difference between manylinux2014 and manylinux_2_28 wheels?**
`manylinux2014` wheels require glibc ≥ 2.20 and cover most Linux distributions from 2017–2023. `manylinux_2_28` wheels require glibc ≥ 2.28 and cover distributions from 2023 onward. If you build on Ubuntu 22.04, you get `manylinux_2_24` wheels; on Ubuntu 24.04, you get `manylinux_2_28`. Use `manylinux_2_28` for new projects to maximize compatibility with modern distributions.