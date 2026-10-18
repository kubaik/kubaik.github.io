# Golden paths that don't handcuff

## What a golden path is, and how it fails

A golden path is the paved road a team agrees on: one scaffold, one deploy pipeline, one documented way to add a database migration. For a small team, it is the difference between shipping a feature on Tuesday and spending Tuesday re-deciding whether to use Postgres or SQLite for the third time.

The failure mode is not that golden paths are bad. It is that they calcify. The scaffold still generates a bundler config the rest of the ecosystem moved past. The internal CLI hardcodes a single cloud region and nobody remembers why. A new requirement appears — data residency, a second product line, a different runtime — and the paved road turns out to be a wall.

Most golden-path guidance treats the path as a product to be polished. The more useful framing is to treat it as an interface to be kept *reversible*. A path that cannot be escaped is not a productivity tool; it is coupling with a friendly name.

The escape hatch has to be designed at the same time as the path itself, not bolted on after the first team complains. The sections below cover how to build a scaffold and an internal toolchain that stay evolvable: exact version pinning, a compatibility contract, drift observability, and an eject path verified in CI.

## Prerequisites and what you will build

You will need:

- A current Node.js LTS release, or Python 3.11+ if you prefer — the pattern is the same
- Docker for local service parity
- A Git repository you control (GitHub, GitLab, or self-hosted Gitea)
- Access to a cloud region where you can create resources
- Basic familiarity with CI (GitHub Actions, GitLab CI, or similar)

The build produces a minimal golden-path scaffold for a web service. It will:

1. Generate a project from a template with pinned versions
2. Provide a deploy command that works in more than one region
3. Include a compatibility shim so older projects still build after the template changes
4. Emit data that tells you when a project has drifted off the path
5. Ship a documented escape hatch: an `--eject` flag that produces a standalone repo with no internal dependencies

The goal is not a perfect internal developer platform. It is a path that survives a couple of years of tool churn without forcing a rewrite. Internal scaffolds commonly need their first breaking change within 6–12 months; the techniques here push that out by treating the path as a versioned interface rather than a folder of files.

## Step 1 — decide the versioning contract first

Before writing template code, decide what gets versioned. This is the hard-to-reverse decision. If the template is named `web-service` and later needs a `web-service-v2`, every existing project inherits a migration problem. Version the *interface*, not the artifact.

A workable repo structure:

```
golden-path/
  templates/
    web-service/
      template.yaml        # metadata, version, supported regions
      files/
        src/
          index.ts
        package.json.tmpl
        Dockerfile
        .github/workflows/deploy.yml
  cli/
    src/
      create.ts
      eject.ts
      drift.ts
  compat/
    v1-to-v2.ts
```

`template.yaml` is the source of truth:

```yaml
name: web-service
interface_version: 1.3.0
supported_regions:
  - us-east-1
  - eu-west-1
  - ap-southeast-1
runtime:
  node: "20.11.1"
  package_manager: "pnpm@8.15.4"
compat:
  min_interface: 1.0.0
  max_interface: 2.0.0
deprecation_notice: null
```

Pin exact versions. `node: "20"` is not a pin; it is a range. A specific patch version is a pin. The same applies to the package manager, Docker base images (`node:20.11.1-bookworm-slim`), and CI actions — pin the action to a tag that does not move, or better, to a commit SHA. The reason is reproducibility: when a new hire runs `create` six months from now, they should get the same tree, not whatever the range resolves to that day.

Install the CLI dependencies:

```bash
cd cli
pnpm add yaml@2.3.4 commander@11.1.0 execa@8.0.1
pnpm add -D typescript@5.4.5 vitest@1.6.0 @types/node@20.12.7
```

Keep the `-D` distinction honest. Build-time tooling that leaks into runtime dependencies is a common source of image bloat and CVE noise.

## Step 2 — implement create, marker, and deploy

The core of the path is the `create` command. It reads `template.yaml`, copies files, substitutes variables, and writes a `.golden-path.json` marker into the generated project. That marker is what makes the path evolvable: it records which interface version the project was created against, so later tooling can decide whether to offer a migration.

```typescript
// cli/src/create.ts
import { readFile, writeFile, mkdir, cp } from 'node:fs/promises';
import { join } from 'node:path';
import { parse } from 'yaml';
import { execa } from 'execa';

interface TemplateMeta {
  name: string;
  interface_version: string;
  runtime: { node: string; package_manager: string };
}

export async function create(opts: {
  template: string;
  target: string;
  region: string;
}) {
  const tplDir = join(__dirname, '..', 'templates', opts.template);
  const meta = parse(
    await readFile(join(tplDir, 'template.yaml'), 'utf8')
  ) as TemplateMeta;

  await mkdir(opts.target, { recursive: true });
  await cp(join(tplDir, 'files'), opts.target, { recursive: true });

  const pkg = JSON.parse(
    await readFile(join(opts.target, 'package.json.tmpl'), 'utf8')
  );
  pkg.engines = { node: meta.runtime.node };
  await writeFile(
    join(opts.target, 'package.json'),
    JSON.stringify(pkg, null, 2)
  );

  await writeFile(
    join(opts.target, '.golden-path.json'),
    JSON.stringify(
      {
        template: meta.name,
        interface_version: meta.interface_version,
        created_at: new Date().toISOString(),
        region: opts.region
      },
      null,
      2
    )
  );

  await execa(meta.runtime.package_manager.split('@')[0], ['install'], {
    cwd: opts.target,
    stdio: 'inherit'
  });
}
```

Two details matter. First, `.golden-path.json` is deliberately plain JSON, readable by any tool, not a proprietary format. Second, the region is recorded at creation time but not baked into the deploy script — the script reads it from the marker at runtime. That is the difference between a path that can support multiple regions and one that cannot.

The deploy script lives in the template as `scripts/deploy.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REGION=$(jq -r .region .golden-path.json)
IFACE=$(jq -r .interface_version .golden-path.json)

# Compare semver fields, not raw strings.
IFS=. read -r MAJ MIN PATCH <<< "$IFACE"
if (( MAJ == 1 && MIN < 2 )); then
  echo "This project is on interface $IFACE, which predates multi-region."
  echo "Run 'golden-path migrate' to upgrade."
  exit 1
fi

case "$REGION" in
  us-east-1)      STACK=myapp-use1 ;;
  eu-west-1)      STACK=myapp-euw1 ;;
  ap-southeast-1) STACK=myapp-apse1 ;;
  *) echo "Unsupported region: $REGION"; exit 1 ;;
esac

TAG=$(git rev-parse --short HEAD)
docker build -t "$STACK:$TAG" .
docker push "$STACK:$TAG"
```

The version check is the compatibility shim. It is a handful of lines, and it is the difference between a path that can evolve and one that forces every project to upgrade in lockstep. Note the semver comparison: string comparison of version numbers breaks the moment a two-digit minor appears, which is a classic latent bug in shims like this.

## Step 3 — handle edge cases and errors

The most common trap is assuming a single environment. Real projects need at least local, staging, and production. If the template hardcodes `process.env.DATABASE_URL` and expects it to be set, the first time someone runs the local stack on a laptop they get a connection refused error against `127.0.0.1:5432` and no indication of why. Ship a `.env.example` with sane defaults and a compose file that actually starts Postgres on port 5432. That is a two-minute fix that saves every future contributor twenty minutes.

Another documented failure mode: the package install in `create` runs before the registry config file is copied, so private registry auth fails and the install falls back to the public registry. The error surfaces as a "no matching version" for an internal package, which looks like a version typo but is not. Copy the registry config before running install, or pass the registry explicitly.

For the `eject` command, the edge case is subtler. Ejecting should produce a repo with zero references to the golden path — no shared CI runners, no internal registry, no marker file. The test for this is a grep:

```bash
if grep -r "golden-path" --exclude-dir=node_modules --exclude-dir=.git .; then
  echo "Eject failed: golden-path references remain"
  exit 1
fi
```

Run that in CI against the ejected output. If it passes, the escape hatch is real. If it fails, the handcuffs are real too.

## Step 4 — add drift detection and tests

You cannot manage what you cannot see. Every generated project should record two facts on every deploy: the interface version it was built against, and the timestamp of the last successful deploy. Push them to whatever telemetry you already run, or to a small object store. The point is to have a queryable answer to "which projects are on which interface version."

A minimal drift report:

```typescript
// cli/src/drift.ts
import { glob } from 'glob';
import { readFile } from 'node:fs/promises';
import { parse } from 'yaml';

export async function drift(root: string) {
  const current = parse(
    await readFile('templates/web-service/template.yaml', 'utf8')
  );
  const projects = await glob('**/.golden-path.json', { cwd: root });

  const stale = [];
  for (const p of projects) {
    const marker = JSON.parse(
      await readFile(`${root}/${p}`, 'utf8')
    );
    const [major] = marker.interface_version.split('.');
    const [curMajor] = current.interface_version.split('.');
    if (major !== curMajor) {
      stale.push({
        project: p,
        on: marker.interface_version,
        latest: current.interface_version
      });
    }
  }
  return stale;
}
```

Run this on a schedule. If a large share of projects sit a major version behind, the path is drifting and you have a decision: invest in migration tooling, or declare the old version the path and stop pretending. Both are valid outcomes. Ignoring the number is not.

Tests for the CLI itself should cover three cases: create a project and assert `.golden-path.json` exists with the expected version; eject and assert the grep finds nothing; and run drift against a fixture tree with one stale and one current project. All three run fast enough to gate every commit.

| Concern | Naive approach | Evolvable approach |
|---|---|---|
| Version pinning | `node: "20"` | `node: "20.11.1"` |
| Region support | Hardcoded `us-east-1` | Read from `.golden-path.json` |
| Template upgrades | Rewrite every project | Compatibility shim + migrate command |
| Escape hatch | "Delete the CI config" | `eject` + grep assertion in CI |
| Drift detection | None | Scheduled drift report with a threshold |
| Deprecation | Surprise breakage | `deprecation_notice` in `template.yaml` |

## How to measure whether this is working

Do not trust anecdotes about golden paths. Instrument four things and compare before and after.

**Time to first deploy for a new service.** Timestamp the `create` invocation and the first successful production deploy. Median across new services is the number that matters; the mean hides the outliers that hurt most.

**Interface-version spread.** From the drift report, compute the share of active projects on the current major interface version. This is a single query against the marker files or your telemetry store. Track it weekly.

**Eject duration and correctness.** Time the `eject` command end to end, and record whether the grep assertion passed. A failing assertion is a bug report against the template, not a user error.

**Build reproducibility.** Periodically check out a project at a six-month-old commit and rebuild it from a clean cache. If the build fails because a dependency range moved, the pinning contract has a hole.

For each of these, the useful comparison is the same project before and after the change, not a cross-team benchmark. Absolute numbers vary too much by stack and team size to be meaningful.

## Common questions and variations

**Should the golden path be a monorepo or separate repos?**
Separate repos for the template and the CLI; a monorepo for generated projects only once there are enough of them to justify it. The template changes on a different cadence than the projects, and mixing them makes versioning painful.

**What if there is only one project?**
Then a golden path is premature. Write the project, then extract the template from it once a second one appears. Premature templating is its own form of handcuffs.

**How should secrets be handled in the template?**
Never inline them. Ship `.env.example` with placeholder values and document where real secrets live — a managed secrets service, a CLI-based password manager, or an encrypted-files workflow. Templates that reference secrets directly become a security incident waiting to happen.

**When should the major interface version be bumped?**
When the change would break an existing project's build without a migration. Adding a supported region is minor. Changing the deploy script's required environment variables is major. Record the reasoning in the template's changelog so the decision is reviewable later.

**What if a team genuinely needs to leave the path?**
That is the eject command's job, and it should be cheap and boring. If leaving is expensive, teams will fork the template manually instead, which produces a worse outcome: an unmaintained copy that still looks official.

## Where to go from here

The highest-leverage thing you can do in the next 30 minutes is open the root of your current project and check whether it contains a marker file recording which version of your scaffold it was created from. If it does not, create one now — a plain `.scaffold-version` file containing a semver string is enough to start. Then add a `grep -r "scaffold-name"` check to CI on your ejected output. That single check is what keeps the path from becoming handcuffs.
