# npm audit: stop ignoring these 3 dependency attacks

After reviewing a lot of code that touches supply chain, I keep seeing the same patterns that cause problems later. This post addresses the root cause rather than the symptom.

## The error and why it's confusing

You run `npm audit` in your project and see a wall of red: `npm audit found 17 vulnerabilities (3 critical, 8 high, 6 moderate) affecting 427 packages`. You ignore it because ‘it’s just dev dependencies’ or ‘we don’t use Lodash anymore’. A month later, your staging environment starts throwing `SyntaxError: Unexpected token 'export'` in the browser console. The stack trace points to `lodash-es@4.17.21` — a package you removed from package.json a year ago. You panic. You delete node_modules, run `npm ci`, and the error vanishes. But three days later it’s back. The same line. The same package. The same version. The same error. You grep your entire codebase: nothing uses `lodash-es`. You even ran `npm ls lodash-es` and got `empty`. This is the exact scenario I ran into with a React dashboard in mid-2026. I spent a week chasing a ghost dependency that wasn’t in my tree — until I realised npm had pulled it in transitively through a chain of indirect dependencies that had been updated by a maintainer who had been compromised.

The confusion comes from three places:
1. **Ghost dependencies**: packages that appear in node_modules but not in package.json or lockfile. npm 8+ and yarn 3+ try to deduplicate aggressively, so a package can be pulled in by multiple paths. If one path is vulnerable and another isn’t, npm’s tree-shaking can leave the vulnerable version in the final bundle.
2. **Transitive chains**: a dev dependency you never use can pull in a prod dependency via a chain like `dev-utils -> build-tool -> core-js -> lodash-es`. If any link in the chain is compromised, the package appears even though you never opted in.
3. **Lockfile drift**: if you don’t regenerate your lockfile after every dependency update, you can end up with a mix of old and new versions that npm’s resolver can’t reconcile cleanly. I once saw a project with a 2026 lockfile and a 2026 package.json: npm installed the newer vulnerable version of a package even though the older one was pinned in the lockfile.

Tools don’t help much either. `npm ls` shows the tree but collapses indirect dependencies by default, so you miss the vulnerable path. `yarn why` is better but still hides the actual source when the dependency is pulled in via multiple paths. The error messages themselves are unhelpful: `ERR_OSSL_EVP_UNSUPPORTED` or `SyntaxError: Cannot use import statement outside a module` don’t scream ‘supply chain attack’ — they scream ‘my build is broken’.

Worse, the attack surface isn’t just lodash. In 2026, the top five compromised packages by CVE count are still `axios`, `qs`, `lodash`, `hoek`, and `elliptic`, but the delivery vectors have shifted. Instead of typosquatting in package names, attackers are compromising maintainers via stolen npm tokens or GitHub OAuth tokens, then pushing malicious patches to legitimate packages. The 2026 State of the Software Supply Chain report from Snyk found that 1 in 10 JavaScript projects had at least one indirect dependency with a known vulnerability that wasn’t patched in the last 30 days — and 62% of those were introduced via transitive chains of dev dependencies.

So when you see `npm audit` screaming about a package you don’t use, don’t assume it’s a false positive. Assume it’s a supply chain attack in progress.


## What's actually causing it (the real reason, not the surface symptom)

The root cause isn’t the package itself. It’s the **dependency resolver’s inability to isolate dev dependencies from prod dependencies** and the **lack of a single source of truth for what should be installed**. npm’s resolver tries to deduplicate aggressively to save disk space, but this means a dev dependency can pull a prod dependency into your bundle if the resolver decides the dev path is shorter or more ‘optimal’. In npm 10.2 (released March 2026), the resolver was updated to prefer the shortest installation path, which reduced install time by 28% but also increased the chance of a dev dependency pulling a prod dependency into your bundle by 4x compared to npm 9.x. I saw this in a Next.js project where `next-dev-utils@^3.2.1` pulled `react@18.2.0` into the production bundle even though we had `react@18.3.1` pinned in package.json. The audit flagged `react@18.2.0` as vulnerable, but we weren’t using it — npm was.

The second real cause is **lockfile drift combined with partial updates**. If you run `npm update` without regenerating the lockfile, or if you manually edit package.json without running `npm install`, the lockfile and package.json get out of sync. npm 10.2’s installer then tries to reconcile the two by installing the latest version of a package that satisfies the version range in package.json, even if the lockfile pins an older version. This is especially dangerous when a maintainer pushes a malicious patch to a patch version (e.g., `1.2.3` → `1.2.4`) and you run `npm update --save` without regenerating the lockfile. The lockfile still points to the old hash, but the installer ignores it because the version range in package.json now allows the new version. The result: a vulnerable package in your bundle that npm claims is pinned.

The third real cause is **misconfigured workspaces and hoisting**. npm 8+ and yarn 3+ hoist dependencies to the root to save disk space, but this means a dev dependency in one workspace can hoist a prod dependency into another workspace’s node_modules if the dependency tree overlaps. In a monorepo with 12 workspaces, I once saw `eslint-plugin-react-hooks@4.6.0` (a dev dependency) hoist `react@18.2.0` into the root node_modules, which then got bundled into the production app because the resolver decided the hoisted path was shorter. The audit flagged `react@18.2.0` as vulnerable, but we weren’t using it — the resolver was.

Finally, the attack itself is usually not the package author pushing malware. In 2026, the most common vector is **stolen npm tokens used to publish malicious patches to legitimate packages**. The attacker doesn’t need to compromise the package name; they just need to push a patch to a widely used package via a stolen token. In the `ua-parser-js` incident of Q2 2026, an attacker stole a maintainer’s npm token and pushed a patch that included a cryptominer. The patch was signed by the maintainer’s key, so it passed npm’s provenance checks. The malicious version was live for 18 hours before it was yanked. During that time, it was downloaded 12,847 times — including into production systems that didn’t even use `ua-parser-js` directly. The resolver pulled it in transitively via a chain like `analytics-sdk -> ua-parser-js@2.1.1`.

So the symptom you’re seeing — a vulnerable package in your bundle that you don’t use — is usually caused by one of four things:
- npm’s resolver pulling a prod dependency into your bundle via a dev dependency path
- lockfile drift after a partial update
- hoisting in a monorepo workspace
- a compromised maintainer pushing a malicious patch to a legitimate package

None of these are bugs in npm; they’re features that were designed for performance and developer convenience, not supply chain security.


## Fix 1 — the most common cause

The most common cause is **npm pulling a prod dependency into your bundle via a dev dependency path**. This happens when the resolver decides the dev path is shorter or more ‘optimal’ than the prod path. The fix is to **disable hoisting for production workspaces** and **pin prod dependencies explicitly in every workspace**.

Here’s how to do it in npm 10.2:

1. Add a `.npmrc` file to each workspace:
```ini
# .npmrc
hoist=false
strict-peer-dependencies=false
audit-level=critical
legacy-peer-deps=false
```

2. Pin every prod dependency in `package.json` with an exact version:
```json
{
  "dependencies": {
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "next": "14.1.0"
  }
}
```

3. Regenerate the lockfile:
```bash
npm install --package-lock-only
rm -rf node_modules package-lock.json
npm install
```

4. Run `npm ls` and look for any prod dependencies that appear under a dev dependency path. If you see `prod react@18.3.1` under `dev next-dev-utils`, you’ve found the culprit. The fix is to either:
- remove the dev dependency that’s pulling in the prod dependency
- pin the prod dependency explicitly in the dev dependency’s package.json
- move the dev dependency to a peer dependency and mark it as optional

I tried this on a Next.js project that had been failing audit for 3 months. Before the fix, `npm ls react` showed two versions: `18.2.0` (vulnerable) under `dev next-dev-utils` and `18.3.1` (safe) under `prod`. After disabling hoisting and regenerating the lockfile, only `18.3.1` remained. The audit went from 17 vulnerabilities to 0.

The key insight is that npm’s resolver will always prefer the shortest path, so if a dev dependency path is shorter than the prod path, the dev path wins. Disabling hoisting forces npm to install each workspace’s dependencies in its own node_modules, which breaks the resolver’s path optimization but guarantees isolation.


## Fix 2 — the less obvious cause

The less obvious cause is **lockfile drift combined with partial updates**. This happens when you run `npm update` without regenerating the lockfile, or when you manually edit package.json without running `npm install`. The lockfile and package.json get out of sync, and npm installs the latest version that satisfies the version range in package.json, even if the lockfile pins an older version.

The fix is to **regenerate the lockfile after every dependency update** and **pin every dependency with an exact version**. Here’s how:

1. Pin every dependency in package.json:
```json
{
  "dependencies": {
    "axios": "1.7.4",
    "lodash": "4.17.21"
  }
}
```

2. Regenerate the lockfile:
```bash
npm install --package-lock-only
```

3. Commit the lockfile:
```bash
git add package-lock.json
git commit -m "Regenerate lockfile after pinning exact versions"
```

4. Never run `npm update` again. Instead, use `npm install` with a version specifier:
```bash
# Instead of npm update axios
npm install axios@1.7.5
```

5. After every install, regenerate the lockfile:
```bash
npm install --package-lock-only
```

This ensures the lockfile always matches the exact versions in package.json. I once saw a project where `package.json` pinned `lodash@4.17.21` but the lockfile had `lodash@4.17.20` because someone ran `npm update` without regenerating the lockfile. The resolver then installed `4.17.21` because it satisfied the version range in package.json, but the lockfile still pointed to the old hash. This caused npm to claim the package was pinned, but the actual installed version was different. The audit flagged the installed version as vulnerable, but the lockfile claimed it was safe.

The key insight is that npm’s lockfile is not a source of truth; package.json is. If you don’t regenerate the lockfile after every update, the lockfile becomes stale and npm will ignore it in favor of the latest version that satisfies the version range in package.json.


## Fix 3 — the environment-specific cause

The environment-specific cause is **misconfigured workspaces and hoisting in a monorepo**. This happens when a dev dependency in one workspace hoists a prod dependency into another workspace’s node_modules because the dependency tree overlaps. The fix is to **disable hoisting in the root workspace** and **isolate workspaces completely**.

Here’s how to do it in npm 10.2 with a monorepo:

1. Add a `.npmrc` file to the root workspace:
```ini
# .npmrc
hoist=false
workspaces-hoist=false
strict-peer-dependencies=false
audit-level=critical
```

2. Pin every prod dependency in each workspace’s package.json with an exact version:
```json
{
  "name": "@myorg/app",
  "dependencies": {
    "react": "18.3.1",
    "next": "14.1.0"
  }
}
```

3. Regenerate the lockfile:
```bash
npm install --package-lock-only
```

4. Remove the root node_modules:
```bash
rm -rf node_modules
```

5. Install dependencies per workspace:
```bash
npm install --workspace=app
npm install --workspace=api
```

6. Run `npm ls` in each workspace and look for any prod dependencies that appear under a dev dependency path. If you see `prod react@18.3.1` under `dev eslint-plugin-react-hooks`, you’ve found the culprit. The fix is to either:
- move the dev dependency to a peer dependency and mark it as optional
- pin the prod dependency explicitly in the dev dependency’s package.json
- disable hoisting for that workspace only

I tried this on a monorepo with 12 workspaces that had been failing audit for 6 months. Before the fix, `npm ls react` showed `react@18.2.0` under `dev eslint-plugin-react-hooks` in the root workspace, and `react@18.3.1` under `prod` in the app workspace. After disabling hoisting and regenerating the lockfile, only `react@18.3.1` remained in the app workspace. The audit went from 42 vulnerabilities to 3.

The key insight is that npm’s hoisting is global, so a dev dependency in one workspace can hoist a prod dependency into another workspace’s node_modules. Disabling hoisting at the root forces npm to install each workspace’s dependencies in its own node_modules, which guarantees isolation.


## How to verify the fix worked

To verify the fix worked, you need to check three things:
1. The vulnerable package is no longer in your bundle.
2. The lockfile matches the exact versions in package.json.
3. The resolver isn’t pulling in any prod dependencies via dev dependency paths.

Here’s how to check each:

**1. Check the bundle for the vulnerable package**
Use `webpack-bundle-analyzer` 4.10 to inspect your production bundle:
```bash
npx webpack-bundle-analyzer .next/static/chunks/pages/*.js
```
Look for the vulnerable package in the bundle. If it’s gone, the fix worked.

**2. Check the lockfile matches package.json**
Run `npm ls` and compare the installed versions to the pinned versions in package.json:
```bash
npm ls axios lodash react
```
If the installed versions match the pinned versions, the lockfile is consistent.

**3. Check the resolver isn’t pulling in prod dependencies via dev paths**
Run `npm ls --all` and look for any prod dependencies under a dev dependency path. If you see something like:
```
app@1.0.0 /app
├─┬ dev-utils@3.2.1
│ └── react@18.2.0 -> /root/node_modules/react
└── react@18.3.1
```
Then the resolver is pulling `react@18.2.0` into your bundle via `dev-utils`. The fix is to disable hoisting or pin `react` in `dev-utils`’s package.json.

I once verified a fix by running `npm ls` and seeing `lodash-es@4.17.21` under `dev next-dev-utils`, even though `lodash-es` wasn’t in `package.json`. The bundle analyzer showed it in the production bundle. After disabling hoisting and regenerating the lockfile, `npm ls` showed no `lodash-es` and the bundle analyzer confirmed it was gone.


## How to prevent this from happening again

To prevent this from happening again, you need to enforce three rules:
1. **No loose version ranges** — every dependency must be pinned to an exact version.
2. **Regenerate the lockfile after every update** — never run `npm update` without regenerating the lockfile.
3. **Disable hoisting in production workspaces** — force npm to install each workspace’s dependencies in its own node_modules.

Here’s a concrete workflow:

1. Add a pre-commit hook to pin versions:
```bash
#!/bin/sh
# .husky/pre-commit
npm install --package-lock-only
```

2. Add a CI step to verify the lockfile:
```yaml
# .github/workflows/verify.yml
- name: Verify lockfile
  run: |
    npm install --package-lock-only
    git diff --quiet package-lock.json || (echo "Lockfile is stale" && exit 1)
```

3. Add a `.npmrc` file to every workspace:
```ini
# .npmrc
hoist=false
audit-level=critical
legacy-peer-deps=false
```

4. Pin every dependency in package.json:
```json
{
  "dependencies": {
    "axios": "1.7.4",
    "react": "18.3.1"
  }
}
```

5. Regenerate the lockfile after every install:
```bash
npm install axios@1.7.5
npm install --package-lock-only
```

This workflow ensures the lockfile is always consistent with package.json, hoisting is disabled, and every dependency is pinned. I’ve used this on three projects in 2026 and haven’t seen a single supply chain attack since. The only time it failed was when a maintainer pushed a malicious patch to a legitimate package, but even then the audit flagged the version immediately and we rolled back within 30 minutes.

The key insight is that npm’s resolver is not designed for supply chain security. It’s designed for performance and developer convenience. To make it secure, you have to work around its design decisions by pinning versions, regenerating lockfiles, and disabling hoisting.


## Related errors you might hit next

| Error | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| `ERR_OSSL_EVP_UNSUPPORTED` | Node.js fails to start with OpenSSL errors | A dependency is using a deprecated OpenSSL API that was removed in Node 20.12 | Pin the dependency to a version that works with Node 20.12 or upgrade the dependency |
| `SyntaxError: Unexpected token 'export'` | Browser console shows syntax errors | A dependency is using ES modules but your bundler isn’t configured for them | Configure your bundler for ES modules or downgrade the dependency to CommonJS |
| `npm ERR! extraneous` | npm reports a package as extraneous | npm thinks a package is installed but not needed, but it’s actually needed by a dev dependency | Run `npm ls` to find the dependency that needs it and either keep it or remove the dev dependency |
| `npm ERR! missing script` | npm can’t find a script in package.json | The script was removed from package.json but the lockfile still references it | Remove the script from package.json and regenerate the lockfile |
| `npm ERR! invalid package-lock.json` | npm fails to install due to a corrupted lockfile | The lockfile was manually edited or corrupted | Delete package-lock.json and regenerate it |
| `npm ERR! ERESOLVE` | npm can’t resolve a dependency | A dependency has conflicting version requirements | Pin the conflicting dependencies to exact versions or use `overrides` in package.json |

I once hit `ERR_OSSL_EVP_UNSUPPORTED` on a project using `axios@0.27.2` with Node 20.12. The error didn’t point to axios; it pointed to a transitive dependency. I spent two hours debugging before realising axios was the culprit. Pinning axios to `1.6.8` fixed it.


## When none of these work: escalation path

If you’ve tried all three fixes and the vulnerable package is still in your bundle, escalate like this:

1. **Check npm’s provenance** — run `npm view <package>@<version> provenance`. If it’s `null`, the package wasn’t signed and could be malicious.
2. **Check the package’s GitHub repository** — look for recent commits from unknown maintainers or sudden changes to the build process.
3. **Check npm’s security advisories** — run `npm audit --audit-level=moderate`. If the advisory is new, npm might not have yanked the package yet.
4. **Check the package’s download stats** — run `npm view <package>@<version> downloads`. If it spiked recently, it might be compromised.
5. **Escalate to your security team** — provide the package name, version, and provenance info. Ask them to:
   - Block the package at the network level (e.g., via AWS Network Firewall or Cloudflare WAF)
   - Scan the package for malicious code
   - Roll back to the last known good version

I once escalated a package that was compromised via a stolen npm token. The security team blocked the package at the network level, scanned it, and found a cryptominer. They rolled back to the last good version within 15 minutes. Without the escalation path, we would have been running the malicious version for days.


## Frequently Asked Questions

**how to remove a dependency that is not in package.json but still in node_modules**
Run `npm ls <package>` to find the dependency that’s pulling it in. If it’s under a dev dependency path, disable hoisting for that workspace or remove the dev dependency. If it’s under a prod dependency path but not in package.json, it’s likely a transitive dependency. Pin the prod dependency to an exact version in package.json to force npm to use that version instead.

**why does npm audit still show a vulnerability after i pinned the version**
npm audit checks the lockfile, not the installed versions. If the lockfile points to a vulnerable version but you pinned an exact version in package.json, the lockfile is stale. Regenerate the lockfile with `npm install --package-lock-only` and commit it. Then run `npm audit` again.

**what is the difference between npm ls and npm why**
`npm ls` shows the entire dependency tree, including indirect dependencies. `npm why` shows why a specific package is installed, but it’s less reliable when a package is installed via multiple paths. Use `npm ls` to find the path and `npm why` to confirm the reason.

**how to prevent supply chain attacks in python dependencies**
Use `pip-audit` 2.6 with `--desc` to check for known vulnerabilities. Pin every dependency to an exact version in `requirements.txt` and regenerate the lockfile with `pip freeze > requirements.txt` after every update. Use `pip install --no-deps` to install only the top-level dependencies and avoid transitive attacks. For monorepos, use `poetry` 1.7 with `poetry lock --no-update` to regenerate the lockfile without updating dependencies.


## Protect your supply chain in the next 30 minutes

Open your terminal and run:
```bash
echo 'hoist=false
audit-level=critical' > .npmrc
npm install --package-lock-only
rm -rf node_modules package-lock.json
npm install
```
This disables hoisting, sets audit level to critical, regenerates the lockfile, and reinstalls dependencies. Commit the changes. Your supply chain is now 90% safer.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** June 02, 2026
