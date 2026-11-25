# Codebase due diligence: read the tests first

Interview loops optimize for the wrong signal. You spend hours on algorithm puzzles and system design whiteboard sketches, then accept an offer based on a vibe from four people who have all rehearsed the same pitch. The codebase you will actually maintain for the next two years gets maybe fifteen minutes of your attention — usually a glance at the README and a quick scroll through the repo tree.

The problem is that a codebase's health is not visible from the README. It is visible in the tests, the CI configuration, the dependency graph, and the commit history. Those are the artifacts that tell you whether you are joining a team that ships or a team that spends its Fridays firefighting. This post gives you a repeatable way to read those artifacts before you sign, so you can ask better questions and make a decision based on evidence rather than the interviewer's enthusiasm.

## The README lies, the test suite doesn't

Every README says the project is well-tested and easy to set up. That is not a lie exactly — it is aspirational documentation written by someone who believed it at the time. The test suite, on the other hand, is a fossil record of what the team actually cares about. It shows you which modules get attention, which edge cases are considered real, and whether tests are treated as a first-class deliverable or an afterthought bolted on before a release.

Start with the test directory structure. If tests mirror the source tree (`src/auth/` has a corresponding `test/auth/`), that is a good sign — it suggests tests are written alongside features. If there is a single `tests/` folder with a handful of files covering only the happy path of the most recent feature, you are looking at a team that tests reactively, usually after a production incident.

Look at the ratio of test files to source files, but do not treat it as a score. A codebase with 80% test coverage of trivial getters is worse than one with 40% coverage concentrated on the payment reconciliation logic. The question is not "how many tests" but "what do these tests protect?" Find the most business-critical module — payments, auth, data ingestion — and check whether it has tests that exercise failure paths, not just success paths. A test suite that only asserts the happy path is documentation of intent, not a safety net.

One more signal: do the tests use mocks for everything? Heavy mocking of internal modules often means the code is tightly coupled and hard to test in isolation. Tests that hit a real database (even a containerized one) or use fixtures are usually a sign the team has invested in test infrastructure. Neither approach is universally right, but a team that mocks every dependency is often a team that has given up on integration testing.

## The dependency graph is a liability ledger

Open `package.json`, `go.mod`, `requirements.txt`, or whatever manifest the project uses. Count the direct dependencies. Now count the transitive ones — most lockfiles will tell you. That number is not inherently bad, but it is a proxy for how much surface area you are trusting to strangers.

A project with 40 direct dependencies and 600 transitive ones is not automatically worse than one with 5 direct dependencies. The question is whether the team knows why each dependency is there. Look for dependencies that duplicate functionality: two HTTP clients, three date libraries, two testing frameworks. That pattern suggests dependencies were added ad hoc by different contributors without a review process. It also means you will inherit the maintenance burden of upgrading all of them.

Check the age of the lockfile. If the last lockfile update was six months ago and the project is actively developed, dependencies are drifting. That is not always a crisis — some teams deliberately pin versions — but it means security patches are not flowing. Look at whether there is a bot (Dependabot, Renovate) configured to open upgrade PRs. If there is, check whether those PRs are actually merged or whether they pile up ignored. An ignored Dependabot queue is a strong signal that maintenance work is deprioritized.

The counterintuitive part is that a very small dependency graph can also be a warning. If the team has hand-rolled its own HTTP client, ORM, and date parser to avoid dependencies, you are inheriting a lot of bespoke code that nobody outside the team understands. The right question is not "how many dependencies" but "is there a coherent policy, and is it followed?"

## Reading CI is reading the team's priorities

Continuous integration configuration is the most honest document in any repository. It shows what the team considers a merge blocker. If CI runs only a linter and a build, tests are optional. If it runs tests but not on every branch, the team is comfortable merging untested code. If it runs a full matrix of tests, security scans, and deployment previews, you are looking at a team that has invested in guardrails.

Open the CI config file (`.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`, etc.) and answer these questions:

- What triggers a run? Every push, or only pull requests to main?
- What steps are required to pass? Are tests required, or can they be skipped?
- Is there a deploy step? If so, is it automated or manual?
- How long does a typical run take? (You can infer from timeout settings or caching configuration.)

A CI pipeline that takes 45 minutes to run is a pipeline people will bypass. Look for signs of caching (dependency caches, build caches) and parallelism. A team that has optimized CI has done so because they feel the pain of slow feedback — that is a good sign.

Also check the failure history. If you have access to the repository, look at recent CI runs. A high failure rate on the main branch is a red flag: it means the team is comfortable with a broken build, which erodes trust in the pipeline. If every run is green, that is either a very healthy team or a pipeline that does not actually test anything meaningful.

## The commit history tells you how work actually happens

Git history is a narrative of the team's development process. You can read it like a log file. Start with `git log --oneline -50` and look at commit message quality. Messages like "fix", "update", "wip" suggest a team that does not value traceability. Messages that reference ticket numbers and describe the change ("fix: handle null user in payment webhook (#1234)") suggest a team that has a review process and cares about future maintainers.

Next, look at commit size. Large commits that touch dozens of files are harder to review and harder to revert. A healthy codebase tends to have smaller, focused commits. You can approximate this with `git log --stat` and scanning the summary lines. If most commits touch 20+ files, the team is probably batching work, which often means reviews are superficial.

Branching strategy matters too. A long-lived `develop` branch with many feature branches suggests a GitFlow-style process that can lead to integration pain. A trunk-based approach with short-lived branches and frequent merges suggests a team that prioritizes continuous integration. Neither is universally better, but the former often correlates with release trains and merge conflicts, while the latter correlates with feature flags and faster feedback.

Finally, look at the ratio of commits to merges. If the history is full of merge commits from pull requests, the team uses PRs. If it is a linear history of direct pushes, the team may not have a review process at all. You can check with `git log --merges --oneline | wc -l` and compare to total commits. A team that never merges PRs is a team where code review is optional.

## A worked example: reading a repository in 20 minutes

Assume you have been given read access to a repository before your final interview. Here is a sequence you can run in a terminal, with reasoning for each step. The commands are illustrative; adapt to the project's language and tooling.

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd <repo>

# 2. Get a sense of size and activity
git log --oneline | wc -l          # total commits
git shortlog -sn --all | head -20  # top contributors

# 3. Check test coverage configuration (if any)
ls -a | grep -i coverage           # .coveragerc, .nycrc, etc.
find . -name "*test*" -type d | head

# 4. Inspect CI configuration
ls .github/workflows/ 2>/dev/null || ls .gitlab-ci.yml 2>/dev/null

# 5. Look at dependency manifests
cat package.json 2>/dev/null | jq '.dependencies | length'
cat go.mod 2>/dev/null | grep -c '^\s'  # rough count

# 6. Check recent commit messages
git log --oneline -20

# 7. See how often tests are updated alongside source
git log --name-only --pretty=format: -- '*test*' | sort | uniq -c | sort -rn | head
```

After running these, you should have a rough picture. For example, if you see 2,000 commits, 15 contributors, a `.github/workflows/test.yml` that runs on every push, a `package.json` with 30 direct dependencies and a lockfile updated last week, and commit messages that reference issue numbers, that is a team with process. If you see 200 commits, 2 contributors, no CI directory, and commit messages like "stuff", that is a different story.

The goal is not to judge the team harshly — early-stage projects often lack process. The goal is to calibrate your expectations and generate specific questions. "I noticed the CI only runs on main; how do you handle testing feature branches?" is a better question than "Do you have tests?"

## Failure modes: when a clean repo hides a messy reality

A polished repository can be misleading. Some failure modes to watch for:

- **The abandoned monorepo.** A large repository with many packages, but the last commit to most of them is years old. The active code is a small corner. You will be expected to maintain the rest.
- **The test suite that always passes.** Tests exist, but they are so heavily mocked or so shallow that they cannot fail. Look for assertions that only check for non-null or that mock the function under test. A test suite with 100% pass rate over hundreds of commits is suspicious.
- **The documentation drift.** The README describes a setup process that no longer works. This suggests the team does not onboard new members often, or does not value reproducibility. You will spend your first week fighting the environment.
- **The single maintainer.** One contributor accounts for 80% of commits. That person is a bus factor of one. If they leave, the project stalls. Check `git shortlog -sn` to see distribution.
- **The force-push culture.** If the main branch history is frequently rewritten, it means the team prioritizes a clean history over traceability. That can be fine, but it also means you cannot rely on git bisect or historical context.

None of these are automatic disqualifiers. They are conversation starters. The point is to go into your final interview with specific observations, not generic enthusiasm.

## When this approach is the wrong choice

Reading a codebase before joining is not always possible or useful. If you are joining a team that is about to start a greenfield project, the existing repository may be irrelevant. If the company is under strict confidentiality before signing, you may not get access at all. And if you are joining as a manager or designer, the codebase may be less important than the team dynamics.

Also, beware of over-indexing on code quality. A slightly messy codebase with a healthy team and a clear product direction is often a better bet than a pristine codebase maintained by a burned-out team. The code is a means, not an end. Use it as one input among many, and weight it according to your role. If you will be writing code daily, it matters a lot. If you will be managing, it matters less.

Finally, remember that codebases change. A team that has poor practices today may be actively improving. Look for signs of investment: recent refactoring commits, new CI steps, documentation updates. A trajectory matters more than a snapshot.

## Frequently Asked Questions

**How do I evaluate a codebase if I can't see it before accepting an offer?**

Ask for a code walkthrough during the interview process. Many companies will let you pair with an engineer on a small task or review a pull request. If they refuse, that itself is a signal. You can also ask specific questions about their testing and deployment process and listen for whether the answers are concrete or vague.

**What is the most important file to check in a new codebase?**

The CI configuration. It reveals what the team considers non-negotiable. A close second is the test directory structure, because it shows where the team invests in safety.

**How can I tell if a codebase is well-maintained?**

Look at the age of the lockfile, the frequency of dependency updates, and the ratio of open issues to closed ones. Also check whether there is a recent commit that is not a feature — refactoring, documentation, or test improvements. A well-maintained codebase has ongoing maintenance work, not just feature work.

**Should I avoid a codebase with no tests?**

Not necessarily. Early-stage projects often lack tests. The question is whether the team acknowledges the gap and has a plan to address it. If they are proud of having no tests, that is a red flag. If they are embarrassed and actively working on it, that is a sign of self-awareness.

## Your next step: run the 20-minute audit

Before your next interview, ask for read access to the repository (or a sample of it). If granted, run the commands in the worked example above. If not, ask the hiring manager for a 15-minute screen-share walkthrough of the codebase and request that they show you the CI config and the test directory. Then write down three specific observations and three questions. You will walk into the final interview with evidence, and you will make a decision based on what the code actually says, not what the recruiter promised.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
