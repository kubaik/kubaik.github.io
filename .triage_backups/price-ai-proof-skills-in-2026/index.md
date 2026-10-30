# Price AI-proof skills in 2026

Tutorials on AI-assisted engineering tend to show the happy path: a prompt, a working function, a merged pull request. What they rarely cover is the compensation conversation that follows, when a title changes, a productivity claim is made on someone else's behalf, or a budget line is reallocated. This article is about preparing for that conversation with evidence rather than assertion.

## Why AI-era compensation conversations are harder

Three structural problems show up repeatedly when engineers with AI-adjacent titles negotiate pay.

**Title drift outpaces band updates.** Job descriptions now routinely include titles such as "AI Engineer", "ML Platform Engineer", or "Prompt Architect". Internal compensation bands, however, are usually revised on an annual or semi-annual cycle, and often map new titles onto existing engineering ladders without adjustment. The result is that a title change can be compensation-neutral or even compensation-negative. Before accepting any retitle, ask which existing band the new title maps to and whether that mapping has been ratified by whoever owns the compensation framework.

**Productivity claims are measured on the wrong axis.** A demo where a model generates a working script in seconds is genuinely impressive, but it measures generation speed on a matched prompt, not the cost of the whole task. The parts that dominate real delivery time — reproducing a race condition under load, deciding which of two plausible designs is correct, reviewing a change against a compliance regime — are usually not visible in that demo. When an organisation ties raises to a proxy metric such as "lines of code removed" or "PRs opened", the proxy can move in the opposite direction from the value delivered. A defensible counter is to measure the same task both ways and show the divergence.

**Budget cycles lag headcount reallocation.** When a company shifts headcount budget toward AI tooling, the new roles often launch at frozen or provisional bands until the next planning cycle. Engineers who move into those roles early can be locked into a band below the market rate for the work they are actually doing. The remedy is not to refuse the role but to document the scope and revisit the band at the first scheduled cycle, with evidence.

None of these problems is solved by arguing about productivity in the abstract. They are solved by producing artefacts that a manager, a compensation partner, and a finance reviewer can each independently check.

## What you will build

The output of this process is a **compensation evidence pack**: a small, version-controlled repository containing three things.

1. A title-normalised salary benchmark in CSV form, derived from sources you can cite.
2. A short narrative document that maps your non-automatable work to business outcomes.
3. A negotiation script you can paste into a one-to-one document or a message thread.

The tooling is deliberately lightweight. If the raw data is already available, the mechanical work takes under two hours. The thinking — deciding what counts as non-automatable — takes longer and is the part that actually matters.

## Prerequisites

- A Git repository (GitHub, GitLab, or Bitbucket) with a meaningful commit history over the last twelve months, or access to a team repository if your own is thin.
- Access to whatever your organisation uses to record outcomes: OKRs, a metrics dashboard, incident records, quarterly review slides, or a finance summary.
- Python 3.12 or Node 20 LTS to run the normalisation scripts.

Before writing any scraper, check the target site's `robots.txt`, terms of service, and rate limits. Public compensation aggregators frequently block automated access, and a scraper that gets your account suspended is worse than no scraper. Prefer an official API where one exists, and cache aggressively when it does.

## Step 1 — Set up the environment

Create a virtual environment and install the core stack:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install pandas requests-cache python-dotenv
```

Pin versions in a `requirements.txt` so the pack is reproducible:

```text
pandas==2.2.2
requests-cache==1.2.1
python-dotenv==1.0.1
```

Store credentials in a `.env` file that is never committed:

```env
COMP_API_KEY=your_token_here
GITHUB_TOKEN=ghp_your_token
```

Add `.env` to `.gitignore` before the first commit. An evidence pack that leaks a token is a liability, not an asset.

## Step 2 — Normalise titles before comparing salaries

The single most common error in compensation research is comparing raw titles. "AI Engineer" at one company is a product-facing role; at another it is a research role; at a third it is a rebranded backend position. Normalise first, then compare.

The script below reads a JSON payload from a compensation source and maps titles onto your own internal bands. The exact endpoint and response shape depend on the provider you use, so treat the URL and the `levels` key as placeholders to adapt.

```python
import os
import pandas as pd
import requests
import requests_cache
from datetime import datetime, timezone

requests_cache.install_cache('comp_cache', expire_after=3600)

API_KEY = os.environ['COMP_API_KEY']
HEADERS = {'Authorization': f'Bearer {API_KEY}'}

# Adapt this URL and the response key to your actual provider.
url = 'https://api.example-comp-source.com/v1/levels.json'
response = requests.get(url, headers=HEADERS, timeout=30)
response.raise_for_status()

records = response.json()['levels']
df = pd.DataFrame(records)

df['title_clean'] = (
    df['title']
    .str.replace(r'\(.*\)', '', regex=True)
    .str.strip()
)

mapping = {
    'Software Engineer': 'SWE',
    'Senior Software Engineer': 'SWE',
    'AI Engineer': 'SWE',
    'ML Engineer': 'DS',
    'Data Scientist': 'DS',
    'Prompt Engineer': 'SWE',
}

df['band'] = df['title_clean'].map(mapping).fillna('Other')
df['snapshot_utc'] = datetime.now(timezone.utc).isoformat()

df.to_csv('benchmarks.csv', index=False)
print(f"Saved {len(df)} records to benchmarks.csv")
```

Two details matter here. First, the `mapping` dictionary is a judgement call, and it should be documented in the repository so a reviewer can challenge it. Second, recording `snapshot_utc` on every row means the benchmark is dated. Compensation data decays; an undated CSV is not evidence.

### How to measure the title premium properly

Do not rely on a single aggregate figure. To establish whether a title premium exists in your market, instrument the comparison directly:

- Pull a sample of postings or reported salaries for the AI-adjacent title and for the equivalent generalist title.
- Restrict to the same seniority level, the same location, and the same company size band.
- Compute the median for each group and the difference between them.
- Report the sample size alongside the difference. A gap computed from a handful of rows is noise.

The command is trivial once the CSV exists:

```bash
python -c "
import pandas as pd
df = pd.read_csv('benchmarks.csv')
for band, g in df.groupby('band'):
    print(band, len(g), g['total'].median())
"
```

If the AI-adjacent title shows a lower median than the generalist title at the same level, that is a finding worth bringing to the conversation — provided the sample is large enough to defend.

## Step 3 — Measure your own work honestly

Pull your contribution history from the repository host. The GitHub API exposes weekly contribution statistics per repository:

```python
import os
from datetime import datetime, timedelta, timezone
import requests

GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
OWNER = 'your_org_or_username'
REPO = 'your_repo_name'

url = f'https://api.github.com/repos/{OWNER}/{REPO}/stats/contributors'
headers = {'Authorization': f'Bearer {GITHUB_TOKEN}',
           'Accept': 'application/vnd.github+json'}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).timestamp()
weeks = response.json()[0]['weeks']
recent = [w for w in weeks if w['w'] >= cutoff]
total_commits = sum(w['c'] for w in recent)

print(f"Commits in the last 12 months: {total_commits}")
```

Commit count is a weak signal on its own — it is exactly the kind of proxy metric that misleads. Use it as an index into the work, not as the argument. The argument comes from the artefacts: the incident that was resolved, the migration that was completed, the review that caught a problem.

### Separating AI-assisted from human-only work

For each significant piece of work in the period, record four fields:

| Field | What to record | Why it matters |
|---|---|---|
| Task | A one-line description of the outcome, not the activity | Reviewers can verify outcomes; activity descriptions invite debate |
| AI contribution | What the model produced, and under what conditions | Establishes the boundary of the automation |
| Human contribution | The decisions, debugging, or review that the model did not perform | This is the compensable part |
| Verifiable impact | A metric from a system of record, with a link | Removes the conversation from opinion |

The "AI contribution" column is where most evidence packs go wrong. If a model generated a function that worked on the first attempt, say so. Overstating human contribution is the fastest way to lose credibility with a manager who has seen the same demo. The claim being made is narrower and stronger: *the model handled the generation, and the human handled the parts where the model was unreliable.*

A typical failure mode looks like this. A team adopts a code assistant, measures a large speed-up on a scripted task, and extrapolates to the whole role. The extrapolation breaks down because the scripted task had a fixed specification and a known-correct answer, while the role consists largely of tasks where the specification is contested and correctness is only established after deployment. Documenting that distinction, with examples, is the substance of the evidence pack.

## Step 4 — Write the narrative document

The narrative is the part a human reads. Keep it to two pages. Structure it as:

```markdown
# Non-Automatable Work — Evidence Pack

Period: <start> to <end>
Level: <your level>
Band under review: <band>

## Scope
- Primary objective this period: <one sentence, with the metric>
- Systems owned: <list>
- AI tooling in use: <list, with what it was used for>

## Evidence

| Task | AI contribution | Human contribution | Impact (source) |
|------|-----------------|--------------------|-----------------|
| <task> | <what the model did> | <what it could not do> | <metric, system of record> |

## Narrative
<Three short paragraphs: what was hard, what the model could not do,
and what changed in the business as a result.>
```

The table does the work that prose cannot. A manager reading "AI contributed to X; the human resolved Y" is being handed a distinction they can act on. A manager reading "I worked hard on AI projects" is being handed a claim they must evaluate from scratch.

Two rules for the impact column. First, every figure must come from a system of record — a ticketing system, a finance summary, an incident tracker — and the source should be named. Second, if a figure is an estimate, label it as an estimate and show the arithmetic. For example: "reduced manual review by an estimated 30 hours per quarter (400 prompts reviewed, 4.5 minutes saved per prompt, arithmetic shown)". A reviewer who can redo the sum will trust the number; a reviewer handed a bare figure will discount it.

## Step 5 — Handle the awkward cases

**Your organisation does not use OKRs.** Substitute whatever the organisation does use: delivered story points, closed issues, incident counts, or customer-facing release notes. The format matters less than the traceability.

**Your personal repository is thin.** Use the team repository and attribute only the work you can demonstrate. Merged pull requests with your authorship are verifiable; a commit count on a shared branch is not.

**The model genuinely did most of the work.** This is not a failure of the evidence pack; it is information. If the role has largely been automated, the honest move is to negotiate a scope change or a retitle — but only after checking what the new title pays. A retitle into a lower band is a pay cut with better marketing.

**Budget is frozen.** Ask about the mechanisms that sit outside the salary cycle: spot bonuses tied to specific artefacts, equity refresh, additional leave, or a training budget. Whatever is agreed, get it in writing with a date attached. A verbal commitment to "revisit next cycle" is not a commitment.

**The manager says the job description will be automated next year.** Shift the conversation to the parts of the role that are currently ambiguous or require domain judgement: data quality, security review, regulatory compliance, incident response. Those areas are where automation is least reliable, and they are where evidence of human contribution is easiest to produce.

## Step 6 — Make the pack reproducible

An evidence pack that cannot be re-run is a PDF with extra steps. Add tests that assert the shape and the invariants of your data:

```python
import pandas as pd
import pytest

def test_benchmark_file_exists():
    df = pd.read_csv('benchmarks.csv')
    assert len(df) > 0, "Benchmark file is empty"
    assert 'band' in df.columns, "Missing band column"
    assert 'snapshot_utc' in df.columns, "Missing snapshot timestamp"

def test_bands_are_populated():
    df = pd.read_csv('benchmarks.csv')
    unknown = (df['band'] == 'Other').mean()
    assert unknown < 0.5, f"Too many unmapped titles: {unknown:.0%}"
```

The second test is the useful one. If more than half of your rows fall into `Other`, your title mapping is too narrow and the benchmark is not measuring what you think it is. Adjust the mapping before drawing conclusions.

Run the suite:

```bash
pytest tests/ -v
```

Add a CI workflow so the pack is validated on every push:

```yaml
name: Benchmarks CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

Now the pack has a URL, a history, and a green check. When review season arrives, the artefact speaks for itself.

## Decision checklist before the conversation

- [ ] Every salary figure has a source and a date.
- [ ] Titles are normalised, and the mapping is documented and defensible.
- [ ] Sample sizes are reported alongside every median.
- [ ] Every impact figure traces to a system of record, or is labelled as an estimate with the arithmetic shown.
- [ ] The AI contribution is stated honestly, including the cases where the model did most of the work.
- [ ] The pack is in version control and the tests pass.
- [ ] The ask is specific: a band, a number, or a mechanism, with a date.
- [ ] A fallback is prepared for each of the frozen-budget, retitle, and deferred-decision cases.

## FAQ

**Should the benchmark use public data or internal data?**
Both, kept separate. Public data establishes the market rate. Internal data establishes where your organisation sits relative to that market. Conflating them makes the argument unfalsifiable.

**What if the organisation refuses to share bands?**
Build an inferred model from public postings and internal titles, and label it as inferred. Present it as a question — "does this mapping match the internal framework?" — rather than as an assertion. A manager who cannot share the bands can often confirm or deny a mapping.

**Is equity a substitute for a raise?**
It depends on liquidity and vesting. For a private company with no near-term liquidity event, an equity refresh has no verifiable value at the time of negotiation. A spot bonus tied to a documented artefact is comparable across companies; an equity grant is not.

**How long should the pack be?**
Two pages of narrative plus the CSV. If it is longer, the reviewer will not read it. The repository can hold more detail; the document that gets read should not.

## The next 30 minutes

Create the repository, commit the title-normalisation script, and run it against whatever compensation data you can legitimately access. The output does not need to be complete — it needs to exist, with a dated snapshot and a documented mapping. That single artefact converts the next compensation conversation from a discussion about how you feel about your work into a discussion about what the data shows.
