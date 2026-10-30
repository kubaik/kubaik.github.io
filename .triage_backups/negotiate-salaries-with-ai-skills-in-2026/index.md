# Negotiate salaries with AI skills in 2026

## The negotiation problem this addresses

Job descriptions and salary bands often lag behind what the role actually involves. A posting may still list "write API specs" or "debug memory leaks" even when those tasks are partly handled by tooling. When a candidate tries to negotiate on that gap, the usual response is a band from an internal compensation framework, and the conversation stalls.

A more productive frame is to separate the tasks in a role into two groups: those that current tooling can substantially accelerate or automate, and those that depend on judgment, ownership, and coordination under constraint. The second group is where negotiation leverage tends to sit, because it is harder to substitute. This article describes a reproducible way to build that argument: a small data-collection pipeline for public job postings, a task-by-task analysis of your own role, and a script for the conversation itself.

Two caveats before starting. First, scraping job boards may violate their terms of service; check the terms for any source you use, and prefer official APIs or manual sampling where they exist. Second, the salary figures you derive are only as good as your sample. Treat them as one input, not as proof.

## What you will build

A minimal Python 3.11 command-line tool that:

- Collects a sample of job postings for a given title and location from sources you are permitted to query
- Tags each posting by whether it mentions AI coding tools or assistants
- Computes median and quartile salary figures for the tagged and untagged groups
- Emits a markdown report you can attach to a counter-offer discussion

You will also produce a one-page artifact mapping each task in your job description to an automation-risk estimate and a human-judgment estimate.

The pipeline runs in a few minutes on a laptop; the negotiation artifact takes longer because it requires honest self-assessment.

## Step 1 — environment setup

```bash
mkdir ai-comp-negotiation && cd ai-comp-negotiation
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

Install dependencies:

```bash
pip install requests==2.31.0 beautifulsoup4==4.12.2 pandas==2.1.3 python-dotenv==1.0.0
```

If you scrape a JavaScript-heavy source, you may need a headless browser such as Playwright:

```bash
pip install playwright==1.40.0
playwright install
```

Create a `.env` file for any credentials or filters you need:

```env
SOURCE_LOCATION="London, UK"
SOURCE_TITLE="Software Engineer"
```

Verify:

```bash
python --version
pip list | grep -E 'requests|beautifulsoup4|pandas|python-dotenv|playwright'
```

Do not commit `.env` to version control. If a source requires a session cookie, treat that cookie as a credential: it grants access to your account.

## Step 2 — collect postings

The example below reads postings from a local JSON file. This keeps the code runnable without depending on any specific site's markup, which changes frequently. Populate the file by exporting results from an official API, or by manually saving postings you are permitted to collect.

```python
# load_postings.py
import json
from pathlib import Path
from typing import List, Dict

def load_postings(path: str = "postings.json") -> List[Dict]:
    """Load postings from a JSON array of {title, location, salary, description}."""
    data = json.loads(Path(path).read_text())
    required = {"title", "location", "salary", "description"}
    for i, row in enumerate(data):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {i} missing keys: {missing}")
    return data
```

A minimal `postings.json`:

```json
[
  {
    "title": "Backend Engineer",
    "location": "London, UK",
    "salary": "72000",
    "description": "Own the payments service. Align with stakeholders on roadmap."
  },
  {
    "title": "Backend Engineer",
    "location": "London, UK",
    "salary": "63000",
    "description": "Use Copilot to accelerate feature delivery."
  }
]
```

Next, tag postings by whether they mention AI tooling:

```python
# tag.py
from typing import List, Dict

AI_KEYWORDS = {
    "copilot", "cursor", "codeium", "tabnine",
    "ai pair programmer", "llm", "large language model",
    "automated code review", "ai assistant", "ai reviewer",
}

def tag_ai_mention(postings: List[Dict]) -> List[Dict]:
    for p in postings:
        text = p["description"].lower()
        p["mentions_ai_tool"] = any(k in text for k in AI_KEYWORDS)
    return postings
```

Keyword matching is crude. "LLM" inside "LLM-adjacent" and "AI" inside "said" are both false positives; word-boundary matching or a small classifier reduces this. Report the false-positive rate you observe so the reader can judge the sample.

## Step 3 — compute the comparison

```python
# report.py
import pandas as pd
from pathlib import Path
from typing import Dict, List

def summarize(postings: List[Dict]) -> Dict:
    df = pd.DataFrame(postings)
    df["salary_num"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.dropna(subset=["salary_num"])

    ai = df[df["mentions_ai_tool"]]
    non_ai = df[~df["mentions_ai_tool"]]

    def quartiles(s: pd.Series) -> Dict[str, float]:
        return {
            "n": int(s.count()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
        }

    return {
        "all": quartiles(df["salary_num"]),
        "mentions_ai_tool": quartiles(ai["salary_num"]),
        "no_ai_mention": quartiles(non_ai["salary_num"]),
    }

def write_markdown(stats: Dict, path: str = "report.md") -> None:
    lines = ["| Group | n | p25 | p50 | p75 |", "|---|---|---|---|---|"]
    for label, s in stats.items():
        lines.append(
            f"| {label} | {s['n']} | {s['p25']:,.0f} | {s['p50']:,.0f} | {s['p75']:,.0f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")
```

Run it:

```bash
python -c "
from load_postings import load_postings
from tag import tag_ai_mention
from report import summarize, write_markdown
posts = tag_ai_mention(load_postings())
stats = summarize(posts)
write_markdown(stats)
print(stats)
"
```

### How to interpret the output honestly

The difference between the two medians is a correlation in your sample, not a causal estimate of a "premium." Postings that mention AI tools may differ in seniority, company size, sector, or region. Before using the number, check:

- **Sample size.** With fewer than ~30 postings per group, quartiles are noisy. Report n alongside every figure.
- **Currency and period.** Mixed currencies must be converted at a stated rate and date, or dropped.
- **Title drift.** "Backend Engineer" at one company may be "Software Engineer II" at another. Restrict to exact titles or a small, stated set.
- **Selection bias.** Postings that list salary are not a random sample of postings.

If the two groups differ on seniority, the salary gap may reflect seniority, not AI-tool mentions. A simple check: split each group by a seniority keyword ("senior", "staff", "principal") and compare within strata.

## Step 4 — handle common failure modes

**Rate limiting.** If a source returns HTTP 429, back off. A retry helper:

```python
import time
import requests

def get_with_backoff(url, headers, params, attempts=5):
    delay = 30
    for i in range(attempts):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("Exhausted retries after repeated 429 responses")
```

**Malformed salary strings.** Normalize before parsing:

```python
import re

def clean_salary(s: str) -> float:
    s = s.replace("a year", "").replace("per annum", "").strip()
    match = re.search(r"(\d{1,3}(?:,\d{3})+(?:\.\d{2})?)", s)
    if match:
        return float(match.group().replace(",", ""))
    return 0.0
```

**Empty results.** If a location filter returns nothing, widen it (city to country) and re-run, but note the change in the report so the sample is not silently altered.

**Keyword drift.** New tools appear constantly. Keep the keyword set in a separate file and record the version used for any report you share.

## Step 5 — the negotiation artifact

The data is one half. The other half is a task-level analysis of your own role. Build a table like this:

| Task | Automatable today? | Human judgment required? | Evidence |
|---|---|---|---|
| Write API reference docs | Mostly | Low | Tooling drafts from signatures |
| Design service boundaries | Partly | High | Trade-offs depend on org constraints |
| Debug memory leaks | Partly | Medium | Profilers narrow the search; fix requires judgment |
| Own incident response | No | High | Time pressure, cross-team coordination |
| Stakeholder alignment on roadmap | No | High | Requires trust and negotiation |

Fill the "Evidence" column with something concrete: a tool you have used for that task, a documented limitation, or a specific incident. Vague claims weaken the argument.

Then compute two ratios from the table:

- `human_heavy = (rows where Human judgment required is High) / total rows`
- `automatable_high = (rows where Automatable today is Mostly) / total rows`

These are illustrative formulas, not market indices. They give you a sentence: "N of M responsibilities in this role are judgment-heavy and not currently automatable with the tooling the team uses."

## A worked negotiation example

The following is illustrative, not a reported outcome.

Suppose the offer is 65,000 and your sample's non-AI median for the title is 72,000 (n=40), while the AI-mention median is 63,000 (n=35). The gap is 9,000, or about 14% of the lower figure.

Do not lead with the gap. Lead with the task analysis. A script:

> "I want to make sure we are aligned on scope. The role as described covers X, Y, and Z. Based on my experience, X and Y are the parts where I add the most value because they depend on judgment under deadline rather than tooling. I have a sample of comparable postings for this title and location; the median for roles that do not emphasize AI assistance is around 72,000, versus 63,000 for roles that do. Given the scope here, I am targeting 72,000. Can we get there, or is there a structure — a signing component, a six-month review — that closes the gap?"

Notice what the script does not do: it does not claim a causal premium, does not assert that the recruiter's band is wrong, and offers a non-base-salary path. If the recruiter cites a budget, ask which parts of the scope are negotiable. If they cite the AI tools the team uses, ask what fraction of the role those tools cover — a question the task table already answers.

## Adding tests and observability

A small pytest suite catches regressions when you change the keyword set or parsing:

```python
# test_pipeline.py
from tag import tag_ai_mention
from report import summarize

def test_tagging():
    posts = [
        {"title": "E", "location": "L", "salary": "100", "description": "Use Copilot daily."},
        {"title": "E", "location": "L", "salary": "200", "description": "Own the roadmap."},
    ]
    tagged = tag_ai_mention(posts)
    assert tagged[0]["mentions_ai_tool"] is True
    assert tagged[1]["mentions_ai_tool"] is False

def test_summary_counts():
    posts = [
        {"title": "E", "location": "L", "salary": "100", "description": "Copilot", "mentions_ai_tool": True},
        {"title": "E", "location": "L", "salary": "200", "description": "Own", "mentions_ai_tool": False},
    ]
    stats = summarize(posts)
    assert stats["mentions_ai_tool"]["n"] == 1
    assert stats["no_ai_mention"]["n"] == 1
```

Log every run with the source, query, timestamp, and counts, so a report you share can be reproduced:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

## Common questions

**Why not just use a public salary aggregator?**
Aggregators are useful for a coarse range. They generally do not let you filter by whether a posting mentions AI tooling, so you cannot reproduce the specific comparison this method relies on. Use them to sanity-check your sample, not to replace it.

**What if the employer says AI tooling is optional?**
Then the relevant question is which tasks the tooling covers, not whether it is allowed. The task table answers that. If tooling covers little of the role, the AI-mention comparison is less relevant to your case; if it covers a lot, the scope conversation is more important than the salary comparison.

**How do I handle equity and bonuses?**
Convert everything to a single number with stated assumptions. For equity, state the vesting schedule, the current valuation, and a discount you apply for illiquidity and dilution. Show the arithmetic. Do not present a discounted figure as a guaranteed value.

**What about non-English postings?**
Keyword matching is language-specific. Maintain a separate keyword set per language and note which was used. Machine translation of postings before tagging adds error; state the error rate you observe on a manual sample.

**Is a 14% gap universal?**
No. It is a property of a specific sample at a specific time. Report your sample size, date, source, and filters. If someone challenges the number, the challenge is answerable — that is the point of building it this way.

## What to do in the next 30 minutes

Open your current job description and list every responsibility as a single row. For each, write one sentence of evidence for whether it is automatable today and whether it requires human judgment. Save it as `human_tasks.md`. That file, not the scraper, is the core of the negotiation — the data pipeline only supports it.

Then, before your next compensation conversation, collect at least 30 postings for your exact title and location from a source whose terms permit it, tag them, and compute the two medians. Report n for each group. If the difference is under 5% or either group has fewer than 30 postings, do not lead with the number; lead with the task table instead.
