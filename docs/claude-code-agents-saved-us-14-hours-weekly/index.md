# Claude Code agents saved us 14 hours weekly

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

Our team spent 20 hours a week on repetitive maintenance tasks: updating documentation, checking failing tests, rebasing feature branches, and manually verifying that our staging environment mirrored production. These weren’t complex problems, but they blocked developers from shipping features. We tried automating parts of this with GitHub Actions, but the workflows required constant updates whenever our repository structure changed. Bash scripts grew to 400 lines and became a maintenance nightmare. We needed a system that could adapt to changes without breaking—something that could reason about our codebase rather than just execute commands.

That’s when we heard about Claude Code’s agent mode. We’d used Anthropic’s tools for code review and debugging before, but never as an autonomous agent. Could it handle our specific workflows? We decided to test it on the most painful task: keeping documentation up to date with code changes.

Our repository had 150 Python files across 12 microservices. Every time a developer updated a function signature or added a new parameter, we had to manually update:
- The docstring in the function
- The API documentation (OpenAPI spec)
- The internal wiki pages
- The release notes template

We estimated this took 6–8 hours per sprint. We wanted to reduce that to under 30 minutes.

## What we tried first and why it didn’t work

Our first attempt was a Python script using AST parsing to extract function signatures and update docstrings. It worked for simple cases but failed spectacularly on:
- Decorated functions (we use FastAPI decorators)
- Functions with complex type hints (we use Pydantic models)
- Files with multiple classes and inheritance

The script produced 47 incorrect updates in our first test run. Worse, it didn’t handle merge conflicts—if two developers updated the same file, the script would overwrite one change. We tried wrapping it in a GitHub Action with a concurrency lock, but the lock caused timeouts when multiple PRs updated documentation concurrently.

Then we tried a RAG-based approach using our codebase as a vector store. We built a system that:
1. Indexed all Python files
2. Used Claude to query for functions needing updates
3. Generated the new documentation
4. Created PRs with the changes

The first run took 45 minutes and mostly worked, but the PRs were messy. Claude would sometimes:
- Update the wrong function
- Miss edge cases in type hints
- Include unrelated changes in the diff
- Fail to resolve merge conflicts properly

Most critically, we realized the RAG approach couldn’t distinguish between intentional changes (a developer actually updating a function) and accidental ones (a refactor that shouldn’t trigger documentation updates). We had no way to verify intent.

We also tried off-the-shelf tools like Swimm and Mintlify. Swimm required manual setup for each repository and didn’t integrate well with our monorepo. Mintlify’s AI-generated docs were impressive but didn’t auto-update—it only created initial documentation. None of these tools handled our specific branching strategy where documentation updates need to be coordinated with code releases.

After three weeks of trying different approaches, we were back to manual updates. The tools either didn’t work for our codebase or required too much upfront configuration.

## The approach that worked

We pivoted to a **targeted agent workflow** using Claude Code’s new agent mode. The key insight was that we didn’t need a general-purpose documentation updater—we needed an agent that could:
1. Understand our specific code patterns (FastAPI, Pydantic, our internal decorators)
2. Follow our documentation standards (specific docstring format, OpenAPI spec structure)
3. Respect our branching strategy (documentation updates go to a dedicated branch)
4. Handle conflicts gracefully (skip files with conflicts, leave them for manual review)

Here’s the workflow we designed:

```python
# agent_workflow.py
from claude_code_agent import Agent
from codebase_analyzer import CodebaseAnalyzer
from documentation_updater import DocumentationUpdater

class DocumentationAgent(Agent):
    def __init__(self, repo_path):
        super().__init__(
            repo_path=repo_path,
            instructions_file=".claude/documentation_agent_instructions.md"
        )
        self.analyzer = CodebaseAnalyzer(repo_path)
        self.updater = DocumentationUpdater(repo_path)

    def run(self, target_branch="main"):
        # Step 1: Analyze changed files since last update
        changed_files = self.analyzer.get_changed_files(target_branch)
        
        # Step 2: Filter to Python files with public functions/classes
        python_files = [f for f in changed_files if f.endswith('.py')]
        public_items = self.analyzer.extract_public_api(python_files)
        
        # Step 3: Generate documentation updates
        updates = []
        for item in public_items:
            update = self.updater.generate_documentation(item)
            if update:
                updates.append(update)
        
        # Step 4: Apply updates only to matching files
        applied = self.updater.apply_updates(updates)
        
        # Step 5: Create PR if changes were made
        if applied:
            self.updater.create_pr(applied, target_branch)
        
        return {"updates_applied": len(applied), "files_skipped": len(python_files) - len(applied)}
```

The critical difference from our previous attempts was the **narrow scope** of the agent. Instead of trying to handle everything, it focused specifically on:
- Python files with public APIs (skipping internal modules)
- Functions and classes with specific decorators (our FastAPI endpoints)
- Changes that actually modify the signature or behavior

We created a dedicated instruction file that taught the agent:
- Our docstring format (`"""Endpoint to create user with email and password."""`)
- How to handle Pydantic model references
- When to skip a function (internal use, deprecated, test helper)
- How to format OpenAPI operation IDs

The agent ran in our CI pipeline nightly, analyzing changes since the last run. If it found updates needed, it created a PR with the documentation changes. Developers could review the PR, approve it, and merge it—no manual coordination needed.

## Implementation details

Getting the agent to work reliably required several iterations. Here’s what we learned:

### Setting up the agent

We used Claude Code v0.12.0 with the following configuration:

```json
{
  "name": "documentation-agent",
  "version": "0.2.1",
  "model": "claude-sonnet-4-20250411",
  "system_prompt": "You are a precise documentation agent...
  You must only update files that match our documentation patterns...",
  "max_iterations": 15,
  "temperature": 0.1
}
```

The low temperature (0.1) was crucial—it made the agent follow instructions precisely rather than improvising. We set max_iterations to 15 to prevent infinite loops when it got stuck.

### Handling edge cases

The agent’s biggest challenge was distinguishing between:
- A function signature change (needs documentation update)
- A function rename (needs documentation update)
- A comment update (should be ignored)
- A type hint refinement (should be ignored if behavior unchanged)

We solved this by using the `git diff` to see what actually changed in the file:

```python
class CodebaseAnalyzer:
    def get_changed_files(self, target_branch):
        result = subprocess.run(
            ["git", "diff", f"origin/{target_branch}...HEAD", "--name-only", "--diff-filter=d"],
            capture_output=True,
            text=True
        )
        return [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]

    def extract_public_api(self, files):
        # Use tree-sitter to parse Python files
        # Only return functions/classes with public names (no leading underscore)
        # Skip test files and internal modules
        ...
```

### Resolving conflicts

The agent would often encounter merge conflicts when multiple developers updated the same file. Our solution was to skip files with conflicts and leave them for manual review. We added this to the agent’s instructions:

> "If you encounter a merge conflict in a file, skip that file entirely. Add it to a skip list and continue processing other files. Do NOT attempt to resolve the conflict—it requires human judgment."

We also configured the agent to handle Git operations safely:

```python
class DocumentationUpdater:
    def apply_updates(self, updates):
        applied = []
        skipped = []
        
        for update in updates:
            try:
                # Check if file has merge conflicts
                if "<<<<<<<" in open(update['file']).read():
                    skipped.append(update)
                    continue
                    
                # Apply change
                with open(update['file'], 'r') as f:
                    content = f.read()
                
                # Use difflib to apply specific changes
                new_content = self._apply_diff(content, update['diff'])
                
                with open(update['file'], 'w') as f:
                    f.write(new_content)
                
                applied.append(update)
                
            except Exception as e:
                skipped.append(update)
                self.logger.warning(f"Failed to update {update['file']}: {e}")
        
        return applied, skipped
```

### CI/CD integration

We run the agent nightly at 2 AM UTC in a GitHub Actions workflow:

```yaml
# .github/workflows/documentation-agent.yml
name: Documentation Agent

on:
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM UTC
  workflow_dispatch:     # Manual trigger

jobs:
  run-agent:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for git diff to work properly
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install claude-code-agent==0.2.1 tree-sitter==0.20.10
      
      - name: Run documentation agent
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}
        run: |
          python agent_workflow.py \
            --repository . \
            --target-branch main \
            --skip-files .claude/skip-list.txt
      
      - name: Create PR if changes
        if: steps.agent.outputs.updates_applied > 0
        uses: peter-evans/create-pull-request@v6
        with:
          title: "docs: auto-update documentation from code changes"
          body: "This PR was automatically generated by the documentation agent.
          Please review the changes carefully."
          branch: "auto/docs-update-${{ github.run_id }}"
          commit-message: "docs: update documentation for ${{ steps.agent.outputs.updates_applied }} files"
          labels: "automated, documentation"
```

The workflow runs in about 8–12 minutes total, including:
- 3 minutes to fetch and analyze changes
- 5 minutes for the agent to process files (most of the time)
- 2 minutes to create PRs and handle Git operations

### Cost tracking

We were concerned about API costs. The agent uses Claude’s API at approximately $0.008 per 1000 tokens (Claude Sonnet 4 pricing as of May 2025). Each documentation update averages:
- 200 tokens for analysis
- 500 tokens for generation
- 300 tokens for validation

For our codebase, that’s about **$0.0072 per run** when processing 50 files. At 30 runs per month, the cost is **$0.22/month**—negligible compared to the time saved.

## Results — the numbers before and after

The results exceeded our expectations:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Weekly documentation time | 6–8 hours | 15 minutes | 96% reduction |
| Documentation accuracy | 60% (manual errors) | 98% (agent + human review) | 38% improvement |
| PR creation time | Manual (30–60 min) | Automated (5 min) | 92% faster |
| Merge conflicts | 12% of updates | 3% of updates | 75% reduction |
| Developer satisfaction (survey) | 2.1/5 | 4.7/5 | 2.6 point jump |

Our first month with the agent reduced documentation maintenance from 20 hours/week to 6 hours/week—a **70% reduction in total maintenance time**. The remaining 6 hours were spent reviewing the agent’s PRs and handling edge cases it couldn’t resolve.

The accuracy improvement was the biggest surprise. Before, we had a "documentation debt" of about 40%—functions with incorrect or missing documentation. After deployment, this dropped to 2%. The agent consistently followed our documentation standards, something even experienced developers struggled with.

One unexpected benefit was **discovering code smells**. The agent’s analysis often highlighted:
- Functions with missing type hints
- Undocumented public APIs
- Inconsistent docstring formats
- Functions that should be internal but were public

This led us to refactor 12 functions into internal modules and add type hints to 45 functions—improvements we’d been putting off for months.

The only area where we didn’t see improvement was **complex type hints**. The agent struggled with:
- Nested Pydantic models
- Custom validators
- Union types with complex discriminators

For these cases, we added a manual review step: the agent creates a PR, but developers must approve complex type changes before merging.

## What we'd do differently

Looking back, several aspects of our implementation could have been better:

### 1. Starting too broad

We initially tried to make the agent handle **all** documentation updates: docstrings, OpenAPI specs, wiki pages, release notes. This was a mistake. The agent performed best when we narrowed its scope to **Python docstrings only**. Once that worked reliably, we expanded to OpenAPI specs (which use a similar structure) but stopped there.

**Lesson:** Agents work best with narrow, well-defined tasks. The broader the scope, the more edge cases you’ll encounter.

### 2. Not tracking agent decisions

Early versions of the agent sometimes made questionable changes. Without a record of what it did and why, debugging was painful. We added a decision log:

```python
class AgentLogger:
    def log_decision(self, file, item, decision, reason):
        with open(".claude/agent-decisions.log", "a") as f:
            f.write(f"{datetime.now()} | {file} | {item} | {decision} | {reason}\n")
```

This allowed us to:
- Review why the agent skipped certain files
- Understand its reasoning for complex changes
- Identify patterns in its mistakes

**Lesson:** Log everything the agent does with timestamps and explanations.

### 3. Underestimating Git complexity

Git operations turned out to be the most fragile part of our system. The agent would sometimes:
- Create invalid commits (missing author info)
- Fail to push changes due to authentication issues
- Corrupt the Git index when multiple processes tried to update

We solved this by:
- Using `git reset --hard` before each run to ensure a clean state
- Running the agent in a dedicated GitHub Actions job with proper permissions
- Adding retry logic for push operations

**Lesson:** Treat Git operations as a critical path—don’t assume they’ll work.

### 4. Not measuring agent performance

We deployed the agent without proper monitoring. After two weeks, we realized:
- The agent was skipping 15% of files due to errors
- Some updates took 2 minutes instead of 30 seconds
- The agent was creating duplicate PRs occasionally

We added Prometheus metrics:

```python
from prometheus_client import Counter, Gauge

DOCS_UPDATES_APPLIED = Counter('docs_updates_applied_total', 'Total documentation updates applied')
DOCS_FILES_SKIPPED = Counter('docs_files_skipped_total', 'Total files skipped due to errors')
AGENT_PROCESSING_TIME = Gauge('agent_processing_seconds', 'Time spent processing updates')
```

This revealed that the agent was taking longer on large files with complex decorators. We optimized our parser to handle these cases better.

**Lesson:** Monitor agent performance from day one—don’t wait for outages.

### 5. Ignoring developer onboarding

Our initial assumption was that developers would trust the agent’s PRs automatically. In practice, many developers hesitated to merge AI-generated changes. We added a **human review checklist** to each PR:

```markdown
## Agent-generated documentation update

**Changes made:**
- Updated docstring for `create_user` in `services/auth.py`
- Added parameter descriptions for `email` and `password`
- Updated OpenAPI spec for `/auth/register` endpoint

**Verification checklist:**
- [ ] Does the docstring match the function behavior?
- [ ] Are all parameters documented?
- [ ] Does the OpenAPI spec reflect the actual API?
- [ ] Are there any security implications?

**If unsure:** Reject the PR and open an issue for manual review.
```

This increased trust significantly. Developers still review the changes, but the checklist makes the process more structured.

## The broader lesson

The key insight from this experiment is that **agents work best as specialized problem-solvers, not general-purpose assistants**. Our agent succeeded because it had a narrow, well-defined task with clear success criteria. It didn’t need to understand the entire codebase—just the patterns we taught it.

This principle applies beyond documentation:
- **Testing agents** should focus on specific test cases, not entire test suites
- **Code review agents** should specialize in security issues or performance patterns, not general reviews
- **Deployment agents** should handle environment-specific configurations, not entire deployment pipelines

The second lesson is that **edge cases are the killer of agent workflows**. Every time we tried to make the agent "smarter" by handling more cases, it broke on something unexpected. The most reliable systems were those that failed fast and left the complex cases for humans.

Finally, **measure everything**. Agents introduce new failure modes (API limits, rate limits, unexpected outputs) that can silently degrade over time. Without proper monitoring, you won’t know when the system is no longer working.

## How to apply this to your situation

If you’re considering building your first agentic workflow, start with these steps:

1. **Identify a repetitive, well-defined task**
   Look for tasks that:
   - Take 2+ hours per week
   - Have clear inputs and outputs
   - Follow a pattern you can describe in a few sentences
   - Don’t require subjective judgment

   Good candidates: 
   - Updating changelogs from commit messages
   - Generating API client code from OpenAPI specs
   - Syncing database schemas across environments
   - Validating PR descriptions against templates

2. **Write the instructions as if you’re training a junior developer**
   The most successful agents we’ve seen have instructions that read like:
   ```markdown
   You are a documentation specialist. Your job is to update docstrings
   when function signatures change. Follow these rules:
   
   1. Only update functions with @api decorator
   2. Preserve existing formatting in docstrings
   3. Skip functions with @internal decorator
   4. If unsure, skip the function and log the decision
   ```

3. **Start with a 10% scope reduction**
   Take the task you want to automate and narrow it by 90%. For example:
   - Instead of "update documentation", start with "update FastAPI endpoint docstrings"
   - Instead of "review PRs", start with "check for missing type hints"
   - Instead of "deploy to production", start with "validate environment variables"

4. **Measure before you automate**
   Run the manual process for a week and record:
   - Time spent per task
   - Error rate
   - Frequency of interruptions
   - Developer satisfaction

   This gives you a baseline to compare against.

5. **Build the agent incrementally**
   Week 1: Simple script that logs what it would do
   Week 2: Script that creates a PR with changes (no applying)
   Week 3: Script that applies changes to a test branch
   Week 4: Full production deployment

   Each step gives you confidence before moving forward.

6. **Plan for rollback**
   Even the best agents will have off days. Have a manual override:
   - A flag to disable the agent
   - A way to revert its changes
   - A process for emergency fixes

   Our rollback plan was simple: any developer could close the agent’s PR and manually update documentation. We never needed it, but the knowledge it existed gave us confidence to deploy.


## Resources that helped

Here are the tools, articles, and templates that made our agent successful:

### Tools
- [Claude Code v0.12.0](https://github.com/anthropics/claude-code) – The agent framework we used
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) – For precise code parsing
- [GitHub Actions](https://github.com/features/actions) – For CI/CD integration
- [Prometheus](https://prometheus.io/) – For monitoring agent performance
- [claude-code-agent](https://github.com/your-org/claude-code-agent) – Our wrapper library (internal)

### Articles and guides
- [Anthropic’s agent mode documentation](https://docs.anthropic.com/claude/docs/agents) – The official guide we followed
- ["When to use agents vs tools" by Simon Willison](https://simonwillison.net/2025/Apr/15/agents-vs-tools/) – Helped us understand agent boundaries
- ["The Rise of the AI Engineer" by Eugene Yan](https://eugeneyan.com/writing/ai-engineer/) – Context on agent workflows
- ["How to write good prompts for agents" by Lilian Weng](https://lilianweng.github.io/posts/2025-04-05-agent-prompts/) – Improved our instruction quality

### Templates
- [claude-agent-template](https://github.com/anthropics/claude-agent-template) – Starter template for Claude agents
- [documentation-agent-instructions.md](https://gist.github.com/your-org/documentation-agent) – Our instruction file (adapt for your use)
- [ci-workflow.yml](https://github.com/your-org/.github/blob/main/workflows/agent.yml) – Our GitHub Actions config

### Learning from others
- We studied [Linear’s AI workflows](https://linear.app/blog/ai-at-linear) for inspiration on agent design
- [Shopify’s documentation automation](https://shopify.engineering/automating-documentation) gave us ideas for handling conflicts
- [Netlify’s approach to AI agents](https://www.netlify.com/blog/ai-agents-at-netlify/) helped with cost tracking


## Frequently Asked Questions

### How long did it take to build the first version of the agent?

The first working prototype took **5 days** of part-time work. We spent:
- Day 1: Setting up Claude Code and testing basic agent commands
- Day 2: Writing the instruction file and testing on a single file
- Day 3: Expanding to handle multiple files and edge cases
- Day 4: Integrating with GitHub and testing PR creation
- Day 5: Running in CI and fixing initial bugs

The agent wasn’t production-ready after 5 days, but it could handle simple cases. We spent another **2 weeks** refining it before deploying to our main repository.

### What’s the biggest mistake teams make when building agents?

Most teams **underestimate the importance of scope**. They try to build an agent that can handle every possible edge case, which leads to brittle systems that fail in unpredictable ways. The most successful agents we’ve seen have a single, narrow responsibility—like updating docstrings or validating PRs—with clear boundaries on what they can and cannot do. Another common mistake is **not logging agent decisions**—when something goes wrong, you need to know exactly what the agent did and why.

### Can this work for non-Python codebases?

Yes, but you’ll need to adapt the approach. The key is understanding your language’s AST and documentation patterns. For JavaScript/TypeScript, you could use [Babel](https://babeljs.io/) for parsing. For Java, [JavaParser](https://javaparser.org/) works well. For Go, the [go/ast](https://pkg.go.dev/go/ast) package is built-in. The agent’s instruction file should focus on your language’s specific documentation conventions (JSDoc, JavaDoc, Go doc comments, etc.). We’ve seen teams successfully adapt this pattern to Ruby, C#, and Rust codebases.

### How do you handle agent mistakes?

We use a **two-layer approach**:
1. **Pre-merge validation** – The agent creates a PR, but developers must approve changes before merging
2. **Post-merge monitoring** – We track documentation accuracy over time using a simple script that compares docstrings with actual function behavior

When the agent makes a mistake, we:
1. Update the instruction file to prevent recurrence
2. Add the specific case to our test suite
3. Manually fix the documentation
4. Review why the agent failed (usually a missing edge case in instructions)

So far, about 8% of the agent’s PRs have required manual fixes, but this rate is decreasing as we refine the instructions.

### What’s the maintenance burden of an agent like this?

The maintenance burden is **low but non-zero**. After the initial setup (about 20 hours), we spend:
- **1–2 hours/month** updating the instruction file for new code patterns
- **30 minutes/week** reviewing agent PRs
- **5 minutes/day** monitoring the agent dashboard

The biggest maintenance tasks are:
- Handling new decorators or patterns in the codebase
- Adjusting for changes in documentation standards
- Updating the agent when new versions of Claude Code change behavior

Compared to the 20 hours/week we were spending before, this is a massive improvement. The agent essentially pays for itself in the first month and continues saving time thereafter.

### Is there a point where the agent becomes more trouble than it’s worth?

Based on our experience, the agent stops being worth it when:
- The task it’s automating takes **less than 30 minutes per week**
- The codebase changes **more than 50% of its structure monthly**
- The agent’s error rate exceeds **10%** (after accounting for false positives)
- Developer trust in the agent drops **below 70%**

In our case, we hit the first threshold naturally as we improved documentation standards—the task became smaller over time. The agent was still useful for handling edge cases, but we reduced its frequency from daily to weekly runs.

### How do you prevent the agent from creating too many PRs?

We use **batch processing** and **smart filtering**:
1. The agent only runs **once per day** (at 2 AM UTC)
2. It processes **all changes since the last run**, not just recent ones
3. We filter files by **change type**—only files with signature changes get processed
4. We skip files with **recent activity** (modified in the last 24 hours) to avoid conflicts

This results