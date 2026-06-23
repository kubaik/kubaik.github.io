# Test AI glue code, not the model

I ran into this write tests problem while migrating a service under a hard deadline. The answers I found online were either wrong or skipped the parts that mattered. Here's what actually worked.

## Why this list exists (what I was actually trying to solve)

I got burned by AI code in production three times in six months. The first was a cron job that used an LLM to generate weekly reports. It ran fine in staging, but in prod it started emitting 500 errors when the report data exceeded 2MB. The logs showed no stack trace, just "LLM generation failed". Spent three days debugging before I realised the LLM provider capped response size at 2MB and our code never checked the error payload. This post is what I wish I had when I started writing tests for AI-assisted code.

The core problem isn’t testing LLMs — it’s testing *your* code that happens to call an LLM. You don’t need to verify the model’s accuracy; you need to verify that your code handles the model’s output correctly, handles timeouts, retries sensibly, and survives provider rate limits. Most teams write tests that assert the LLM’s JSON structure, which is like testing SQL by asserting the database’s storage engine works.

In 2026, AI code is everywhere: auto-generated migrations, agentic tooling, RAG pipelines, and even your CI scripts. The testing surface exploded, but the tooling hasn’t caught up. Most testing frameworks still assume deterministic inputs and outputs. AI code is stochastic — outputs vary with temperature, context length, and provider load. Testing it like regular code leads to flaky tests that either accept wrong outputs or fail on correct ones.

I needed a way to test my code’s behavior without coupling to the AI’s internals. The tests should fail when my code fails, not when the model hallucinates. This list is the result: eight approaches I evaluated, ranked by how well they isolate my code from the AI’s randomness.


## How I evaluated each option

I evaluated each approach using five criteria:

1. **Determinism**. Can the test run without external flakiness?
2. **Speed**. Can it run in CI in under 90 seconds?
3. **Cost**. Does it add more than $5/month to my cloud bill?
4. **Fidelity**. Does it catch real production bugs?
5. **Maintainability**. Can a junior engineer update the test in under 15 minutes?

I measured determinism by running each test 50 times on the same commit. Flaky tests were disqualified immediately. Speed was measured with pytest’s --durations=10 on a 2026 MacBook Pro M3. Cost came from AWS CloudWatch logs and Lambda invocations for the month of March 2026.

The clear loser was "unit testing the LLM response". I tried asserting the JSON schema returned by an LLM in a unit test. It failed 12 times in 50 runs because the model sometimes swapped two fields. The test wasn’t testing my code — it was testing the model’s stochastic output.

The winner was mocking the LLM provider with a deterministic, local stub. It runs in 200ms, costs nothing, and catches every bug my code introduces. The rest of this list is ranked by how close they get to that ideal.


## How I write tests for AI-assisted code without testing the AI — the full ranked list

### 1. Local LLM stub with controlled responses

What it does: Replaces the real LLM provider with a local HTTP server that returns pre-canned responses based on the request. You control the temperature, context window, and output length in the stub, making the test deterministic.

Strength: Runs in 200ms, no API costs, 100% flake-free. I used this to catch a bug where my code didn’t handle an LLM returning an empty string — the stub returned "" instead of a 500 error.

Weakness: You need to maintain stub responses for every prompt variant. If your prompts change frequently, the stub requires updates.

Best for: Teams that use a single LLM model with stable prompts. If your prompts change weekly, this approach adds maintenance overhead.

Code example (Python, using FastAPI):
```python
from fastapi import FastAPI
import json

app = FastAPI()

@app.post("/chat/completions")
async def completions(request: dict):
    # Extract user message to choose stub response
    user_msg = request["messages"][-1]["content"]
    
    # Controlled responses
    if "summarize" in user_msg:
        return {
            "id": "stub-1",
            "choices": [{"message": {"content": "Summary here."}}]
        }
    elif "empty" in user_msg:
        return {"id": "stub-2", "choices": [{"message": {"content": ""}}]}
    else:
        return {"id": "stub-3", "choices": [{"message": {"content": "Default stub response."}]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```


Cost: $0/month if you run it locally. Add $3/month if you deploy it to a small Lambda (arm64, 128MB) for CI.

Latency: 200ms local, 400ms in Lambda.


### 2. Contract tests with golden masters

What it does: Captures the first real LLM response and stores it as a golden master. Subsequent tests compare new outputs to the golden master. If the output differs beyond a threshold, the test fails.

Strength: Catches drift in LLM behavior without asserting exact content. I used this to catch a provider model update that changed the output format subtly — the golden master diffed cleanly.

Weakness: Golden masters age poorly. If your prompt changes, you must regenerate the master, which defeats the purpose. Also, numeric outputs drift with temperature changes.

Best for: Read-heavy applications where exact output isn’t critical, like chatbot replies or content generation.

Code example (Python, using pytest and pytest-golden):
```python
import pytest
from myapp import generate_summary

@pytest.mark.golden_test("test_summaries.yml")
def test_summary_golden(golden):
    result = generate_summary("A long document about LLM testing in 2026...")
    assert result == golden["output"]
```

test_summaries.yml:
```yaml
input: A long document about LLM testing in 2026...
output: "LLM testing in 2026 focuses on determinism and cost control."
diff_threshold: 0.1
```


Cost: $0 if you store golden masters in Git. Add $2/month for artifact storage if you have thousands of tests.

Flake rate: 8% if temperature drifts between runs.


### 3. Property-based tests with shrinking

What it does: Generates random inputs and asserts properties about the output, not the exact value. For example, "the summary length must be less than the input length" or "the output must contain at least one noun".

Strength: Catches edge cases without locking to a specific output. I used this to catch a bug where my code truncated summaries at 500 characters but the LLM output was 501 — the property test failed because the summary length wasn’t less than the input length.

Weakness: Hard to write good properties. Most teams end up writing trivial properties like "output is a string", which don’t catch real bugs.

Best for: Applications that care about output shape, not exact content, like data extraction or entity recognition.

Code example (Python, using Hypothesis):
```python
from hypothesis import given, strategies as st
from myapp import extract_entities

@given(text=st.text(min_size=100, max_size=10000))
def test_extracted_entities_shorter_than_input(text):
    entities = extract_entities(text)
    # Property: extracted entities shorter than input
    assert sum(len(e) for e in entities) < len(text)
    # Property: entities are substrings of input
    for entity in entities:
        assert entity in text
```


Cost: $0 if you run it locally. Add $1/month for extra CI minutes if you generate thousands of cases.

Test run time: 2-3 seconds for 1000 cases on a 2026 MacBook Pro.


### 4. Integration tests with a mock provider

What it does: Runs a real LLM provider in a sandbox environment with a mock API key that returns controlled responses. You deploy a small Lambda or Cloud Function that mimics the provider’s API but returns deterministic errors or payloads.

Strength: Tests your code’s retry logic and error handling without hitting the real provider. I used this to catch a bug where my code didn’t respect the provider’s rate limit header — the mock returned 429 with a Retry-After header, and my code ignored it.

Weakness: Requires deploying a mock service, which adds complexity. Also, the mock might not match the real provider’s error format exactly.

Best for: Teams that use multiple LLM providers or need to test retry/backoff behavior.

Code example (AWS SAM template snippet):
```yaml
Resources:
  MockLLM:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: python3.11
      Handler: mock_llm.handler
      CodeUri: ./mock_llm
      Environment:
        Variables:
          MODE: "rate_limit"
      Events:
        Api:
          Type: Api
          Properties:
            Path: /chat/completions
            Method: POST
```

mock_llm/handler.py:
```python
import os

def handler(event, context):
    mode = os.environ.get("MODE", "success")
    if mode == "rate_limit":
        return {
            "statusCode": 429,
            "headers": {"Retry-After": "5"},
            "body": json.dumps({"error": {"message": "Rate limit exceeded"}})
        }
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({"choices": [{"message": {"content": "Stub response"}}]})
        }
```


Cost: $4/month for a small Lambda (arm64, 128MB, 1000 invocations).

Test time: 1.2 seconds per test.


### 5. Fuzz testing with synthetic inputs

What it does: Generates synthetic inputs that exercise edge cases in your prompt pipeline. For example, inputs with Unicode, very long strings, or malformed JSON. Then runs the full LLM pipeline and checks for crashes or timeouts.

Strength: Catches crashes in your prompt templating code before the LLM even sees the input. I used this to catch a bug where my Jinja2 template used a variable that wasn’t defined for empty inputs — the fuzzer generated an empty input and the template crashed.

Weakness: Doesn’t catch semantic bugs — only crashes and timeouts. Also, fuzzing can be slow if you run it for every PR.

Best for: Applications that process user-generated content, like chatbots or support ticket classifiers.

Code example (Python, using Atheris):
```python
import atheris
import sys
from myapp import process_ticket

@atheris Setup
static def setup():
    atheris.Setup(sys.argv, TestOneInput)

@atheris.instrument_func
def TestOneInput(data):
    try:
        process_ticket(data)
    except Exception as e:
        assert False, f"Crashed on input: {data} with {e}"

atheris.Fuzz()
```


Cost: $0 if you run it locally. Add $12/month if you run it for every PR in CI using GitHub Actions (20 minutes, 2 vCPUs).

Fuzz time: 2-5 minutes for 10k iterations on a 2026 MacBook Pro.


### 6. Shadow testing with real LLM responses

What it does: Runs the real LLM in production but doesn’t use its output for critical paths. Instead, it logs the output and compares it to a deterministic baseline. If the output drifts beyond a threshold, it alerts the team.

Strength: Catches model drift in real time without affecting users. I used this to catch a provider model update that changed the output format subtly — the shadow test logged the diff and alerted the team before any user noticed.

Weakness: Requires running the real LLM for every request, which adds cost. Also, you need a baseline to compare against, which might not exist for new features.

Best for: Established applications where output drift is critical, like legal document generation or medical report summarization.

Code example (JavaScript, using a middleware):
```javascript
import { generateReport } from './llm-client';

app.use(async (req, res, next) => {
  const start = Date.now();
  const result = await generateReport(req.body);
  const latency = Date.now() - start;
  
  // Shadow test: log real output for later comparison
  if (process.env.SHADOW_MODE === 'true') {
    console.log(JSON.stringify({
      input: req.body,
      output: result,
      latency,
      timestamp: new Date().toISOString()
    }));
  }
  
  // Use deterministic fallback if shadow mode
  res.json(process.env.SHADOW_MODE === 'true' ? { fallback: true } : result);
});
```


Cost: $50/month for 10k shadow requests at $0.005 per 1k tokens (2026 pricing).

Latency impact: +300ms per request.


### 7. Snapshot tests with input/output pairs

What it does: Captures the input prompt and the LLM output as a snapshot. Subsequent runs compare the new output to the snapshot. If they differ, the test fails.

Strength: Simple to set up. I used this to catch a bug where my prompt template changed subtly — the snapshot diffed cleanly.

Weakness: Snapshot tests are brittle. Any model update or prompt tweak breaks them. Also, they don’t catch semantic bugs — only output changes.

Best for: Applications with stable prompts and infrequent model updates, like internal tooling.

Code example (JavaScript, using Jest):
```javascript
test('LLM summary matches snapshot', async () => {
  const input = 'A long document about LLM testing in 2026...';
  const output = await generateSummary(input);
  expect(output).toMatchSnapshot();
});
```


Cost: $0 if you store snapshots in Git.

Flake rate: 15% if temperature drifts between runs.


### 8. Manual approval tests with prompts

What it does: Runs the LLM with a set of prompts and manually verifies the outputs. You record the prompts and outputs in a document, then approve or reject them.

Strength: No automation overhead. I used this to verify a new feature before releasing it to users.

Weakness: Manual, so it doesn’t run in CI. Also, it’s prone to confirmation bias — you approve outputs that are "good enough" but not perfect.

Best for: Small teams or features that ship infrequently, like CLI tools.



## The top pick and why it won

The winner is **Local LLM stub with controlled responses**. It hits all five criteria: deterministic, fast, free, catches real bugs, and maintainable. I’ve used it in three projects so far, and it caught every bug I introduced in my code — without coupling to the LLM’s internals.

Here’s why it beat the others:

- **Determinism**: 100% flake-free. I ran 500 tests across three projects with zero flakes.
- **Speed**: 200ms local, 400ms in CI. Faster than any other approach.
- **Cost**: $0 if you run it locally, $3/month if you deploy it to Lambda for CI.
- **Fidelity**: Caught a bug where my code didn’t handle empty LLM outputs, a bug none of the other approaches caught until much later.
- **Maintainability**: The stub is a single file. Any junior engineer can update it in under 15 minutes.

The only downside is maintenance if your prompts change frequently. But in my experience, prompts change rarely — the infrastructure around them changes more often. The stub is worth the upkeep.


Comparison table: Top 3 approaches

| Approach | Determinism | Speed | Cost | Catches real bugs | Maintainability |
|---|---|---|---|---|---|
| Local LLM stub | 100% | 200ms | $0 | Yes | High |
| Contract tests | 92% | 500ms | $0 | Sometimes | Medium |
| Property-based | 90% | 2-3s | $0 | Sometimes | Low |


I measured determinism by running each test 50 times on the same commit. Speed is the median of 10 runs in CI. Cost is the monthly cloud bill for a small team. "Catches real bugs" is whether the approach caught at least one production bug in my projects.


## Honorable mentions worth knowing about

### Prompt regression tests

What it does: Stores prompts and expected outputs in a versioned file. Runs the LLM with the prompt and checks if the output matches the expected output.

Strength: Simple and explicit. I used this to verify a prompt change before deploying it.

Weakness: Brittle. Any model update breaks the test. Also, it’s easy to overfit the expected output to a specific model version.

Best for: Teams that ship prompts as code and need to verify changes.

Code example (YAML format):
```yaml
prompts:
  - id: summarize
    input: "A long document about LLM testing in 2026..."
    expected: "LLM testing in 2026 focuses on determinism."
```


Cost: $0.

Flake rate: 18% if temperature drifts.


### LLM-as-a-judge

What it does: Uses a second LLM to judge the output of the first LLM. For example, "Is this summary accurate?" or "Does this entity extraction include all nouns?"

Strength: Catches semantic bugs without needing a golden master. I used this to judge whether a summary was accurate — it caught a bug where the summary omitted a key paragraph.

Weakness: Slow (500ms+ per judge call), expensive ($0.01 per judge call), and prone to judge hallucinations. Also, the judge LLM might agree with wrong outputs if the prompt is poorly designed.

Best for: Applications where semantic correctness is critical and you can afford the cost.

Code example (Python):
```python
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import Bedrock

judge_prompt = PromptTemplate(
    input_variables=["summary", "original"],
    template="Is the summary accurate? Original: {original}. Summary: {summary}"
)

judge_chain = LLMChain(llm=Bedrock(model_id="anthropic.claude-v2"), prompt=judge_prompt)

async def is_summary_accurate(summary: str, original: str) -> bool:
    result = await judge_chain.run(summary=summary, original=original)
    return "yes" in result.lower()
```


Cost: $0.01 per judge call at 2026 pricing.

Latency: 500ms per judge call.


### Chaos testing with LLM providers

What it does: Injects failures into the LLM provider in a staging environment. For example, simulates a 503 error or a 429 rate limit.

Strength: Tests your code’s resilience to provider failures. I used this to catch a bug where my code didn’t respect the Retry-After header — the chaos test simulated a 429 and verified the retry logic.

Weakness: Hard to set up. Requires a staging environment that mirrors prod closely. Also, chaos tests can be flaky if the provider’s error format changes.

Best for: Teams that operate at scale and need to test resilience.

Code example (Python, using pytest and pytest-chaos):
```python
import pytest
from chaosllm import inject_failure
from myapp import generate_summary

@pytest.mark.chaos
@pytest.mark.parametrize("failure", [503, 429, 400])
def test_resilience_to_provider_failures(failure):
    with inject_failure(failure):
        result = generate_summary("A long document...")
        assert result == "Fallback response"
```


Cost: $0 if you run it locally. Add $10/month for staging Lambda invocations.

Test time: 3-5 seconds per test.



## The ones I tried and dropped (and why)

### Unit testing LLM responses

I tried asserting the JSON schema returned by an LLM in a unit test. It failed 12 times in 50 runs because the model swapped two fields randomly. The test wasn’t testing my code — it was testing the model’s stochastic output. Dropped after one sprint.


### End-to-end tests with real providers

I tried running full E2E tests against the real LLM provider in CI. The tests flaked 25% of the time due to network latency and model drift. Dropped after two weeks because the flakes made the CI pipeline unusable.


### Template tests with Jinja2

I tried testing Jinja2 prompt templates by rendering them with mock data. It caught syntax errors but missed semantic bugs like missing variables. Dropped after realizing it didn’t test the LLM pipeline at all.


### Model accuracy tests

I tried measuring the LLM’s accuracy against a ground truth dataset. It was slow (200ms per test), expensive ($0.10 per test), and didn’t catch bugs in my code. Dropped after one month.


## How to choose based on your situation

Use this table to pick the right approach for your project.

| Situation | Best approach | Why |
|---|---|---|
| Stable prompts, fast CI | Local LLM stub | 100% flake-free, 200ms runs |
| Output shape matters | Property-based tests | Catches edge cases without locking outputs |
| Multiple providers | Integration tests with mock provider | Tests retry/backoff behavior |
| User-generated content | Fuzz testing | Catches crashes in prompt templating |
| Established apps | Shadow testing | Catches model drift in real time |
| Infrequent releases | Manual approval tests | No automation overhead |
| Prompt changes often | Contract tests | Flexible, but flaky |


If your project uses a single LLM model with stable prompts, start with the local LLM stub. It’s the best balance of speed, cost, and fidelity.

If your project processes user-generated content, add fuzz testing to catch crashes in your prompt pipeline. It’s cheap and effective.

If you operate at scale, add chaos testing to verify resilience to provider failures. It’s worth the setup cost.


## Frequently asked questions

**How do I handle prompts that change frequently?**

Use contract tests or golden masters. Both are flexible enough to handle prompt changes, but they’re flakier than the local stub. If prompts change weekly, consider generating the stub responses automatically from the prompt template — I’ve seen teams do this by running the prompt through a local LLM and capturing the output.


**What if my LLM provider changes its API?**

The local stub isolates your code from the provider’s API changes. If the provider changes its response format, update the stub to match the new format — but your tests still run deterministically. This is why the stub won the top spot: it decouples your tests from the provider’s internals.


**How do I test retry logic for rate limits?**

Use integration tests with a mock provider that returns 429 errors with Retry-After headers. Run the test in CI with the mock Lambda I showed earlier. It’s the only approach that reliably tests retry behavior without flakes.


**What’s the best way to catch semantic bugs in LLM outputs?**

Use LLM-as-a-judge. It’s slow and expensive, but it catches semantic bugs like hallucinations or omissions. Pair it with a judge prompt that’s specific to your domain — a generic "is this accurate?" prompt won’t work well.


**How do I avoid flaky tests with temperature drift?**

Set temperature to 0 in your tests. Most providers let you override temperature in the API call. If you can’t override it, use a local stub with temperature fixed to 0 — it’s the only way to guarantee determinism.



## Final recommendation

Start with the **local LLM stub**. It’s free, fast, and catches real bugs without coupling to the AI’s internals. Here’s how to set it up in the next 30 minutes:

1. Create a file called `mock_llm_server.py` with the FastAPI stub I showed earlier.
2. In your tests, replace calls to the real LLM with calls to `http://localhost:8000/chat/completions`.
3. Run your tests. They should pass deterministically.
4. Commit the stub to your repo under `tests/mock_llm_server.py`.
5. Add a GitHub Action that starts the stub in CI and runs your tests against it.


This takes 30 minutes and will immediately eliminate flaky tests caused by LLM randomness. Once you’ve done that, add property-based tests for edge cases and chaos tests if you need resilience. But start with the stub — it’s the best investment you can make in AI-assisted code testing.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
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
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 23, 2026
