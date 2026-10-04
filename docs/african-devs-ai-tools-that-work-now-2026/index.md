# African devs: AI tools that work now (2026)

AI coding assistants are usually documented as if every user has fibre, a recent laptop and an unmetered cloud budget. That assumption breaks down for a large share of developers worldwide, and the failure is rarely announced — the tool simply becomes slow, expensive or unreliable, and the developer concludes that "AI doesn't work here" rather than "this tool's assumptions don't match my environment."

This article treats that mismatch as an engineering problem. It covers the three failure modes that dominate on constrained networks, how to diagnose each one, how to measure whether a fix actually worked, and how to choose tools that fit rather than fight your setup.

## The three constraints that actually matter

Most tool-selection advice focuses on model quality. In practice, model quality is rarely the binding constraint. Three environmental factors decide whether an assistant is usable:

**Bandwidth and latency.** Cloud assistants send request payloads — file context, cursor position, surrounding code — on every completion. Response payloads are smaller but frequent. On a metered mobile connection, the cost is both financial and in round-trip latency, which determines whether a suggestion arrives before you have already typed the line yourself.

**Memory and compute.** Local models compete for RAM with your editor, browser, container runtime and language server. A model that technically "runs" on 4GB may still push the machine into swap, at which point every editor action stutters.

**Network policy.** Corporate firewalls, captive portals, VPNs and ISP-level filtering can block or throttle API endpoints. A tool that works on a home connection may fail entirely on an office or campus network, often with an unhelpful error.

Power reliability is a fourth factor, but it interacts with the first two: a machine on battery has less headroom for background inference, and an unexpected shutdown mid-download wastes whatever data was already spent.

## Failure mode 1: background traffic you did not ask for

**Symptom.** The editor becomes sluggish after the first suggestion. A data monitor shows sustained traffic even when you are reading, not typing. On a metered connection, the daily allowance disappears within an hour or two of ordinary work.

**Cause.** Autocomplete, "explain this" hover actions and background indexing features are frequently enabled by default. Many of them transmit on every keystroke pause, not only when you explicitly invoke the assistant.

**Diagnosis.** Identify which process is consuming bandwidth before changing any settings:

```bash
# Per-process bandwidth, sorted by usage
sudo apt update && sudo apt install -y nethogs
sudo nethogs
```

On macOS, `nettop -P -l 1` gives a comparable per-process view. Run the monitor for five minutes of normal typing and note which process dominates. If the assistant's language-server process is responsible, configuration is the lever.

**Fix.** Disable the features you are not actively using. In VS Code, the relevant settings key for GitHub Copilot is:

```json
// settings.json
{
  "github.copilot.enable": {
    "*": false,
    "editor": false,
    "terminal": false,
    "markdown": false
  }
}
```

Note that disabling Copilot's inline suggestions does not necessarily stop other extensions you have installed. Check the extension list as well as the settings.

In JetBrains IDEs, AI Assistant features are toggled under Settings > Languages & Frameworks > AI Assistant. Turning them off removes the background requests but also removes the feature; there is no partial "only when I ask" mode in every version, so verify the behaviour on your build.

**Trade-off.** Disabling autocomplete removes the feature that many developers find most valuable. A more targeted approach is to keep suggestions in files where they help (application code) and disable them where they generate noise (configuration, lockfiles, generated code).

## Failure mode 2: local models that exceed the machine

**Symptom.** The assistant loads, then the whole system becomes unresponsive. `dmesg` or the system monitor shows the out-of-memory killer terminating processes. Alternatively, the model loads but produces a token every few seconds.

**Cause.** Model memory requirements scale with parameter count and quantisation. A rough planning figure for quantised weights is bytes ≈ parameters × bytes-per-parameter. At 4-bit quantisation that is roughly 0.5 bytes per parameter, so:

- A 3.8B-parameter model needs about 1.9GB for weights alone.
- A 7B-parameter model needs about 3.5GB for weights alone.

Add context (the KV cache grows with sequence length), the runtime's own overhead, and everything else the machine is doing. A 4GB machine running a 7B model at 4-bit quantisation has essentially no headroom, which is why it swaps.

**Diagnosis.** Measure actual resident memory rather than trusting the model card:

```bash
# Watch memory while the model loads and answers a prompt
watch -n 1 'free -m'
```

If available memory approaches zero and swap usage climbs, the model is too large for the machine as configured.

**Fix.** Choose a smaller model, or reduce quantisation further, or reduce context length. Ollama is a common local runtime:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:3.8b
ollama serve
```

Then point your editor extension at the local endpoint rather than a cloud provider. The exact configuration depends on the extension; most local-first assistants accept a base URL for an OpenAI-compatible API.

**Trade-off.** Smaller models are faster and lighter but weaker at multi-file reasoning and less reliable at following long instructions. For autocomplete and short refactors they are often adequate; for architectural questions they are not. Match the model to the task rather than expecting one model to cover everything.

## Failure mode 3: the network blocks the endpoint

**Symptom.** The tool works on one network and fails on another. Errors may be timeouts, TLS failures, HTTP 403, or a connection reset.

**Cause.** Firewalls and proxies may block or intercept the API host. TLS interception in particular produces certificate errors that look like a bug in the tool.

**Diagnosis.** Test the endpoint directly, outside the editor, so you can separate tool bugs from network policy:

```bash
# Substitute the host your tool actually calls
curl -v --max-time 10 https://api.example-ai-provider.com/v1/models
```

Read the output carefully. A DNS failure, a connection timeout, a TLS handshake failure and an HTTP 403 all point to different causes and different owners (your resolver, the network, the proxy, or the provider).

**Fix.** Options, in rough order of preference:

1. Use a local model for the work that must continue regardless of network state.
2. Ask the network administrator to allowlist the API host if policy permits.
3. Route through an approved proxy if one exists.

Avoid disabling TLS verification as a workaround. It converts a connectivity problem into a security problem, and on a network that is already intercepting traffic, it removes the only signal that interception is happening.

## Measuring whether a fix worked

Claims about "reduced data usage" or "faster suggestions" are only meaningful if you can reproduce the measurement. Instrument three things.

**Bandwidth per unit of work.** Use `nethogs` or `nettop` and record usage over a fixed task — for example, implementing one function with ten accepted completions. Compare before and after. Report the number alongside the task, because "MB per hour" depends entirely on how much you typed.

**Latency to first token.** Measure from the moment you trigger a completion to the moment text appears. A stopwatch is adequate for a rough figure; for anything more precise, timestamp requests in the client if the tool exposes a log. Compare like with like: same task, same network, same time of day.

**Suggestion acceptance and correctness rate.** This is the metric that actually predicts value, and it is the one vendors never publish for your codebase. A minimal harness:

```python
# evaluate_suggestions.py
# Illustrative harness: replace the client with your tool's API.
import statistics

def evaluate(client, prompts):
    accepted = 0
    compiled = 0
    latencies = []

    for prompt in prompts:
        suggestion = client.complete(prompt)
        latencies.append(suggestion.latency_ms)
        if suggestion.accepted:
            accepted += 1
        try:
            compile(suggestion.code, "<suggestion>", "exec")
            compiled += 1
        except SyntaxError:
            pass

    n = len(prompts)
    return {
        "acceptance_rate": accepted / n,
        "syntax_valid_rate": compiled / n,
        "median_latency_ms": statistics.median(latencies),
    }
```

Two cautions. First, `compile()` only checks syntax; it does not check that the code is correct or that the imports exist. For a stronger signal, run the suggested code against a test suite. Second, keep the environment fixed across runs — same interpreter version, same dependency set — or you will measure environment drift instead of model behaviour.

## A decision checklist for tool selection

Work through these in order. The first "no" usually determines the answer.

1. **Does the tool work fully offline, or only degrade gracefully?** If offline operation is a hard requirement, only local-inference tools qualify.
2. **What is the measured memory footprint at your chosen model size?** Check with `free -m` under load, not from the model card.
3. **What is the measured bandwidth per working hour?** Check with `nethogs` on a representative task.
4. **Does the endpoint survive your most restrictive network?** Test with `curl` before rolling the tool out to a team.
5. **Can you pin the model version?** A model that updates silently will change behaviour and invalidate your measurements.
6. **What is the failure mode when the network drops?** A tool that blocks the editor is worse than one that simply stops suggesting.
7. **Who owns the licence and the data?** Self-hosted tools shift cost from subscription to infrastructure and operations; that trade is worth stating explicitly.

## Cost arithmetic you can check yourself

Vendor pricing changes and regional pricing varies, so no fixed table is reliable for long. Instead, build the estimate from stated assumptions. A worked example, clearly illustrative:

- Assume a cloud GPU instance at $1.00 per hour (check current pricing for your region and instance type).
- Assume 8 hours of use per working day, 22 days per month: 176 hours.
- 176 × $1.00 = $176 per month for the instance alone.

Now compare against a local model on hardware you already own:

- Assume the machine draws 60W under inference load.
- 176 hours × 0.06 kW = 10.56 kWh per month.
- At $0.20 per kWh, that is about $2.11 per month in electricity, plus the amortised hardware cost.

The comparison is not "cloud bad, local good." The cloud instance has no upfront cost, needs no maintenance, and can run a much larger model. The local model has near-zero marginal cost but is capped by your RAM. The point is to compute both numbers with your own figures rather than accepting either vendor's framing.

The same discipline applies to data. If a completion costs roughly 10KB of request and 2KB of response, then 1,000 completions per day is about 12MB per day, or roughly 260MB per month. That is trivial on a fixed connection and material on a metered one. Substitute your provider's actual payload sizes, which you can observe with `nethogs`.

## Failure modes to expect after the main fixes

**Out-of-memory during model load.** The model is too large for available RAM. Reduce parameter count or quantisation, or close other memory-heavy processes.

**TLS certificate verification failures.** Usually indicates interception by a corporate proxy. The correct fix is to install the proxy's root certificate in your trust store, not to disable verification.

**Rate limiting.** Free tiers commonly cap requests per day or per minute. Local models remove the cap but also remove the capability of the larger hosted model. Decide which you need for the task at hand.

**Model not found.** Registries change and tags are sometimes removed. List what is actually available locally before assuming a configuration error:

```bash
ollama list
```

**Disk exhaustion from cached models.** Model files are large and are not always cleaned up. Check the cache directory size periodically and remove models you no longer use.

## FAQ

**Can a local model replace a cloud assistant entirely?**
For autocomplete, short refactors and boilerplate, often yes. For tasks that require reasoning across many files or a large context window, current small local models are usually weaker. Many developers run both and choose per task.

**Is disabling autocomplete the right first move?**
It is the fastest way to confirm that background traffic is the problem. Once confirmed, prefer narrowing autocomplete to specific file types over disabling it globally.

**How do I know if a model will fit before downloading it?**
Estimate from parameter count and quantisation, then verify with `free -m` under load. Downloads are large enough that guessing is expensive on a metered connection.

**Do local models remove all data cost?**
They remove inference traffic. You still pay once to download the model, and you pay in electricity and hardware. For a team, add the operational cost of running and updating the runtime.

**What should a team document about AI tooling?**
Which tools are approved, which networks they work on, the measured bandwidth and memory footprint, and the fallback when the network is unavailable. That document prevents each new team member from rediscovering the same constraints.

## What to do in the next 30 minutes

Run a bandwidth monitor for five minutes of your normal work and identify the top consumer:

```bash
sudo apt install -y nethogs
sudo nethogs
```

If an AI assistant process is near the top of that list while you are not actively requesting completions, you have found a configuration problem rather than a fundamental limitation. Open its settings, disable background suggestions, and re-measure the same five-minute task. Record both numbers — before and after — so that the next person on your team has evidence instead of an opinion.
