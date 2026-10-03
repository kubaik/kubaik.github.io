# SBOM + attestation that gets used

Opinions on building accessible tend to change fast after watching it fail somewhere it wasn't supposed to. Here's what changes once the guessing is replaced with measurement. The workaround gets copy-pasted forward long after the original reason is forgotten.

## What this list actually solves

An SBOM that sits in an S3 bucket and an attestation nobody verifies are compliance theater. The artifact exists, the audit checkbox is ticked, and the next Log4Shell-style event still takes three weeks to triage because the data is stale, unsigned, or in a format no tool can query. The real problem is not generating an SBOM — most build systems can emit one in a few lines — it is making the SBOM and its attestation a live input to deployment decisions, incident response, and policy enforcement.

This list covers the tools and patterns that push SBOMs and attestations into the path of actual work: admission control, CI gates, vulnerability triage, and provenance verification. Each entry is a mini-review: what it does, one concrete strength, one concrete weakness, and who it fits. The framing throughout is that an SBOM is only useful when something consumes it automatically, and an attestation is only useful when something rejects the build without it.

## Evaluation criteria

Six criteria separate tools that get used from tools that get filed.

1. **Format coverage.** Does it emit or consume SPDX and CycloneDX, the two formats with the widest tooling support? A tool that only speaks a proprietary schema forces every downstream consumer to write a converter.
2. **Signing and verification.** Can the attestation be signed with a keyless or key-based identity and verified by a policy engine? An unsigned attestation is a text file with extra steps.
3. **Integration surface.** Does it hook into CI (GitHub Actions, GitLab CI, Jenkins) and admission controllers (Kubernetes, OPA/Gatekeeper, Kyverno)? If the only way to use it is a CLI a human runs, it will not run.
4. **Queryability.** Can you ask "which running images contain package X version Y?" without writing a custom parser? This is the difference between a 10-minute incident response and a three-day one.
5. **Operational cost.** Storage, signing infrastructure, and the compute to regenerate on every build. A tool that adds two minutes to every CI run gets disabled within a quarter.
6. **Failure mode clarity.** What happens when the SBOM is missing, malformed, or unsigned? A tool that fails open is worse than no tool, because it creates false confidence.

## SBOM + attestation that actually gets used — the options compared

### 1. Syft + Grype (Anchore)

**What it does.** Syft generates SBOMs from container images, filesystems, and archives in SPDX and CycloneDX. Grype consumes those SBOMs (or scans images directly) and matches packages against vulnerability databases.

**Strength.** The two tools share a data model, so the SBOM Syft emits is exactly what Grype wants. You can scan an image once in CI, store the SBOM, and re-scan that same SBOM months later when a new CVE drops — without rebuilding the image. That is the single most useful property an SBOM can have.

**Weakness.** Syft's accuracy depends on the package ecosystem. For language-level dependencies resolved by a lockfile (npm, pip, go.mod), it is reliable. For OS packages installed via scripts that bypass the package manager, coverage drops. Teams that install binaries with `curl | sh` in a Dockerfile will find gaps.

**Best for.** Teams that want a fast, open-source path to SBOM generation and vulnerability matching without committing to a commercial platform. Start here if you have nothing.

### 2. Trivy (Aqua Security)

**What it does.** Trivy scans container images, filesystems, git repos, and Kubernetes clusters. It can output SBOMs in SPDX and CycloneDX and also scan an existing SBOM for vulnerabilities.

**Strength.** One binary covers image scanning, IaC misconfiguration, secrets, and SBOM generation. For teams that do not want to assemble a pipeline from four tools, Trivy is the shortest path from zero to a scan running in CI.

**Weakness.** Because it does so much, the SBOM output is a feature rather than the core design. Teams with strict SPDX conformance requirements or complex multi-architecture image layouts sometimes hit edge cases that Syft handles more predictably. It is also a large binary, which matters in constrained CI runners.

**Best for.** Teams that want one tool for scanning and SBOM generation, especially if they are already using Trivy for image scanning.

### 3. cosign (Sigstore)

**What it does.** cosign signs and verifies container images, blobs, and attestations. It supports keyless signing via OIDC identity (Fulcio) and stores signatures in a transparency log (Rekor).

**Strength.** Keyless signing removes the hardest operational problem in attestation: key management. A CI job can sign an attestation with its workload identity, and a verifier can check that signature against the expected identity without anyone handling a long-lived private key.

**Weakness.** Keyless verification requires network access to the transparency log and the OIDC issuer's public keys. Air-gapped environments need a different approach (key-based signing with a managed key). Teams also frequently misconfigure the identity check — verifying that *a* signature exists rather than that the signature came from the *expected* CI workflow.

**Best for.** Teams on GitHub Actions, GitLab CI, or any CI with OIDC support that want signed attestations without running their own PKI.

### 4. in-toto attestations

**What it does.** in-toto defines a framework for supply chain attestations. The attestation format wraps a statement (what happened) with a predicate (details) and can be signed. The SLSA provenance format is an in-toto attestation with a specific predicate type.

**Strength.** It is the format that the rest of the ecosystem is converging on. SLSA provenance, Sigstore attestations, and GitHub's build provenance all use in-toto-style envelopes. Adopting it means your attestations are consumable by policy engines and verification tools you have not adopted yet.

**Weakness.** The specification is flexible to the point of being underspecified for beginners. Two teams can both claim "we emit in-toto attestations" and produce payloads that no shared tool can verify, because predicate types and subject naming differ. You need a verification policy, not just a format.

**Best for.** Teams building a supply chain that needs to interoperate with SLSA, Sigstore, and policy engines. It is infrastructure, not a product.

### 5. SLSA provenance (levels 1–3)

**What it does.** SLSA defines a ladder of build integrity. Level 1 means provenance exists. Level 2 means it is signed by the build platform. Level 3 means the build runs in a hardened, isolated environment with non-forgeable provenance.

**Strength.** It gives a shared vocabulary for "how much do we trust this build" that auditors and downstream consumers understand. Reaching level 2 with GitHub Actions or a similar hosted CI is largely a configuration exercise, not a research project.

**Weakness.** Provenance tells you *how* an artifact was built, not *what is in it*. SLSA and SBOM are complementary, not substitutes. Teams that adopt one and skip the other often discover the gap during an incident.

**Best for.** Teams that need to prove build integrity to a customer, regulator, or downstream team, and want a standard rather than a bespoke claim.

### 6. OPA / Gatekeeper and Kyverno

**What they do.** These are Kubernetes admission controllers. They evaluate policies against resources at admission time and can reject workloads that fail. Both can verify image signatures and check attestations.

**Strength.** This is where an attestation stops being a file and becomes a gate. A policy that rejects any pod whose image lacks a valid SLSA provenance attestation from the expected builder converts a supply chain practice into an enforced control.

**Weakness.** Admission control is a blunt instrument. A misconfigured policy can block all deployments, including the one you need to fix the policy. Teams need a dry-run or audit mode first, and a documented break-glass path. Kyverno's policy language is YAML-native and easier for Kubernetes-native teams; Gatekeeper's Rego is more expressive but has a steeper learning curve.

**Best for.** Teams that already run Kubernetes and want SBOM and attestation data to influence what actually gets deployed.

### 7. Dependency-Track (OWASP)

**What it does.** Dependency-Track is an SBOM analysis platform. You upload SBOMs, it tracks components across projects, and it alerts on new vulnerabilities affecting components you already use.

**Strength.** It solves the "stale SBOM" problem. Instead of re-scanning images, you upload the SBOM once and the platform monitors the components in it. When a new CVE is published, you get a list of affected projects without rebuilding anything.

**Weakness.** It is another service to run and back up. The API is capable but the UI is dated, and teams used to commercial dashboards find the reporting thin. It also needs a vulnerability data source configured and kept current.

**Best for.** Organizations with many projects that want a central inventory of components and continuous monitoring rather than per-build scanning.

### 8. GitHub Artifact Attestations

**What it does.** GitHub Actions can generate signed attestations for build artifacts using the workflow's OIDC identity, stored in GitHub's attestation API and verifiable with the `gh` CLI.

**Strength.** If your builds run on GitHub Actions, this is the lowest-friction path to signed provenance. There is no key to manage and no signing infrastructure to run. Verification can be done in a later workflow step or by a downstream consumer.

**Weakness.** It ties your attestation to GitHub. If builds move or you need to verify outside GitHub's tooling, you depend on the `gh` CLI or the underlying Sigstore primitives. It also does not generate an SBOM for you — you still need Syft, Trivy, or similar.

**Best for.** Teams fully on GitHub Actions that want signed provenance without operating Sigstore themselves.

## The strongest default, and why

For most teams starting from zero, the strongest default is **Syft for SBOM generation, cosign for signing, and a policy engine (Kyverno or Gatekeeper) for enforcement**, with SLSA provenance added once the basics work.

The reasoning is that this combination separates concerns cleanly. Syft produces the SBOM. cosign signs the SBOM and the image. The policy engine verifies signatures and attestations at admission time. Each piece can be replaced without rewriting the others, and each piece has a clear failure mode you can test.

The critical detail is that the SBOM must be **attached to the image** as an attestation, not stored in a separate bucket keyed by image digest in a custom scheme. Attaching it means any consumer that can pull the image can also pull the SBOM, using the same registry credentials and the same digest. A separate bucket is where SBOMs go to die.

A minimal cosign attestation flow looks like this:

```bash
# Generate SBOM with Syft
syft packages registry:ghcr.io/example/app:1.2.3 -o spdx-json > sbom.spdx.json

# Attach the SBOM as a signed attestation to the image
cosign attest --predicate sbom.spdx.json \
  --type spdxjson \
  --key env://COSIGN_PRIVATE_KEY \
  ghcr.io/example/app:1.2.3

# Verify the attestation later
cosign verify-attestation \
  --type spdxjson \
  --certificate-identity-regexp 'https://github.com/example/.github/workflows/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/example/app:1.2.3
```

The verification command is the part teams skip. Without it, the attestation is decoration. With it, you have a check that can run in a deployment pipeline and fail the deploy when the attestation is missing or the identity does not match.

## Honorable mentions worth knowing about

**CycloneDX CLI and libraries.** The CycloneDX project provides tooling for generating, validating, and converting SBOMs. Useful when you need to convert between SPDX and CycloneDX or validate that an SBOM conforms to the schema before it enters your pipeline.

**SPDX tools.** The SPDX project maintains its own tooling. SPDX has stronger roots in license compliance; CycloneDX has stronger roots in security. If your primary driver is license obligations rather than vulnerability response, SPDX tooling is worth a look.

**GUAC (Graph for Understanding Artifact Composition).** Aggregates SBOMs, attestations, and other supply chain metadata into a queryable graph. It is the answer to "which of our images contain this package" when the data is spread across many registries and formats. It is also early-stage and adds an operational dependency, so it fits teams with a real query problem, not teams that just want to generate SBOMs.

## Options that look appealing but fail in practice

**The "SBOM as a CI artifact" pattern.** Uploading the SBOM to the CI system's artifact storage and calling it done. This fails because artifacts expire, are not linked to the image digest in any standard way, and are not available to admission controllers or incident responders. It satisfies a checkbox and nothing else.

**Unsigned attestations.** An attestation without a signature proves nothing about who produced it. It is trivially forgeable. If the verification step does not check a signature and an identity, the attestation is not a security control.

**SBOM generation in a separate pipeline.** If the SBOM is generated by a job that runs after the image is built, on a different runner, from a different checkout, the SBOM may not describe the image that was actually shipped. The SBOM must be generated from the same artifact that is deployed, ideally from the image itself rather than the source tree.

**Failing open on missing attestations.** A policy that allows deployments when the attestation is absent is worse than no policy, because it creates the appearance of enforcement. Policies should fail closed, with a documented and audited exception path.

**Treating SBOM as a one-time deliverable.** An SBOM generated at release and never updated is a snapshot. The value comes from continuous monitoring — either re-scanning stored SBOMs or feeding them into a platform that watches for new vulnerabilities. A one-time deliverable is a PDF with extra steps.

## How to choose based on your situation

| Situation | Start with | Add next |
|---|---|---|
| No SBOM or attestation today, on GitHub Actions | Syft + GitHub Artifact Attestations | cosign for image signing, Kyverno for enforcement |
| Kubernetes, want admission control | Trivy for scanning, Kyverno or Gatekeeper | cosign verification in policy |
| Many projects, need central inventory | Dependency-Track | Syft or Trivy to feed it SBOMs |
| Regulated, need build integrity proof | SLSA level 2 via hosted CI | SLSA level 3, in-toto attestations |
| Air-gapped or self-hosted CI | Syft + cosign with key-based signing | Internal Rekor or equivalent transparency log |
| License compliance primary driver | SPDX tooling | CycloneDX conversion if security tooling needs it |

The decision framework is: pick the enforcement point first, then work backward. If the enforcement point is Kubernetes admission, choose tools whose output the admission controller can verify. If the enforcement point is a CI gate, choose tools that run fast enough to gate. If there is no enforcement point, the first project is to create one — otherwise every tool choice is premature.

## Frequently asked questions

**What is the difference between an SBOM and an attestation?**

An SBOM lists the components in an artifact — packages, versions, licenses, and often hashes. An attestation is a signed statement about the artifact, which can include the SBOM as its payload or describe how the artifact was built. The SBOM answers "what is in this?" and the attestation answers "who says so, and can I verify it?" You generally want both, attached to the same image digest.

**Do I need SLSA if I already have an SBOM?**

They solve different problems. An SBOM tells you what components are in an artifact. SLSA provenance tells you how the artifact was built and whether the build environment was trustworthy. A compromised build can produce a perfectly accurate SBOM for a malicious artifact. If your threat model includes build system compromise, you need provenance, not just an SBOM.

**How do I stop SBOMs from going stale?**

Either re-scan stored SBOMs on a schedule or feed them into a platform that monitors components continuously. The key is that the SBOM is stored in a queryable form linked to the image digest, not as a flat file in a bucket. Dependency-Track and GUAC are two approaches; re-running Grype against stored SBOMs in a scheduled job is a third.

**What happens if the attestation verification fails in production?**

The policy should block the deployment and alert. The failure mode to avoid is a policy that logs a warning and proceeds. Teams should test the failure path deliberately — deploy an image without an attestation in a staging cluster and confirm the admission controller rejects it. If it does not, the policy is not working, regardless of what the logs say.

## Final recommendation

Start with one image, one pipeline, and one enforcement point. Generate an SBOM with Syft, attach it to the image with cosign, and add a verification step that fails the build if the attestation is missing or the identity does not match. Do not try to cover every repository on day one.

The next 30 minutes: pick your most-deployed image, run `syft packages <image> -o spdx-json` against it, and look at the output. If the component list is obviously incomplete — missing OS packages you know are installed, or missing language dependencies — you have found your first real problem, and it is a problem worth fixing before you build any policy on top of the data.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the AI content policy.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** October 2026
