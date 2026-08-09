"""
requeue_fabricated_citations.py

Takes the confirmed named-source citation list from content_triage_v2.py
and marks those posts for regeneration through the existing pipeline
(same code path as a normal refresh — nothing new, no manual editing).

Run from repo root: python requeue_fabricated_citations.py
Requires blog_system.py's BlogSystem class to already exist in this repo
with a refresh/regenerate entrypoint (adjust the import + call below to
match your actual class/method name — this assumes the same one used by
your documented `refresh` pipeline command).
"""
import json
from pathlib import Path

CONFIRMED_SLUGS = [
    "ai-incident-response-when-it-helps-and-when-it-burns",
    "ai-proof-your-salary-5-skills-that-still-pay-2026",
    "llm-latency-what-kills-speed-first",
    "ai-proof-your-salary-the-3-skills-worth-40-more",
    "self-healing-pipelines-llms-vs-agents-in-2026",
    "temporal-changed-how-we-think-about-agent-workflows",
    "ai-debt-trap-llm-code-vs-human-code-in-prod",
    "ai-backend-costs-in-africa-3-hidden-taxes",
    "agent-drift-is-eating-your-product",
    "ai-interviews-the-new-tests-replacing-leetcode",
    "6-ways-to-ship-real-time-without-websockets",
    "avoid-webhook-delivery-nightmares-in-2026",
    "prove-platform-roi-by-measuring-what-matters",
    "automate-sboms-before-audits-hit",
    "ai-sandboxes-when-self-service-goes-wrong",
    "2026-remote-pay-beat-the-ppp-trap",
    "ai-ops-tools-automated-capacity-vs-runbooks-in-2026",
    "5-signs-youre-done-with-big-tech",
    "survive-ai-saas-disruption-in-2026",
    "ai-code-review-the-20-trap-in-production",
    "ask-for-20-more-in-2026",
    "ai-interviews-need-real-logs-the-401-you-never",
]

QUEUE_FILE = Path("./regeneration_queue.json")


def main():
    existing = json.loads(QUEUE_FILE.read_text()
                          ) if QUEUE_FILE.exists() else []
    existing_slugs = {e["slug"] for e in existing}

    added = 0
    for slug in CONFIRMED_SLUGS:
        if slug in existing_slugs:
            continue
        existing.append({
            "slug": slug,
            "reason": "fabricated_named_source_citation",
            "priority": "high",
        })
        added += 1

    QUEUE_FILE.write_text(json.dumps(existing, indent=2))
    print(
        f"Queued {added} new posts for regeneration ({len(existing)} total in queue).")
    print("Run your existing refresh/regenerate command against this queue file —"
          " the fabricated-citation gate added to blog_system.py will block a repeat.")


if __name__ == "__main__":
    main()
