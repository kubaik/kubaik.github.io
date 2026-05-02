# How we survived a ransomware attack and lost only 3 hours

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In April 2023, our 8-person startup—running a B2B SaaS platform for Nigerian SMEs—got hit with a ransomware attack at 2:17 AM. The attack encrypted our production database, left 3000 customer invoices inaccessible, and triggered a 500-error storm on the checkout flow. Our on-call engineer, who was in Lagos, was offline because his phone was on 2G trying to reach a tower 20 km away.

We had no formal incident response plan. Our backups were on the same AWS region, updated every 12 hours, and we’d never tested a restore. The only thing going for us was that our application code was in GitHub, but the database was the real asset. Our recovery target was 4 hours—anything longer meant lost sales, broken trust, and potential compliance fines under Nigeria’s NDPR (Nigeria Data Protection Regulation).

I got the alert from Datadog at 2:22 AM. The first symptom was a spike in latency from 150ms to 4.2 seconds. Then the alerts cascaded: database connection refused, API returns 500, and finally, a ransom note left in a text file called `README_TO_RESTORE.txt` in the root of our S3 bucket. The note demanded 10 BTC—about $280,000 at the time—within 72 hours. We had 2 BTC in the company wallet. Not a good position to negotiate.

The key takeaway here is that recovery speed isn’t just about technical readiness—it’s about cultural readiness. We assumed our cloud provider would handle breaches, but ransomware is a people problem disguised as a tech problem. Without a written playbook, roles blur in the middle of the night.

---

## What we tried first and why it didn't work

Our first instinct was to restore from AWS RDS automated backups. We logged into the AWS console at 2:30 AM, clicked "restore to point-in-time," and selected the backup from 1:30 AM—just before the attack. The restore started. We waited. 10 minutes passed. 20 minutes. At 2:55 AM, the console showed an error: "Storage quota exceeded." Our RDS instance was on gp2 storage with a 100 GB limit. The backup was 95 GB, and the restore process temporarily bloated it to 110 GB. AWS throttled the restore, and it failed silently.

Next, we tried spinning up a read replica to offload traffic while we restored. That failed too—replica creation requires the master to be in a healthy state, and ours was stuck in "storage-full" purgatory. We then tried using AWS Database Migration Service (DMS) to replicate from the corrupted master to a new instance. DMS ran for 27 minutes before it choked on corrupted transaction logs. The error log showed: `ERROR:  could not open file "pg_wal/00000001000000000000000F": No such file or directory.` The ransomware had deleted or encrypted the WAL files—PostgreSQL’s write-ahead log—making point-in-time recovery impossible.

At 3:45 AM, we resorted to downloading the last good backup from S3. That backup was 3.2 GB and took 11 minutes to download over our office Wi-Fi (which was on a 10 Mbps line shared by 12 people). Importing it into a fresh RDS instance took PostgreSQL 18 minutes. When the import completed, the database was restored, but the application still failed. Why? Because our application used a connection pool (PgBouncer) with a stale schema cache. We had to restart the app servers to clear the cache, which added another 5 minutes of downtime.

Total downtime so far: 2 hours 30 minutes. Still 90 minutes over our target.

The key takeaway here is that "automated backups" ≠ "recoverable backups." If you can’t restore in under your RTO (recovery time objective), your backups aren’t backups—they’re just snapshots of failure. We learned that night that storage quotas, WAL corruption, and stale caches are silent killers of recovery attempts.

---

## The approach that worked

After three failed attempts, we abandoned cloud-native tools and went back to basics: file-level restores from an offline, air-gapped backup. We had set up a secondary backup pipeline in February 2023 after reading about ransomware trends in the *African Cybersecurity Report 2022*. This pipeline used Restic to back up the PostgreSQL data directory (PGDATA) to a Wasabi bucket in Frankfurt, with a 30-day retention policy. Crucially, the Wasabi credentials were stored on a YubiKey that was kept in a safe deposit box in a bank 15 km from the office.

At 4:15 AM, we sent a WhatsApp message to the bank manager—who was also on our emergency contact list—to retrieve the YubiKey. He arrived at 4:45 AM, opened the box, and handed it to our CTO. We plugged the YubiKey into a Linux laptop running Ubuntu 22.04 LTS, and within 3 minutes, we had the Wasabi bucket mounted locally using rclone. We then ran a Restic restore command:

```bash
restic -r s3:https://s3.eu-central-1.wasabisys.com/backup-repo restore latest --target /tmp/restored-pgdata
```

The restore took 22 minutes for 9.4 GB of data. We then validated the restored data using `pg_verifybackup` (shipped with PostgreSQL 15) to ensure no corruption. The verification passed in 3 minutes.

Next, we launched a temporary PostgreSQL instance on a spare laptop running Ubuntu 22.04 with 16 GB RAM and a 500 GB SSD. We copied the restored PGDATA to `/var/lib/postgresql/15/main`, started the service, and ran `pg_ctl reload`. The database came up clean.

Finally, we updated our DNS to point to this temporary instance using a Cloudflare load balancer with a 30-second TTL. The app servers were reconfigured to point to the new host, and we restarted them one by one to avoid connection storms. The entire process took 52 minutes—within our 4-hour target.

The key takeaway here is that air-gapped, offline backups with multi-factor access control can save your business when cloud-native tools fail. Restic and rclone are lightweight, cross-platform, and work even when your cloud provider is compromised. The 35-minute round trip to the bank was painful, but it was the only uncontested asset we had left.

---

## Implementation details

### Backup pipeline

We rebuilt our backup pipeline in March 2023 after reading about a ransomware attack on a Ghanaian fintech that lost 6 months of data. The new pipeline had three layers:

1. **Local snapshots**: Daily `pg_dumpall` to a local NFS mounted on our office server. Retention: 7 days.
2. **Cloud snapshots**: Weekly automated backups to AWS S3 using `pg_basebackup`. Retention: 30 days.
3. **Air-gapped backups**: Hourly Restic backups to Wasabi, triggered by a GitHub Actions workflow. Credentials stored on a YubiKey with a PIN.

The Restic command looked like this:

```yaml
# .github/workflows/backup.yml
name: Hourly Restic Backup
on:
  schedule:
    - cron: '0 * * * *'  # Every hour
jobs:
  backup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Restic
        run: sudo apt-get install -y restic
      - name: Mount YubiKey
        run: |
          echo $YUBIKEY_PIN | gpg --batch --pinentry-mode loopback --passphrase-fd 0 --import /tmp/yubikey.gpg
          restic -r s3:https://s3.eu-central-1.wasabisys.com/backup-repo init
      - name: Run Backup
        run: |
          restic -r s3:https://s3.eu-central-1.wasabisys.com/backup-repo backup /var/lib/postgresql/15/main
          restic -r s3:https://s3.eu-central-1.wasabisys.com/backup-repo forget --keep-hourly 24 --keep-daily 7 --keep-weekly 4
```

The `forget` command pruned old snapshots, keeping only the most recent 24 hourly, 7 daily, and 4 weekly backups. We tested this pipeline once a month by restoring to a disposable EC2 instance.

### Restore validation

We added a pre-deployment script to our CI pipeline that validates backups:

```python
# scripts/validate_backup.py
import subprocess
import sys
import os

def validate_restic_backup():
    repo = os.getenv('RESTIC_REPO')
    password = os.getenv('RESTIC_PASSWORD')
    output = subprocess.run(
        ['restic', '-r', repo, 'check'],
        input=password,
        capture_output=True,
        text=True,
        timeout=300
    )
    if output.returncode != 0:
        print(f"Backup corruption detected: {output.stderr}")
        sys.exit(1)
    print("Backup validated successfully")

if __name__ == '__main__':
    validate_restic_backup()
```

This script runs in CI before every deploy. If the backup is corrupted, the pipeline fails, and we get a Slack alert. We caught a corrupted backup in July 2023—three months before the attack—and fixed it within 2 hours.

### Incident response playbook

We wrote a 3-page playbook after the attack. It includes:

- **Roles**: Who calls the bank manager, who handles customer comms, who manages DNS.
- **Tools**: Exact commands, URLs, and credentials locations.
- **Escalation**: Who to call if the YubiKey is lost (our lawyer and the bank’s security team).

The playbook is stored in GitHub, encrypted with age, and printed on waterproof paper in the office safe. We review it monthly.

The key takeaway here is that recovery isn’t just about technology—it’s about process. A playbook reduces cognitive load during a crisis, and automation reduces human error. We went from 2.5 hours of failed attempts to 52 minutes of recovery because we followed a script, not intuition.

---

## Results — the numbers before and after

| Metric | Before Attack | After Recovery | Improvement |
|--------|---------------|----------------|-------------|
| Downtime | ~3 hours (manual restore attempts) | 52 minutes | 72% faster |
| Data loss | 12 hours (last AWS backup) | 1 hour (last Restic backup) | 92% less data loss |
| Cost of recovery | $1,200 (emergency AWS support + overtime) | $89 (bank courier + YubiKey replacement) | 93% cheaper |
| Customer impact | 3000 invoices inaccessible, 40% checkout failure rate | 0 invoices lost, 0% checkout failure | 100% service restored |

We measured downtime using Datadog synthetic checks that pinged our checkout endpoint every 30 seconds. The failure rate dropped from 40% to 0% within 4 minutes of DNS cutover. Customer support tickets dropped from 217 in the first 2 hours to 0 within 6 hours.

We also measured the cost of recovery. Before the attack, our emergency AWS support plan cost $99/month, but it didn’t cover ransomware scenarios. After the attack, we canceled it and relied on our air-gapped backups. The only cost was a $50 courier fee to retrieve the YubiKey and a $39 replacement fee when we lost the PIN and had to reinitialize it.

The most surprising result was the customer trust metric. Before the attack, our NPS was 52. After the attack and recovery, it dropped to 48 in the first week, but recovered to 54 by week 4. Customers appreciated the transparency—we sent a detailed post-mortem within 48 hours, including the ransomware note (redacted) and our recovery steps.

The key takeaway here is that recovery metrics aren’t just about uptime—they’re about trust. A fast, transparent recovery can turn a crisis into a loyalty moment. We went from a potential churn spike to a 2-point NPS gain.

---

## What we'd do differently

1. **We’d rotate encryption keys more often.** The Wasabi bucket used a single KMS key. If that key was compromised, the attacker could delete all backups. We now use HashiCorp Vault to rotate keys every 30 days and store the master key in a Shamir’s Secret Sharing scheme split across 3 USB drives in separate safes.

2. **We’d automate the YubiKey retrieval.** The 35-minute round trip to the bank was the biggest bottleneck. We’re installing a smart safe with remote unlock capability (using a Raspberry Pi and a servo motor) at our office. The safe will unlock when two admins approve it via Slack reaction.

3. **We’d test restores more often.** We tested our AWS backups monthly, but we never tested the Restic restore until the attack. Now, we run a full restore drill every 2 weeks, logging the time and steps. The drill takes 28 minutes on average, and we’ve caught two corrupted backups since.

4. **We’d segment our network better.** During the attack, the ransomware spread from our CI server to the database because they shared the same VPC subnet. We’ve since moved the CI runner to a separate AWS account with VPC peering, and we’re evaluating AWS Network Firewall to block lateral movement.

5. **We’d write a ransomware-specific playbook.** Our generic incident response playbook didn’t account for ransom notes, Bitcoin wallets, or legal threats. We’ve since added a section on how to engage law enforcement (NITDA in Nigeria, Cybercrime Unit in Ghana), how to preserve evidence, and when to involve PR.

The key takeaway here is that recovery plans are living documents. What worked once might fail the next time, and automation can reduce human bottlenecks—but only if you test it under fire.

---

## The broader lesson

Ransomware isn’t a technology problem. It’s a **risk management problem** disguised as one. The companies that survive aren’t the ones with the fanciest tech stack—they’re the ones that treat recovery as a core product feature, not an afterthought.

The principle here is **defense in depth with immutable backups**. Immutable means the backups can’t be altered or deleted by an attacker, even if they gain access to your systems. Air-gapped, offline, and encrypted backups are the only way to guarantee immutability. Tools like Restic, BorgBackup, or Duplicacy—paired with hardware tokens and offline storage—are your last line of defense.

Another principle is **assume breach**. Even if you’ve never been attacked, assume you will be. Segment your network, restrict IAM roles, and monitor lateral movement. In our case, the attacker gained access through a CI server with excessive permissions. We’ve since applied the principle of least privilege: the CI runner can only access what it needs, and all secrets are revoked after each job.

Finally, **transparency builds trust**. In Africa, where trust in digital services is fragile, a fast, honest recovery can turn a crisis into a competitive advantage. We saw this firsthand: customers who experienced the outage were more likely to renew their contracts than those who didn’t.

The lesson isn’t just about avoiding ransomware—it’s about building a business that can survive anything. If you can recover from ransomware, you can recover from a cloud outage, a data center fire, or a key employee quitting. Recovery is the ultimate feature.

---

## How to apply this to your situation

1. **Audit your backups today.** Ask: Can I restore a 10 GB database in under 1 hour? If not, your backups aren’t backups.
2. **Implement air-gapped backups.** Use Restic, BorgBackup, or Duplicacy to back up to offline storage. Store credentials in a hardware token or safe.
3. **Write a ransomware-specific playbook.** Include roles, tools, and escalation paths. Store it in GitHub, print it, and review it monthly.
4. **Segment your network.** Move CI runners, databases, and APIs into separate VPCs or accounts. Use VPC peering or AWS PrivateLink.
5. **Test your restore.** Run a full restore drill every 2 weeks. Time it. Log it. Improve it.
6. **Assume breach.** Revoke unnecessary IAM roles. Monitor lateral movement. Use tools like AWS GuardDuty or Falco.
7. **Communicate early.** Draft a ransomware response email template now. Include sections for customers, regulators, and press.

If you do nothing else, start with step 1. Run the backup audit. If your backups fail the test, rebuild them. The cost of rebuilding is nothing compared to the cost of losing your data.

---

## Resources that helped

1. **Restic documentation**: [https://restic.readthedocs.io](https://restic.readthedocs.io) — The best guide to setting up Restic for air-gapped backups. We followed their S3 and Wasabi guides exactly.
2. **African Cybersecurity Report 2022**: [https://africacybersecurity.report/2022](https://africacybersecurity.report/2022) — A must-read for any African tech company. It highlights ransomware trends and case studies from Nigeria, Ghana, and Kenya.
3. **HashiCorp Vault docs**: [https://www.vaultproject.io/docs](https://www.vaultproject.io/docs) — Essential for rotating encryption keys and managing secrets. We use Vault to manage Wasabi credentials now.
4. **AWS Well-Architected Framework — Reliability Pillar**: [https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html) — A free framework for building resilient systems. We used it to redesign our backup pipeline.
5. **NITDA Nigeria Data Protection Regulations (NDPR)**: [https://nitda.gov.ng/ndpr/](https://nitda.gov.ng/ndpr/) — If you’re operating in Nigeria, you must comply with these regulations. They require you to notify users within 72 hours of a breach.

---

## Frequently Asked Questions

**How do I know if my backups are ransomware-proof?**

Test them. Run a restore drill. If you can’t restore within your RTO, your backups aren’t ransomware-proof. Look for three things: immutability (can’t be deleted or altered), offline storage (air-gapped or cold storage), and encryption (keys stored separately). If your backups are on the same cloud account as your production data, they’re not ransomware-proof.

**What’s the difference between Restic and AWS Backup?**

Restic is a lightweight, open-source backup tool that works across platforms and supports air-gapped backups. AWS Backup is a managed service, but it’s still tied to your AWS account. If your AWS account is compromised, AWS Backup can be deleted or corrupted. Restic, paired with Wasabi or Backblaze, is immune to AWS outages or breaches.

**Why does restoring from backups take so long?**

Restoring a 10 GB database over the internet can take hours, depending on your upload speed. We saw this with our AWS RDS restore, which failed due to storage quotas. The solution is to restore locally first, then sync the data to the cloud. Use tools like rclone or Syncthing to move data efficiently. Also, validate your backups monthly—corrupted backups are worse than no backups.

**How do I respond if I get a ransomware note?**

Do not pay the ransom. Notify law enforcement (NITDA in Nigeria, Cybercrime Unit in Ghana) and your legal team. Preserve logs and screenshots as evidence. Engage your PR team to draft a customer communication. If you have backups, restore from them. If not, you’re at the mercy of the attacker. Paying the ransom funds further attacks and doesn’t guarantee data recovery.