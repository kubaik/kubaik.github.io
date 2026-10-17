# Why leader election fails on cloud networks

Leader election looks simple on paper: a set of nodes agrees on one leader, and if the leader dies, another takes over. On cloud networks the failure modes are subtle and often invisible until they produce a split brain, a stalled cluster, or a leader that is repeatedly elected and deposed in a tight loop. The vendor documentation usually covers setup and goes quiet on failure behavior, so the tradeoffs have to be reconstructed from the algorithms themselves.

This article walks through why those failures happen, the approaches that commonly break, and a practical pattern for making leader election more resilient. It closes with the metrics worth watching and the configuration knobs to check first.

## The problem, in general terms

Leader election is a consensus problem. Algorithms such as Raft and Paxos are designed to tolerate crash failures and network partitions, but they rely on assumptions about the network that cloud environments violate in ways that are easy to miss. The core assumption is that message delivery is either timely or not delivered at all — there is no notion of a message that arrives very late. In cloud networks, packets can be delayed by hundreds of milliseconds or seconds due to congestion, retries, or virtual network behavior, and they can also be reordered or duplicated. These delays can cause a follower to conclude the leader is dead when it is merely slow, triggering an election that should not happen.

A second assumption is that clocks are roughly synchronized. Cloud providers offer time synchronization services, but clock drift still occurs, and algorithms that use timeouts to detect failures are sensitive to it. If a node's clock jumps forward, it may time out a healthy leader prematurely. If it jumps backward, it may wait too long to detect a real failure.

A third assumption is that the network is homogeneous and stable. Cloud networks are not: they span multiple availability zones with varying latency, they have noisy neighbors, and they can experience transient packet loss during live migrations or host maintenance. A timeout value that works in one part of the network may be too aggressive or too lax in another.

## The approaches that commonly fail, and why

### Naive heartbeat with a fixed timeout

The simplest approach is for each node to send heartbeats to the others and declare the leader dead if no heartbeat arrives within a fixed timeout. This fails when the timeout is too short: a temporary network hiccup or a garbage collection pause on the leader causes followers to start an election. If multiple followers time out simultaneously, they may split the vote, producing no leader or frequent leader churn. If the timeout is too long, failover takes too long and the system is unavailable for that duration. Finding a fixed timeout that works across all conditions is close to impossible on a cloud network.

### Using a single external lock service

Many teams use a distributed lock service such as etcd, Consul, or ZooKeeper to hold the leader lock. These systems implement consensus internally, but using them as a lock service introduces a new dependency: the lock service itself must be highly available. If it becomes unreachable due to a network partition, the application cannot determine whether the leader is still alive. Lease-based implementations renew a TTL, but if the application cannot renew because of network trouble, it may lose leadership even though it is healthy. Worse, if the lock service is partitioned and a lease expires and is re-acquired without proper fencing, two applications can both believe they hold the lock.

### Relying on cloud provider instance metadata

Some designs use cloud provider APIs, such as instance status checks, to determine whether a node is healthy. These APIs are eventually consistent and can lag behind reality. A node might be marked unhealthy while it is still running and serving traffic, causing unnecessary failover. Conversely, a node might be marked healthy after it has actually failed, delaying failover.

### Ignoring clock skew

Algorithms that use wall-clock time for lease expiration are vulnerable to clock skew. Even with NTP, clocks can drift by milliseconds to seconds. If a leader's clock is slow, it may believe its lease is still valid while followers with faster clocks believe it has expired. That is a direct path to two leaders acting simultaneously.

## The approach that works in practice

The most robust approach is to use a consensus protocol such as Raft, which is designed to handle these issues, but to configure it carefully and add safeguards. Raft uses a randomized election timeout to reduce split votes and a term number to ensure only one leader is elected per term. Raft alone does not eliminate unnecessary elections caused by network delay; it makes them less likely to cause inconsistency.

A practical pattern combines Raft with a lease-based mechanism that is resilient to clock skew. Instead of relying on wall-clock time, use a monotonic clock for timeouts and lease durations. Monotonic clocks are unaffected by NTP adjustments and only move forward. In Go, `time.Now()` carries a monotonic reading that `Sub` and `Before` use, so comparisons are immune to wall-clock jumps. In Java, `System.nanoTime()` provides a monotonic clock. This ensures that a node's timeout calculations are not skewed by clock adjustments.

Another key practice is a quorum-based approach, where a leader must maintain contact with a majority of nodes to remain leader. That is what Raft does. A lightweight alternative is a gossip protocol with an adaptive failure detector; the SWIM protocol, as implemented in memberlist, uses indirect probing and suspicion to reduce false positives. SWIM is a membership protocol, not a consensus protocol, so it does not provide strong consistency for leader election. It can detect failures, but a consensus layer is still needed to agree on the leader.

For many applications the practical approach is a managed or self-hosted coordination service that handles these complexities — etcd, Consul, or ZooKeeper — combined with an application designed to tolerate the loss of that service. That means a fallback: if the coordination service is unreachable, the current leader continues to act as leader for a bounded period (a lease), and followers do not attempt to elect a new leader until that lease expires. This is lease-based leader election with a grace period. The lease duration should be longer than the maximum expected partition duration, but short enough to ensure timely failover. A common starting point is 10–30 seconds, chosen against your recovery time objective rather than copied from a default.

## Implementation details

A minimal lease-based leader election with etcd works like this: each candidate tries to create a lease and then a key attached to that lease. The first to succeed becomes leader. The leader must periodically renew the lease. If it fails to renew, the lease expires and the key is deleted, allowing another candidate to take over. To avoid split brain due to network partitions, the leader should also confirm it can still reach a quorum of etcd nodes before performing critical actions. Because etcd itself uses Raft, a successful lease renewal implies contact with a quorum. The remaining risk is that the leader is partitioned from the etcd cluster but still believes it is leader because its local lease state has not expired. Mitigations: use a short lease TTL and renew frequently, and verify leadership with a linearizable read of the leader key. A serializable read (`clientv3.WithSerializable()`) may be served by a stale follower and is not sufficient; omitting that option requests a linearizable read, which requires quorum.

Here is a simplified example in Go:

```go
package main

import (
    "context"
    "log"
    "time"

    clientv3 "go.etcd.io/etcd/client/v3"
    "go.etcd.io/etcd/client/v3/concurrency"
)

func main() {
    cli, err := clientv3.New(clientv3.Config{
        Endpoints:   []string{"localhost:2379"},
        DialTimeout: 5 * time.Second,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer cli.Close()

    session, err := concurrency.NewSession(cli, concurrency.WithTTL(10))
    if err != nil {
        log.Fatal(err)
    }
    defer session.Close()

    election := concurrency.NewElection(session, "/myapp/leader")
    ctx := context.Background()

    // Campaign blocks until this candidate becomes leader or ctx is canceled.
    if err := election.Campaign(ctx, "candidate1"); err != nil {
        log.Fatal(err)
    }

    log.Println("became leader")

    // Do leader work here. Periodically verify leadership with a
    // linearizable read of the leader key.
    ticker := time.NewTicker(2 * time.Second)
    defer ticker.Stop()
    for {
        select {
        case <-ticker.C:
            resp, err := cli.Get(ctx, "/myapp/leader")
            if err != nil {
                log.Println("lost leadership:", err)
                return
            }
            if len(resp.Kvs) == 0 || string(resp.Kvs[0].Value) != "candidate1" {
                log.Println("lost leadership: another candidate is leader")
                return
            }
        case <-session.Done():
            log.Println("session expired, lost leadership")
            return
        }
    }
}
```

The `session` has a TTL of 10 seconds and sends keepalives automatically. If the session expires, `session.Done()` is closed and the leader must stop acting as leader. The periodic linearizable read of the leader key is the safeguard against split brain: it catches the case where the leader cannot reach etcd but its local session has not yet timed out.

Note that the value stored under the election key is not necessarily the raw candidate name; the concurrency package stores its own encoded value. In production, compare against the value returned by `election.Leader(ctx)` or use the package's own leadership checks rather than hard-coding a string comparison. The example above shows the shape of the check; verify the exact encoding against the client library version in use.

## What outcomes to expect, and their limits

With a properly configured lease-based election, failover occurs within the lease TTL plus the time for a new leader to campaign. With a 10-second TTL, that is roughly 10–15 seconds, and during that window the system may be unavailable for writes. If an application requires sub-second failover, a different approach is needed, such as primary-backup replication with synchronous replication and a hot standby, which carries its own tradeoffs.

The lease-based approach also depends on the coordination service being available. If the etcd cluster loses quorum, no new leader can be elected, and the existing leader loses its lease after the TTL. Quorum loss in the coordination service therefore becomes a complete outage of leader election. Some teams layer a local leader election within a partition on top of a global election across partitions with reconciliation; this is complex and rarely needed.

The limit is fundamental: strong consistency and high availability cannot both hold during a network partition. A choice is required. Most systems choose consistency (CP) for leader election, accepting that the minority partition cannot elect a leader. That is the correct choice for most applications.

## What to watch out for

- **Clock skew**: Use monotonic clocks for timeouts. Ensure NTP is configured and monitored. In Kubernetes, node clocks are usually synchronized; on VMs, check `chronyd` or `systemd-timesyncd`.
- **Network latency variability**: Set timeouts from observed p99 latency, not the average. etcd's internal election timeout is randomized between 1000 and 2000 ms by default; tuning etcd's internals is rarely necessary, so tune the application's lease TTL instead.
- **GC pauses**: A long GC pause on the leader can cause missed keepalives. Choose a lease TTL longer than the maximum GC pause. For JVM applications, monitor GC pauses and size the TTL accordingly. Go pauses are usually sub-millisecond but should still be considered.
- **Fencing tokens**: Even with a lease, there is a window where an old leader may still be acting while a new leader has been elected. Use fencing tokens — a monotonically increasing number — so downstream systems reject requests from an old leader. For example, include the token in database writes and have the database reject any write with a token lower than the highest seen.
- **Monitoring**: Track leader changes, election duration, and lease renewal failures. Alert on frequent leader changes, for example more than one per hour, as it indicates instability.

## The broader lesson

Leader election is a distributed consensus problem, and consensus is hard. Cloud networks introduce variability that breaks simplistic assumptions. The key principle is to design for failure: assume partitions will happen, clocks will skew, and messages will be delayed. Use proven algorithms such as Raft and battle-tested implementations such as etcd or Consul rather than writing your own. Even then, understand the failure modes and add safeguards such as fencing tokens and monotonic clocks. In distributed systems the network is not reliable, and any algorithm that assumes otherwise will eventually fail in subtle ways.

## How to apply this to your situation

Audit the current leader election mechanism. If it uses a fixed heartbeat timeout, replace it with a lease-based approach backed by a coordination service. If etcd or Consul is already in use, check that the lease TTL suits the network conditions and that leadership is verified with linearizable reads. Add fencing tokens to writes to prevent split brain. Monitor leader changes; frequent changes point to network latency or GC pauses.

## Frequently Asked Questions

### Why does the leader keep changing every few minutes?

Frequent leader changes usually mean the lease is expiring prematurely. Common causes are latency spikes causing missed keepalives, GC pauses on the leader, or clock skew. Check the coordination service logs for lease expiration events and correlate them with network and GC metrics. Increasing the lease TTL can help, but fix the underlying cause first.

### Can a relational database serve as the lock service for leader election?

It is possible but risky. Databases are not designed for high-frequency lease renewals, and they may not provide linearizable reads under all isolation levels. For example, PostgreSQL's default Read Committed isolation does not prevent phantom reads, and `SELECT ... FOR UPDATE` must be used carefully. If the database is partitioned, two leaders can result. A dedicated coordination service is a better fit.

### How do you test leader election under network partitions?

Use `tc` (traffic control) on Linux to introduce delays and packet loss, or a chaos engineering tool to simulate partitions. Run a cluster of three or five nodes and partition the leader from the followers. Verify that a new leader is elected and that the old leader stops acting as leader within the lease TTL. Also test clock skew by adjusting the system clock on one node in a test environment and observing the behavior.

### What is a fencing token and why is it needed?

A fencing token is a monotonically increasing number included with every write request. When a new leader is elected, it increments the token. Downstream systems reject any write with a token lower than the highest seen. This prevents an old leader that is still running, for example after a long GC pause, from writing after a new leader has taken over. Without fencing tokens, split brain is possible even with lease-based election.

## Action for the next 30 minutes

Open the leader election configuration for one service — the etcd lease TTL, the Consul session TTL, or the application's own lease setting — and record the current value alongside the maximum GC pause and p99 network latency you can observe for that service. If the TTL is shorter than the sum of those two figures, raise it to at least that sum and note the new failover window it implies.
