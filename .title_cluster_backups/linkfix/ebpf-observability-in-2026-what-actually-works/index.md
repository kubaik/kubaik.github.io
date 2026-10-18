# eBPF observability in 2026: what actually works

The conventional advice on eBPF is incomplete in one specific, costly way: it stops at "hello world" and never explains the verifier failures that make teams roll back. This walks through the failure modes and the reasoning behind the fix.

## Why eBPF projects stall

eBPF promises zero-instrumentation profiling, runtime security enforcement, and packet-level visibility without recompiling applications. In practice, the tooling surface is raw, kernel version requirements are strict, and the learning curve is steep. The part that trips people up is that a verifier rejection at load time can look like a kernel bug, and the error messages rarely map cleanly to the underlying rule.

Teams commonly see a kernel warning or panic after deploying a new eBPF program. The stack trace may end in a BPF trampoline or JIT path, and dmesg shows a verifier rejection with a line like `invalid mem access off=-480 size=16`. That message means the program tried to dereference kernel memory directly instead of using the safe helpers such as `bpf_probe_read_kernel`. The verifier enforces this strictly, and the fix is architectural rather than a one-line patch.

This article covers the two eBPF use cases that justify the complexity: production observability and runtime security. Both require a modern kernel and toolchain, and both are stable enough to deploy if you avoid the common traps.

## Prerequisites and what you will build

To follow this guide you need:

- A Linux host with kernel 6.6 or later. Kernel 6.8 is a reasonable target because it includes `bpf_iter` infrastructure for efficient map iteration and BPF LSM support. Verify with `uname -r`.
- clang 18 and LLVM 18. Older versions may reject valid programs with `unreachable insn` errors. Install with `apt install clang-18 llvm-18`.
- libbpf 1.3.0 or later, built from source with static libraries enabled to avoid runtime linker issues.
- A container runtime that permits the required capabilities. Loading eBPF programs generally requires `CAP_BPF` and often `CAP_SYS_ADMIN` or `CAP_PERFMON` depending on the program type.

You will build two artifacts:

1. A kprobe-based eBPF program that counts TCP retransmissions per socket and exposes the data through a ring buffer for userspace consumption.
2. A syscall filter eBPF program that returns `EPERM` for `execve` calls made by a specific UID, similar to a seccomp profile but enforced at runtime without container restarts.

The observability program uses a hash map sized to hold per-socket counters. The security program uses a per-CPU array to reduce lock contention when many threads hit the same syscall.

## Step 1 — set up the environment

Start with a host running a recent kernel. Confirm BPF support:

```bash
uname -r
# Must print 6.6.0 or later
cat /proc/sys/net/core/bpf_jit_enable
# Should print 1
ls /sys/fs/bpf
# Should show the bpffs mount point
```

Install the toolchain:

```bash
sudo apt update && sudo apt install -y clang-18 llvm-18 libelf-dev libbpf-dev linux-headers-$(uname -r)
clang-18 --version
llvm-config-18 --version
```

Build libbpf from source to ensure static linkage and BTF generation:

```bash
git clone --depth 1 --branch v1.3.0 https://github.com/libbpf/libbpf.git
cd libbpf/src
make BUILD_STATIC_LIBS=ON OBJDIR=./build install_prefix=/usr/local
sudo ldconfig
ls /usr/local/lib/libbpf.a
```

Kernel tuning prevents common issues. Add to `/etc/sysctl.d/99-bpf.conf`:

```
net.core.bpf_jit_enable=1
kernel.unprivileged_bpf_disabled=0
kernel.bpf_stats_enabled=1
```

Apply and reboot:

```bash
sudo sysctl --system
sudo reboot
```

After reboot, verify BPF is usable:

```bash
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_execve { @[comm] = count(); }'
# Ctrl-C after 10 seconds, expect non-zero counts
```

If you see "BPF LSM not enabled" in dmesg and need LSM hooks, add `lsm=bpf` to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`, run `sudo update-grub`, and reboot.

## Step 2 — core implementation

Create a project directory with two subdirectories: `tcpretrans` and `execfilter`. Each will compile into a BPF ELF object and load via libbpf.

Start with `tcpretrans`:

```c
// tcpretrans.bpf.c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, u32);           // socket inode
    __type(value, u64);         // retransmission count
    __uint(map_flags, BPF_F_NO_PREALLOC);
} retrans_map SEC(".maps");

SEC("kprobe/tcp_retransmit_skb")
int BPF_KPROBE(tcp_retrans, struct sock *sk)
{
    u32 ino = BPF_CORE_READ(sk, sk_socket, file, f_inode, i_ino);
    u64 *count = bpf_map_lookup_elem(&retrans_map, &ino);
    if (count) {
        (*count)++;
    } else {
        u64 zero = 0;
        bpf_map_update_elem(&retrans_map, &ino, &zero, BPF_NOEXIST);
        count = bpf_map_lookup_elem(&retrans_map, &ino);
        if (count) (*count)++;
    }
    return 0;
}

char _license[] SEC("license") = "Dual MIT/GPL";
```

The program attaches to the kernel's internal `tcp_retransmit_skb` function via kprobe and increments a per-socket retransmission counter. The map uses `BPF_F_NO_PREALLOC` to avoid preallocating all entries at load time, which reduces startup latency for large maps.

Compile with:

```bash
clang-18 -O2 -target bpf -c tcpretrans.bpf.c -o tcpretrans.bpf.o
llvm-strip-18 --strip-all tcpretrans.bpf.o
file tcpretrans.bpf.o
# Must print "tcpretrans.bpf.o: ELF 64-bit LSB relocatable, eBPF, version 1 (SYSV), statically linked, stripped"
```

Now write a loader in C++ using libbpf. The loader attaches the kprobe and pins the map to bpffs.

```cpp
// tcpretrans_loader.cpp
#include <bpf/libbpf.h>
#include <unistd.h>
#include <sys/resource.h>
#include <iostream>
#include <cstring>

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

int main()
{
    libbpf_set_print(libbpf_print_fn);
    struct rlimit rlim = { RLIM_INFINITY, RLIM_INFINITY };
    setrlimit(RLIMIT_MEMLOCK, &rlim);

    struct bpf_object *obj = bpf_object__open_file("tcpretrans.bpf.o", nullptr);
    if (libbpf_get_error(obj)) {
        std::cerr << "Failed to open BPF object\n";
        return 1;
    }

    if (bpf_object__load(obj)) {
        std::cerr << "Failed to load BPF object: " << strerror(errno) << "\n";
        bpf_object__close(obj);
        return 1;
    }

    struct bpf_program *prog = bpf_object__find_program_by_name(obj, "tcp_retrans");
    if (!prog) {
        std::cerr << "Program tcp_retrans not found\n";
        bpf_object__close(obj);
        return 1;
    }

    struct bpf_link *link = bpf_program__attach(prog);
    if (libbpf_get_error(link)) {
        std::cerr << "Failed to attach: " << strerror(errno) << "\n";
        bpf_object__close(obj);
        return 1;
    }

    struct bpf_map *map = bpf_object__find_map_by_name(obj, "retrans_map");
    if (!map) {
        std::cerr << "Map retrans_map not found\n";
        bpf_object__close(obj);
        return 1;
    }

    const char *pin_path = "/sys/fs/bpf/tcpretrans_retrans_map";
    if (bpf_map__pin(map, pin_path)) {
        std::cerr << "Failed to pin map: " << strerror(errno) << "\n";
        bpf_object__close(obj);
        return 1;
    }

    std::cout << "eBPF program loaded and map pinned at " << pin_path << "\n";

    pause();
    bpf_link__destroy(link);
    bpf_object__close(obj);
    return 0;
}
```

Compile the loader with clang-18 and link against libbpf.a:

```bash
clang-18 -O2 -std=c++17 -I/usr/local/include -L/usr/local/lib tcpretrans_loader.cpp -o tcpretrans_loader -lbpf -lelf -lz
sudo ./tcpretrans_loader
```

The map is now pinned at `/sys/fs/bpf/tcpretrans_retrans_map`. Pinning keeps the map alive after the loader exits and allows other processes to access it via the bpffs path.

Build the security program `execfilter` similarly:

```c
// execfilter.bpf.c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 1);
} block_list SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_execve")
int trace_execve(struct trace_event_raw_sys_enter *ctx)
{
    u32 key = 0;
    u64 *blocked = bpf_map_lookup_elem(&block_list, &key);
    if (!blocked || !*blocked) return 0;

    u32 uid = bpf_get_current_uid_gid();
    if (uid == 1000) { // Block UID 1000
        bpf_override_return(ctx, -1); // EPERM
    }
    return 0;
}

char _license[] SEC("license") = "Dual MIT/GPL";
```

The program attaches to the `sys_enter_execve` tracepoint and returns `EPERM` for the target UID. The per-CPU array map ensures lockless updates.

Compile and load:

```bash
clang-18 -O2 -target bpf -c execfilter.bpf.c -o execfilter.bpf.o
clang-18 -O2 -std=c++17 -I/usr/local/include -L/usr/local/lib execfilter_loader.cpp -o execfilter_loader -lbpf -lelf -lz
sudo ./execfilter_loader
```

The security policy is now active without container restarts. To confirm, run as UID 1000:

```bash
sudo -u user1000 bash -c 'ls /'
# Should print "bash: /bin/ls: Operation not permitted"
```

Note: `bpf_override_return` is only supported on functions annotated with `ALLOW_ERROR_INJECTION` and requires the kernel to be built with `CONFIG_BPF_KPROBE_OVERRIDE`. Check `/proc/kallsyms` for the target symbol and verify error injection support before relying on this technique.

## Step 3 — handle edge cases and errors

The most common failure mode is map access from interrupt context. If your program uses `bpf_map_lookup_elem` or `bpf_map_update_elem` in a kprobe attached to a high-frequency function like `tcp_retransmit_skb`, the verifier may reject the program with an error about caller-saved registers not being restored. That happens because the kprobe runs in interrupt context, and the verifier cannot prove the register state is preserved.

The fix is to use BPF ring buffers to defer heavy work to process context. Replace the retrans_map with a ring buffer:

```c
#include <bpf/bpf_ringbuf.h>

struct retrans_event {
    u32 ino;
    u64 ts;
} __attribute__((packed));

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16MiB
} rb SEC(".maps");

SEC("kprobe/tcp_retransmit_skb")
int BPF_KPROBE(tcp_retrans, struct sock *sk)
{
    u32 ino = BPF_CORE_READ(sk, sk_socket, file, f_inode, i_ino);
    struct retrans_event *e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;
    e->ino = ino;
    e->ts = bpf_ktime_get_ns();
    bpf_ringbuf_submit(e, 0);
    return 0;
}
```

The ring buffer is lockless and works in interrupt context. The userspace loader consumes events via a ring buffer map.

Another edge case is the limit on active BPF programs. The Linux kernel limits the number of active BPF programs per CPU. If you attach multiple programs to the same kprobe, you may hit an error about too many programs loaded. Check the current limit and adjust if needed:

```bash
cat /proc/sys/kernel/bpf_stats_enabled
# Verify BPF stats are enabled for inspection
```

Memory limits also matter. The verifier uses a limited stack per program. If your program exceeds that, you will see an error like `BPF: stack too deep`. Split large functions or use inline assembly to reduce stack usage. Clang 18's BPF backend supports inline assembly for small helpers.

## Step 4 — add observability and tests

Add Prometheus metrics to the loader by reading the ring buffer and exposing a `/metrics` endpoint:

```cpp
// tcpretrans_exporter.cpp
#include <bpf/libbpf.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/counter.h>
#include <unistd.h>
#include <iostream>
#include <memory>

int main()
{
    auto registry = std::make_shared<::prometheus::Registry>();
    auto& retrans_counter = ::prometheus::BuildCounter()
        .Name("tcp_retransmissions_total")
        .Help("Total TCP retransmissions")
        .Register(*registry);

    auto& family = retrans_counter.Add({});

    struct bpf_object *obj = bpf_object__open_file("/sys/fs/bpf/tcpretrans", nullptr);
    if (libbpf_get_error(obj)) {
        std::cerr << "Failed to open pinned object\n";
        return 1;
    }

    int map_fd = bpf_object__find_map_fd_by_name(obj, "rb");
    if (map_fd < 0) {
        std::cerr << "Map rb not found\n";
        bpf_object__close(obj);
        return 1;
    }

    std::unique_ptr<::prometheus::Exposer> exposer = std::make_unique<::prometheus::Exposer>(":9090");
    exposer->RegisterCollectable(registry);

    while (true) {
        struct retrans_event e;
        int err = bpf_map__ringbuf_read(map_fd, reinterpret_cast<void*>(&e), sizeof(e));
        if (err == sizeof(e)) {
            family.Increment();
        } else if (err == -EAGAIN) {
            usleep(1000);
        } else {
            std::cerr << "Ringbuf read error: " << strerror(-err) << "\n";
            break;
        }
    }

    bpf_object__close(obj);
    return 0;
}
```

Compile with:

```bash
clang-18 -O2 -std=c++17 -I/usr/local/include -L/usr/local/lib tcpretrans_exporter.cpp -o tcpretrans_exporter -lbpf -lprometheus-cpp-core -lprometheus-cpp-pull -lelf -lz
sudo ./tcpretrans_exporter
```

`curl localhost:9090/metrics` returns:

```
# HELP tcp_retransmissions_total Total TCP retransmissions
# TYPE tcp_retransmissions_total counter
tcp_retransmissions_total 1247
```

Add tests that verify the verifier accepts the program and the loader loads without segfaults. Use a CI runner on Ubuntu 24.04 and kernel 6.8:

```python
# tests/test_tcpretrans.py
import pytest
from bcc import BPF
from pathlib import Path

def test_program_loads():
    b = BPF(src_file="tcpretrans.bpf.c")
    assert b.prog_load_ok()
    assert "tcp_retransmit_skb" in b.get_kprobes_attached()
    assert b.map_pin_path("retrans_map") == "/sys/fs/bpf/tcpretrans_retrans_map"

def test_map_pinning_after_load():
    b = BPF(src_file="tcpretrans.bpf.c")
    b.load_func("tcp_retrans", BPF.KPROBE)
    b.pin_map("retrans_map", "/sys/fs/bpf/test_map")
    assert Path("/sys/fs/bpf/test_map").exists()
```

Run tests with pytest and tox:

```bash
pip install pytest tox
tox -e py311
```

A common regression is map pinning failing under seccomp-confined containers. The fix usually requires adding `CAP_BPF` to the container profile.

## How to measure overhead and validate results

Do not trust any published overhead number, including the ones in this article. Measure on your own hardware. The procedure is straightforward:

1. Establish a baseline. Run your workload for a fixed duration and record p50, p99, and p99.9 latency using your existing instrumentation. If you do not have latency histograms, use `bpftrace` to record `nsecs` deltas at the syscall entry and exit points for the hot path.
2. Load the eBPF program and repeat the identical workload. Keep the host, kernel, and workload parameters constant.
3. Compare the histograms, not just the averages. A mean shift of 0.1ms can hide a p99.9 shift of several milliseconds.
4. Use `bpftool prog show` to confirm the program is attached and `bpftool map show` to verify map sizes. Check `/sys/kernel/debug/tracing/trace_pipe` for verifier warnings.
5. Measure the eBPF program's own CPU cost with `perf top -g` or by comparing `getrusage` before and after. The kernel's `bpf_stats_enabled` sysctl exposes per-program run time and run count via `bpftool prog show`.

A realistic expectation is that a well-written kprobe adds single-digit microseconds per invocation. Whether that matters depends entirely on your invocation rate. A kprobe on `tcp_retransmit_skb` fires rarely, so the aggregate cost is negligible. A kprobe on a function called millions of times per second will dominate your CPU profile regardless of how efficient the BPF program is.

## Common failure modes

**Verifier rejection: `invalid mem access`**

The program dereferenced kernel memory directly. Replace direct pointer dereferences with `bpf_probe_read_kernel` or `BPF_CORE_READ`. A typical offender is reading `sk->sk_socket->file->f_path.dentry->d_inode` without the helper.

**Verifier rejection: `stack too deep`**

The BPF stack is limited. Split large functions into smaller ones, or move data into maps instead of stack variables.

**Ring buffer reservation failures**

`bpf_ringbuf_reserve` returns NULL when the buffer is full. Always check the return value and drop the event rather than retrying in the hot path.

**Map pinning failures in containers**

Pinning requires write access to bpffs. In containers, the mount may be read-only or the process may lack `CAP_BPF`. Verify with `mount | grep bpf` and check capabilities with `capsh --print`.

**Tracepoint availability**

Not all tracepoints exist on all kernels. `sys_enter_execve` has been available for a long time, but some newer tracepoints are kernel-version-specific. Check `/sys/kernel/debug/tracing/events/syscalls/` before relying on a tracepoint.

## eBPF versus traditional tools

| Feature | eBPF ringbuf + iter | bpftrace script | Prometheus + node_exporter |
|---|---|---|---|
| Zero instrumentation | yes | yes | no |
| Runtime attach/detach | yes | yes | no |
| Persistent storage | bpffs | no | Prometheus TSDB |
| Security enforcement | yes (LSM) | no | no |
| Kernel requirement | 6.6+ | 4.9+ | any |
| Setup complexity | high | low | medium |

The table shows that eBPF wins on zero instrumentation and runtime attach/detach, but requires a modern kernel and careful verifier tuning. Traditional tools are easier to set up but may require code changes for instrumentation. The choice depends on whether you need runtime enforcement and whether your kernel is new enough.

## FAQ

**Why does my eBPF program fail with `invalid mem access off=-480 size=16`?**

The program tried to access kernel memory without using `bpf_probe_read_kernel` or `bpf_probe_read_kernel_str`. The verifier enforces that all kernel memory accesses must use the safe helpers. Replace direct pointer dereferences with the helper functions.

**Can I use eBPF to trace malloc/free in a C++ application?**

Yes, but only if the application is built with frame pointers and the verifier can unwind the stack. Use a uprobe on the `malloc`/`free` functions and read the return address via `bpf_get_stackid`. Frame pointer support depends on your compiler flags and libc build.

**How do I deploy eBPF programs in Kubernetes without breaking the verifier?**

Run the loader as a privileged init container with `hostPID: true` and add `CAP_BPF`, `CAP_SYS_ADMIN`, and `CAP_NET_ADMIN`. Pin maps to bpffs. Verify that the kernel parameters match what the program expects. Managed Kubernetes distributions may restrict these capabilities, so check your provider's documentation.

**What is the difference between BPF LSM and seccomp?**

BPF LSM hooks run in the LSM framework and can block operations based on internal kernel state, but they do not expose struct arguments directly. Seccomp profiles run in the syscall path and can inspect arguments, but they cannot block based on internal kernel state. Use BPF LSM for simple deny-lists and seccomp for argument-based filtering. In practice, teams deploy both.

## What to do next

In the next 30 minutes, run this command on a test host to measure the actual overhead of a kprobe on your hot path:

```bash
sudo bpftrace -e 'kprobe:tcp_retransmit_skb { @start[tid] = nsecs; } kretprobe:tcp_retransmit_skb /@start[tid]/ { @us = hist((nsecs - @start[tid]) / 1000); delete(@start[tid]); }'
```

Run it for 60 seconds under your normal workload, then compare the histogram against the same histogram collected with the kprobe disabled. If the difference in the p99 bucket is larger than your error budget, do not deploy the kprobe in production. If it is smaller, you have a measured baseline to justify the deployment.
