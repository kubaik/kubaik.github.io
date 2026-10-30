# WASM servers beat containers here

## The one-paragraph version

WebAssembly is not a universal speed-up. It is a targeted optimization for one specific case: moving a small slice of CPU-bound work into a guest module that a host process can call without forking a process or starting a container. If that is not what you are doing, keep your containers. The rest of this article explains where the boundary sits, how to measure it, and the failure modes that make teams regret the move.

## What "WebAssembly on the server" actually means

The phrase invites a wrong mental model. Many developers picture a browser-like environment where WASM runs in a sandboxed guest and the host is JavaScript. That model produces two incorrect conclusions: that WASM is browser-only, and that it is inevitably slower because every call crosses a boundary.

On the server, the host is any process that can instantiate a WASM module — a Node.js service, a Rust binary embedding a runtime, an edge worker platform, or a managed function runtime. The guest is a compiled module with no ambient authority: it can only touch memory, call the imports the host provides, and return values. That is the whole contract.

The boundary crossing is real but small. For primitive types (i32, i64, f32, f64), marshaling is a register write plus a call. For strings and byte buffers, the host must copy into the guest's linear memory, which is where the measurable cost lives. The size of that cost is what determines whether WASM helps or hurts, and it is measurable in a few minutes with a microbenchmark — see the measurement section below.

The "containers vs. WASM" framing is also a distraction. Containers remain the right unit of deployment for long-lived services that need a full OS image, package managers, and arbitrary system calls. WASM fits when you want to run a small, pure function without process forking or container startup. The overlap between the two is narrow but real.

## The mental model

A container is a shipping container: it holds everything needed to run a service, and you pay a fixed cost to open and unpack it even when you move something small. A WASM module is a courier envelope: you pay per byte of payload and per unit of compute, and you skip the unpacking. The envelope only carries small, CPU-bound payloads well.

The envelope fits when all of these hold:

- The workload is short-lived — on the order of single-digit milliseconds of CPU.
- The workload is CPU-bound: hashing, signature verification, validation, compression, regex matching, parsing.
- The guest needs no network or filesystem access beyond what the host explicitly provides.
- You are willing to compile the logic once and load it into host processes.

If any condition fails, a container is usually the better answer. The most common failure is the second: teams move I/O-bound code into a guest and discover the boundary copy costs more than the work saved.

## A worked example: JWT validation in a gateway

Consider an API gateway that validates an incoming JWT against a public-key list refreshed periodically. Validation is pure CPU work — an RSA-PSS verify — and the service has a latency budget it must not blow. This is the shape of workload where WASM is worth evaluating.

**Baseline: Node.js container**

- Image: Node 20 LTS on a slim base, roughly 64 MB compressed.
- Cold start on a serverless platform: typically hundreds of milliseconds, dominated by image pull and runtime init.
- Median latency for a signed JWT verify: a few milliseconds, with p95 several times higher.

**WASM variant**

- Compile: Rust to `wasm32-unknown-unknown`, then optimize with `wasm-opt -O2`.
- Guest size: tens of kilobytes for a single-purpose validator.
- Host: the same Node.js process, loading the module at startup.
- Cold start: module instantiation, typically low single-digit milliseconds.
- Median latency: lower than the container path, with the gap dominated by the verify itself.

The numbers above are illustrative, not measured here. The point is the shape: the win comes from removing process and image startup, not from the guest CPU being magically faster. To decide for your system, measure it — the next section shows how.

The failure mode that bites teams is state. The container kept the public-key list in process memory and reloaded it on a schedule. The WASM guest has no persistent store, so the host must push the current key list into the guest on each call or hold it in a host-owned buffer the guest reads. Design the host-guest data flow before you write the guest.

## Measuring whether WASM helps

Do not trust anyone's latency table, including the one above. Build a microbenchmark that isolates the boundary.

**What to instrument**

- Time to instantiate the module once at startup: `performance.now()` around `WebAssembly.instantiate`.
- Per-call time for a no-op export: this is your boundary floor.
- Per-call time for the real work with a representative payload: this is your candidate.
- Per-call time for the same work in the host language: this is your baseline.

**What to run**

A loop of at least 10,000 calls per variant, discarding the first 1,000 as warmup, and reporting median and p95 rather than mean. Run it on the same machine, same CPU governor, same Node version. If you deploy on a serverless platform, run it there too — cold start is a different measurement from steady-state call cost.

**What to compare**

- `wasm_call - noop_call` is the cost of the work itself.
- `wasm_call - host_call` is the win or loss versus the host implementation.
- `instantiate_time` is the cold-start cost you pay per process.

If `wasm_call` is not clearly below `host_call`, and `instantiate_time` is not clearly below your container start time, stop. You have your answer.

## Host and guest code

The snippets below show the shape of a host-guest integration. They use the standard `WebAssembly` API and a WASI shim; adapt the imports to whichever runtime you use.

Host side (Node.js):

```javascript
import { readFile } from 'node:fs/promises'

const wasmBuffer = await readFile('./jwt-validator.wasm')
const wasmModule = await WebAssembly.compile(wasmBuffer)

// The host owns the key list and the scratch buffers.
// The guest only sees what the host copies into its linear memory.
const instance = await WebAssembly.instantiate(wasmModule, {
  env: {
    host_log: (ptr, len) => {
      const view = new Uint8Array(instance.exports.memory.buffer, ptr, len)
      console.log(new TextDecoder().decode(view))
    },
  },
})

const { memory, alloc, validate_jwt } = instance.exports

function validate(jwtBytes, keyBytes) {
  const jwtPtr = alloc(jwtBytes.length)
  const keyPtr = alloc(keyBytes.length)
  new Uint8Array(memory.buffer, jwtPtr, jwtBytes.length).set(jwtBytes)
  new Uint8Array(memory.buffer, keyPtr, keyBytes.length).set(keyBytes)
  return validate_jwt(jwtPtr, jwtBytes.length, keyPtr, keyBytes.length)
}
```

Guest side (Rust):

```rust
static mut HEAP: [u8; 65536] = [0; 65536];
static mut HEAP_TOP: usize = 0;

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    unsafe {
        let ptr = HEAP.as_mut_ptr().add(HEAP_TOP);
        HEAP_TOP += len;
        ptr
    }
}

#[no_mangle]
pub extern "C" fn validate_jwt(
    jwt_ptr: *const u8,
    jwt_len: usize,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    let jwt = unsafe { std::slice::from_raw_parts(jwt_ptr, jwt_len) };
    let key = unsafe { std::slice::from_raw_parts(key_ptr, key_len) };

    // Pure CPU work: parse the key, decode the signature, verify.
    // Return 0 on success, non-zero on failure.
    match verify(jwt, key) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}
```

Two things to notice. First, the host owns the key list and copies it in per call; there is no hidden shared state. Second, the guest exports an `alloc` so the host can place bytes in linear memory. Both are deliberate — they keep the trust boundary explicit.

## Failure modes to plan for

**Boundary copy dominates.** If your payload is large (say, a multi-megabyte document), the copy into linear memory can exceed the compute time. Measure `noop_call` versus `real_call`; if the difference is small relative to the copy, WASM is the wrong tool.

**State lives in the wrong place.** Guests have no persistent store. Any key list, config, or cache must be owned by the host and passed in. Teams that forget this end up re-fetching config per call or, worse, baking it into the module and redeploying to rotate.

**Secrets cross the boundary.** The sandbox prevents the guest from reading host memory, but anything the host passes in is fully visible to the guest. If the guest is untrusted code, treat every argument as public and tokenize or encrypt accordingly.

**ABI mismatch.** WASM is portable across architectures but not across ABIs. A module compiled from Rust expects a specific set of imports; a module from another toolchain expects a different set. Mixing languages inside one module is a source of confusing instantiation errors. Keep one language per module until the interface story settles.

**Traps look like crashes.** A guest trap surfaces in the host as an exception or an exit code. Catch it at the host boundary and log the guest's linear memory to diagnose. Without that, a trap is an opaque failure.

## How this connects to things you already know

If you have dealt with serverless cold starts, you already understand the trade-off: WASM does not eliminate startup cost, it shrinks it. Treat the module as a shared library loaded once per process, not as a process per request.

If you have used gRPC or Protocol Buffers, you already understand marshaling. Moving primitives across a boundary is cheap; moving strings and buffers costs a copy. WASM's boundary is in the same family of costs.

If you have used multi-stage Docker builds, you already understand the image-size-versus-startup-time trade-off. A small WASM guest is analogous to a micro-container, with a much lower startup cost.

One place the analogy breaks is networking. Guests do not get raw sockets. They use whatever the host exposes. If you need raw TCP or UDP, keep that logic in the host and push only the CPU-bound slice into the guest.

## Misconceptions, corrected

**"WASM is always slower than native."** For CPU-bound, short-lived work, a compiled guest can outperform an interpreted host language because the host does not re-optimize the guest code per call. Whether it beats a native implementation is a measurement question, not a rule.

**"You need a browser or a dedicated runtime."** Node.js, Deno, and Bun all ship WASI support in current releases. You can load a module with the standard `WebAssembly` API and call its exports without a separate runtime process.

**"The sandbox means secrets are safe."** The sandbox protects host memory from the guest. It does not protect secrets the host chooses to pass in. Treat guests as untrusted code at the boundary.

**"Modules are portable across languages."** They are portable across CPU architectures. They are not portable across ABIs. Match the host imports to the toolchain that produced the module.

## Advanced patterns

**Reuse across workers.** Load and compile the module once in the main thread, then share the compiled `WebAssembly.Module` with worker threads. Compilation is the expensive part; instantiation per worker is cheap, and the host can keep the key list in shared memory.

**Ahead-of-time compilation.** Some runtimes support compiling a WASM module to a native artifact ahead of time, so process restarts skip the parse and compile step. The exact flags and artifact format depend on the runtime; check its documentation. The win is a lower cold start, at the cost of a build step and a per-architecture artifact.

**Keep the guest pure.** The most maintainable guests export a small number of functions that take pointers and lengths and return integers. Everything else — logging, networking, config — lives in the host. This keeps the boundary auditable and the guest testable in isolation.

## Decision checklist

| Scenario | Container | WASM guest | Reason |
|---|---|---|---|
| Long-running API server | Yes | No | Needs a full OS image and persistent state |
| Short-lived CPU work (single-digit ms) | No | Yes | Avoids process and image startup |
| Large payloads (multi-MB) | Yes | No | Boundary copy dominates |
| Raw socket networking | Yes | No | Guests have no raw sockets |
| Untrusted code isolation | Either | Yes | Guest sandbox is the point |
| Per-request state | Yes | No | Guests cannot persist state |

## FAQ

**How do I debug a WASM module in Node.js?** Compile with debug info, enable source maps, and attach a debugger that understands WASM. The guest's linear memory is visible as a typed array, so you can inspect strings and structs directly.

**Can I compile Go or Python to WASM for server use?** Both are possible with the right toolchain, but the runtime overhead is larger than Rust or C++. For CPU-bound work, prefer a language with a small runtime footprint, and verify the ABI your host expects.

**What happens when the guest traps?** The host sees an exception or an exit code. Catch it at the boundary, log the linear memory, and return a clear error to the caller.

**Is WASM cheaper at scale?** Only for the workloads described here. If you push high request volumes through a small guest, you save on startup and per-request CPU. If you run long-lived services, the savings disappear because you still need a process to host the module.

## One thing to do in the next 30 minutes

Pick your highest-latency CPU-bound endpoint — hashing, signature verification, compression, or parsing. Write a microbenchmark that calls the current implementation 10,000 times and records median and p95. Then compile the same logic to WASM, load it in a Node.js process, and run the identical benchmark. Compare `wasm_call - noop_call` against `host_call`, and compare module instantiation time against your container start time. If WASM does not clearly win on both, keep the container.
