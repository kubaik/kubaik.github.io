# Zig vs Rust vs Go: compile errors compared

Compile errors in systems languages are not just annoyances; they encode a language's philosophy about memory, concurrency, and correctness. For a backend service, a CLI tool, or a network daemon, the errors you stare at daily are a practical signal of how much the language will fight you, and where.

This article compares Go, Rust, and Zig through their most common compile-time and runtime failure modes. It explains what each error means, how to fix it, and when that fix is a sign you picked the wrong tool. The goal is not to crown a winner but to map project constraints to the error profile you can live with.

## The same task in three languages

A useful starting point is a function that reads a file into a buffer and returns it. In Go:

```go
func readFile(path string) ([]byte, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    return data, nil
}
```

This compiles. The error handling is explicit but verbose. The same in Rust:

```rust
use std::fs;

fn read_file(path: &str) -> Result<Vec<u8>, std::io::Error> {
    fs::read(path)
}
```

Also compiles. The `?` operator could shorten it further. Now Zig:

```zig
const std = @import("std");

fn readFile(path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(std.heap.page_allocator, 1 << 20);
}
```

This compiles too, but only if the author understands `defer`, allocator choice, and the `!` error union. The confusion begins when errors stop being about syntax and start being about the language's core model.

For Go, the most common confusing error is `declared and not used`. It appears when a variable is declared and never read. Go treats unused local variables as a compile error, not a warning. That is deliberate: it prevents dead code and forces cleanup. It also means that during refactoring, output like this is routine:

```
./main.go:10:6: x declared and not used
```

The fix is trivial — use `_ = x` or remove the declaration — but the error feels punitive to developers used to languages that warn instead.

For Rust, confusion often starts with the borrow checker:

```
error[E0502]: cannot borrow `v` as mutable because it is also borrowed as immutable
```

This is not a syntax error; it is a semantic error about aliasing and mutation. The compiler is reporting that the code violates Rust's ownership rules. The fix requires restructuring data flow, not adding a semicolon.

For Zig, confusion is often about compile-time evaluation and the `comptime` keyword, or about error unions. A typical error:

```
error: expected type 'u32', found 'comptime_int'
```

This happens when an integer literal is passed to a function expecting a specific integer type. Zig's comptime integer literals are arbitrary precision, but they must coerce to the expected type. The fix is to annotate: `const x: u32 = 42;`.

The underlying confusion is the same in all three cases: the error is not about what was typed, but about the language's invariants. Go enforces simplicity and explicitness; Rust enforces memory safety without a garbage collector; Zig enforces manual memory management with compile-time checks. Each error is a boundary of that invariant.

## What is actually causing it

When Go reports `x declared and not used`, the real cause is not that `x` was forgotten — it is that the Go compiler is designed to reject code that might indicate a mistake. Unused variables and imports are compile errors to keep code clean and to avoid the drift that leads to bugs. This is a philosophical choice: Go prioritizes readability and maintainability over flexibility. The error is a feature, not a bug.

In Rust, `cannot borrow as mutable` is not a limitation but a consequence of the borrow checker's goal: guaranteeing memory safety and data-race freedom at compile time. The error means the code has a potential data race or use-after-free. The compiler is preventing code that would be undefined behavior in C or C++. The fix is to redesign data ownership — often by using `RefCell`, `Mutex`, or by splitting data into disjoint borrows.

In Zig, `expected type 'u32', found 'comptime_int'` reveals that Zig's type system is stricter than C's about integer widths. Zig does not implicitly convert between integer types, even when the value fits. This prevents a class of bugs where a value silently truncates. The error is a nudge to be explicit about integer sizes, which matters in systems programming where memory layout and overflow behavior are critical.

The surface symptom in each case is an error message. The real cause is that the language is enforcing a contract the developer may not have internalized yet. The fix is not just to silence the error but to understand which contract is being violated and whether the developer agrees with it.

## Fix 1 — fighting the language's core invariant

The most common cause of these errors is writing code in the style of another language. A Go developer new to Rust might try to share mutable state across threads with `Rc<RefCell<T>>` and hit `Rc` is not `Send`. A Rust developer new to Go might lean heavily on generics and find the syntax clunky. A C developer new to Zig might expect implicit integer conversions and hit `comptime_int` errors.

The fix is to adopt the language's idioms. For Go, that means explicit error returns and fewer unnecessary abstractions. For Rust, it means designing data flow around ownership and borrowing, often using `Arc<Mutex<T>>` for shared mutable state across threads. For Zig, it means being explicit about types and using `@intCast` or `@as` when a conversion is needed.

A concrete Rust example: modifying a vector while iterating over it does not compile.

```rust
let mut v = vec![1, 2, 3];
for x in &v {
    v.push(*x); // error: cannot borrow `v` as mutable because it is also borrowed as immutable
}
```

The fix is to collect the new elements first:

```rust
let mut v = vec![1, 2, 3];
let new_elements: Vec<_> = v.iter().map(|x| *x).collect();
v.extend(new_elements);
```

Or use indices:

```rust
let mut v = vec![1, 2, 3];
for i in 0..v.len() {
    let x = v[i];
    v.push(x);
}
```

This is not a workaround; it is the correct way to express the algorithm in Rust's ownership model. The index version also grows the vector while iterating, so `v.len()` is re-evaluated each iteration — a detail worth noticing, because it changes the loop's behavior compared with the `collect` version.

## Fix 2 — toolchain and build system mismatches

The second most common cause is a mismatch between the toolchain and the environment. In Go, a typical error is:

```
go: cannot find module providing package github.com/foo/bar
```

This usually means the module cache or `GOPATH` is misconfigured, or the machine is offline. The fix is to run `go mod tidy` and check `GOFLAGS`. In Rust:

```
error: failed to run custom build command for `some-crate v0.1.0`
```

This often means a system dependency is missing, such as `libssl-dev` on Debian-based systems. The fix is to install the missing library or enable a vendored feature. In Zig:

```
error: unable to open file: zig-cache/...
```

This can happen if the cache directory is corrupted or on a read-only filesystem. The fix is to clear the cache and rebuild.

These errors are less about language design and more about environment. They are often the first hurdle when setting up a project on a new machine or in CI.

## Fix 3 — cross-compilation and target platforms

Systems programming often involves cross-compilation. Go makes this relatively easy with `GOOS` and `GOARCH` environment variables, but a common error is:

```
build constraints exclude all Go files in ...
```

This means the files in the directory are not compatible with the target OS/arch. The fix is to add build tags or correct `//go:build` directives.

Rust uses target triples. A common error is:

```
error: linking with `cc` failed: exit status: 1
```

This often means a cross-compilation toolchain is missing, such as `gcc-aarch64-linux-gnu`. The fix is to install the toolchain and set the linker in `.cargo/config.toml`.

Zig is designed for cross-compilation and bundles its own C compiler, but this can still appear:

```
error: unable to find dynamic linker for target
```

This means the target triple is not fully supported or the linker is missing. The fix is to use a supported target or provide the linker path.

Cross-compilation errors are particularly painful because they often surface only in CI or when deploying to a different architecture. The fix is to codify the build environment — a container image, a Nix flake, or a CI matrix that tests all target platforms.

## How to verify a fix worked

After applying a fix, reproduce the original error and confirm it is gone. For Go, run `go build ./...`. For Rust, run `cargo check` (faster than `cargo build`) and then `cargo test`. For Zig, run `zig build` or `zig build test`.

Compilation success is not enough. Run the test suite, preferably with sanitizers. In Rust, `cargo test` with `RUSTFLAGS="-Z sanitizer=address"` on nightly can catch memory issues. In Go, `go test -race` enables the race detector. In Zig, `std.testing.allocator` detects leaks in tests.

For cross-compilation, verify by running the binary on the target platform or in an emulator. If that is not possible, at least check the binary format with `file` and confirm it matches the target architecture.

## How to measure your own error profile

Claims about which language produces "better" errors are only useful when tied to a project. Here is how to measure it on a real codebase rather than trusting a blog post.

1. Pick a representative task, such as adding an endpoint or a CLI subcommand.
2. Record the toolchain versions: `go version`, `rustc --version`, `zig version`.
3. Count compile invocations with `go build ./... 2>&1 | wc -l`, `cargo check --message-format=short 2>&1 | wc -l`, and `zig build 2>&1 | wc -l`.
4. Time a clean build and an incremental build with `time`.
5. Classify each error by category: type/ownership, unused symbol, dependency/toolchain, or build constraint.
6. Repeat on the same task in each language with the same developer.

The output is a per-project profile, not a universal ranking. A team whose work is mostly I/O-bound network services will see a very different profile from one writing a memory allocator.

## Prevention

Prevention starts with understanding the language's invariants and encoding them in the workflow. For Go, enable `go vet` and `staticcheck` in the editor and CI. For Rust, use `clippy` and `rustfmt`. For Zig, use `zig fmt` and consider `zig ast-check`.

Write tests that exercise the boundaries: integer overflow, null pointers, concurrent access, and cross-platform behavior. In Go, table-driven tests with `t.Parallel()` can surface race conditions. In Rust, property-based testing explores edge cases that hand-written tests miss. In Zig, `comptime` tests can verify compile-time invariants.

Document the build environment. A `Dockerfile` or `flake.nix` that pins toolchain versions and system dependencies saves hours of debugging when a new team member joins or when CI breaks.

## Related errors you may hit next

Once the immediate error is resolved, these related issues tend to appear:

- **Go:** `panic: runtime error: index out of range` — often caused by off-by-one errors in slice indexing. Fix with bounds checks or `for i, v := range slice`.
- **Go:** `fatal error: all goroutines are asleep - deadlock!` — indicates a deadlock in channel operations. Use `go vet` or the race detector.
- **Rust:** `error[E0382]: use of moved value` — the value is used after being moved. Fix by cloning or using references.
- **Rust:** `error[E0277]: the trait bound `T: Send` is not satisfied` — a non-`Send` type is being sent across threads. Wrap in `Arc<Mutex<T>>` or use `std::sync::mpsc`.
- **Zig:** `error: integer overflow` — Zig checks integer overflow in safe modes. Use `@addWithOverflow` or `+%` for wrapping arithmetic.
- **Zig:** `error: unreachable code` — the compiler detected code that can never execute. Remove it or adjust the logic.

These errors are not random; they are consequences of the same invariants. Recognizing them quickly is a sign the language is being internalized.

## When the fixes do not work

If the fixes above do not resolve the error, isolate the problem. Create a minimal reproducible example in a new project. For Go, use `go mod init` and copy the failing code. For Rust, use `cargo new` and add dependencies one at a time. For Zig, use `zig init-exe` and strip the code down.

If the minimal example still fails, search for the exact error message in the language's issue tracker or forum. When asking for help, include the full error message, the toolchain version (`go version`, `rustc --version`, `zig version`), the target platform, and the minimal example. This gets a faster response.

If the error is due to a bug in the language or a library, consider filing an issue. If it is a limitation, a workaround or a different tool may be needed. That is not failure; it is engineering.

## Frequently Asked Questions

**Why does Go complain about unused variables but not unused functions?**
Unused local variables and imports are compile errors to prevent dead code and accidental shadowing. Unused functions are allowed because they might be part of an exported API or used via reflection. The design prioritizes local clarity over global completeness.

**How do I fix `cannot borrow as mutable` in Rust when I need to modify a collection while iterating?**
A collection cannot be mutated while an immutable borrow is live. Collect the changes into a temporary vector and apply them after the loop, or use indices to access elements. Alternatively, use `RefCell` for interior mutability in single-threaded code, or `Mutex` for multi-threaded code.

**What does `comptime_int` mean in Zig and how do I resolve the error?**
`comptime_int` is an arbitrary-precision integer known at compile time. When passed to a function expecting a fixed-width integer such as `u32`, Zig requires an explicit cast. Use `@as(u32, 42)` or annotate the variable: `const x: u32 = 42;`.

**When should I choose Zig over Rust or Go for a new project?**
Choose Zig when precise control over memory layout matters, hidden allocations are unacceptable, and manual memory management is comfortable. Zig excels at cross-compilation and C interoperability. Rust is better when memory safety without a garbage collector and a mature ecosystem are priorities. Go is best for rapid development of networked services where simplicity and fast compile times matter more than zero-cost abstractions.

## Closing action

Open your project's `go.mod`, `Cargo.toml`, or `build.zig` and check the toolchain version. Run `go version`, `rustc --version`, or `zig version` and compare it to the version CI uses. A mismatch there is a common cause of "works on my machine" build failures, and it takes 30 seconds to rule out.
