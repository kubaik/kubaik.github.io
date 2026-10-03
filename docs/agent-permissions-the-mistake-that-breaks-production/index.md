# Agent permissions: the mistake that breaks production

Agents that drive real desktop applications fail in production for a predictable reason: they are designed against a development machine's permissions, not the target machine's. The failure usually surfaces as a silent hang, a dialog nobody reads, or a file that never moves out of an input folder. This article works through the constraint set that causes most of these failures, the automation approaches that respect it, and a concrete design that survives legacy Windows desktops with no administrator rights.

## The constraint set that breaks desktop agents

A recurring production failure mode looks like this. An agent needs to read scanned documents or fill forms in a desktop application. It is developed on a machine where the developer has local administrator rights, a modern runtime, and a stable network. It is deployed to machines where none of those hold. The agent then fails in ways that never appeared during development.

The constraints that matter, in rough order of how often they kill a deployment:

- **No local administrator rights.** IT policy forbids service accounts with full control over the machine, often because a previous incident involved a script damaging shared storage. Any design that assumes elevation is dead on arrival.
- **Session-bound GUI.** On Windows, a process running as a service under a system account does not share the interactive desktop session. Tools that simulate mouse and keyboard input depend on that session. If the user locks the screen, the session is no longer interactive and coordinate-based automation stops working.
- **Legacy OS and applications.** Older Windows versions ship older scripting runtimes, older browser engines, and older document viewers. Modern frameworks may not install or may not run within the available memory.
- **Limited hardware.** Low RAM and slow CPUs rule out heavy runtimes, GPU-dependent models, and large in-memory image preprocessing.
- **Intermittent or absent connectivity.** Cloud OCR, cloud queues, and remote state stores become unavailable for long stretches. Any design that requires a round trip to a remote service must define what happens when that service is unreachable.
- **Shared machines.** The user is doing other work on the same desktop. The agent cannot assume it owns the foreground window, the clipboard, or the file system paths it wrote last hour.

The design principle that follows is simple to state and hard to follow: treat every constraint as a first-class requirement. An agent that cannot run without administrator rights, without internet, and without a modern runtime will not run in the field. This applies well beyond any single program — it is the normal situation for NGO field deployments, government offices, small businesses, and any enterprise where endpoint policy is strict.

## Why GUI simulation fails in these environments

Simulating keystrokes and mouse movement is the first approach most teams reach for, because it works against any application with a visible window. It also has the worst failure profile on a locked-down, shared desktop.

The documented behavior is that input simulation operates on the interactive desktop session. Three consequences follow:

1. **Locked screen means no session.** When the user locks the workstation, the interactive session is no longer accepting synthetic input. An agent that was running fine at 09:00 stops making progress at 09:15 when the user steps away.
2. **Foreground contention.** A user clicking, alt-tabbing, or opening a dialog changes which window has focus. A script that assumes a specific window is focused will type into the wrong place. This is not a rare race; on a shared machine it is the normal condition.
3. **Coordinate fragility.** Screen resolution, window position, DPI scaling, and application chrome all affect where a click lands. A Windows update that changes a dialog layout breaks the script with no code change.

There is a narrower case where input simulation is the only option: an application with no programmatic interface at all. In that case, the agent must run inside the user's interactive session, must verify the expected window is actually focused before every action, and must treat any deviation as a hard stop rather than continuing blindly. Those guards are not optional.

A related trap is OCR-driven extraction. Running optical character recognition over a screenshot is attractive because it needs no application cooperation. But recognition accuracy degrades sharply on rotated, low-resolution, or noisy scans, and image preprocessing to compensate costs memory and CPU that low-spec machines do not have. When accuracy is marginal, the failure is silent: the agent writes a wrong value rather than reporting an error. Any OCR path needs a confidence threshold and a quarantine path for low-confidence results.

## The approach that survives: object-model automation

Windows applications frequently expose a COM automation interface — an object model the operating system can drive directly, without simulating input. A spreadsheet application exposes workbooks, sheets, and cells. A document viewer may expose a document object with text extraction. A browser exposes a document object model.

The properties that matter here:

- **No administrator rights required.** COM activation is a per-user capability. A process running as a standard user can instantiate these objects.
- **Independent of the interactive session.** Object-model calls do not require the desktop to be visible, unlocked, or focused. An agent can drive an application while the user works in another window, and can continue while the screen is locked.
- **No coordinates.** Calls address objects, not pixels. A window moved or resized does not change behavior.

The cost is that COM interfaces are unevenly documented, version-dependent, and sometimes disabled by policy. That last point is important: a group policy can block COM activation of a specific application, and the agent must detect that at startup rather than failing on the first real document.

A practical rule: prefer the object model wherever it exists, fall back to input simulation only for applications that expose nothing, and never mix the two within a single transaction without a clear handoff point.

### A minimal agent skeleton

The following is a small agent that walks an input folder, extracts text from each document through the viewer's object model, writes a result, and moves the file atomically. It is written for a scripting host that is present by default on the target OS; the same structure translates directly to a modern scripting language.

```vbs
Option Explicit

Const INCOMING = "C:\Scans\Incoming"
Const DONE_DIR = "C:\Scans\Done"
Const ERR_DIR  = "C:\Scans\Errors"

Sub Main()
    Dim fso, folder, f
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(INCOMING) Then Exit Sub
    Set folder = fso.GetFolder(INCOMING)
    For Each f In folder.Files
        If LCase(fso.GetExtensionName(f.Name)) = "pdf" Then
            ProcessOne f.Path
        End If
    Next
End Sub

Sub ProcessOne(pdfPath)
    Dim fso, text, dest
    Set fso = CreateObject("Scripting.FileSystemObject")
    On Error Resume Next
    text = ExtractText(pdfPath)
    If Err.Number <> 0 Then
        Log "extract failed: " & pdfPath & " :: " & Err.Description
        Err.Clear
        On Error GoTo 0
        MoveTo pdfPath, ERR_DIR
        Exit Sub
    End If
    On Error GoTo 0

    If Len(Trim(text)) = 0 Then
        Log "empty extraction, quarantining: " & pdfPath
        MoveTo pdfPath, ERR_DIR
        Exit Sub
    End If

    If Not WriteResult(pdfPath, text) Then
        Log "write failed: " & pdfPath
        MoveTo pdfPath, ERR_DIR
        Exit Sub
    End If

    MoveTo pdfPath, DONE_DIR
    Log "ok: " & pdfPath
End Sub

Function ExtractText(pdfPath)
    Dim app, doc
    Set app = CreateObject("AcroExch.App")
    Set doc = CreateObject("AcroExch.AVDoc")
    If Not doc.Open(pdfPath, "") Then
        ExtractText = ""
        Exit Function
    End If
    ExtractText = doc.GetPDDoc.GetText(0)
    doc.Close False
    app.Exit
End Function

Function WriteResult(pdfPath, text)
    Dim fso, ts
    Set fso = CreateObject("Scripting.FileSystemObject")
    On Error Resume Next
    Set ts = fso.CreateTextFile(pdfPath & ".txt", True)
    ts.Write text
    ts.Close
    WriteResult = (Err.Number = 0)
    On Error GoTo 0
End Function

Sub MoveTo(src, destDir)
    Dim fso
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.FolderExists(destDir) Then fso.CreateFolder destDir
    ' Name() on the same volume is an atomic rename.
    fso.MoveFile src, fso.BuildPath(destDir, fso.GetFileName(src))
End Sub

Sub Log(msg)
    Dim fso, ts
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set ts = fso.OpenTextFile("C:\Logs\agent.log", 8, True)
    ts.WriteLine Now() & " " & msg
    ts.Close
End Sub

Main
```

Three details in that listing are load-bearing and worth calling out:

- **`MoveFile` within a volume is a rename**, which the file system performs atomically. A crash mid-move leaves the file in exactly one of the two folders, never in both and never in neither. This is why the agent moves files between states rather than copying and deleting.
- **Empty extraction is treated as failure.** A viewer that opens a document but returns no text is the common case for a corrupt or image-only file. Silently writing an empty result is worse than quarantining, because downstream consumers cannot tell the difference between "no data" and "no text found."
- **Errors are logged with the file path and the error description**, then the file is quarantined. The agent never retries in place, which prevents a poison file from blocking the queue.

## State, heartbeats, and failure detection

An agent that runs unattended needs a way to prove it is alive. The cheapest mechanism is a heartbeat row in a local database, written on a fixed interval.

```sql
CREATE TABLE heartbeat (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    last_seen TEXT NOT NULL,
    status    TEXT NOT NULL
);
```

Writing a single row keyed on a constant id avoids unbounded table growth:

```sql
INSERT INTO heartbeat (id, last_seen, status)
VALUES (1, datetime('now'), 'ok')
ON CONFLICT(id) DO UPDATE SET
    last_seen = excluded.last_seen,
    status    = excluded.status;
```

A separate monitor process reads that row and compares `last_seen` against the current time. If the gap exceeds a threshold, the monitor surfaces a message to the user. The threshold should be several multiples of the write interval, not a single multiple — a brief stall during a large document should not trigger an alarm. If the agent writes every five minutes, a fifteen-minute threshold tolerates two missed writes, which is enough to absorb a slow operation without hiding a genuine hang.

Two refinements matter in practice:

- **Write the heartbeat from the main loop, not a separate thread.** A background timer that keeps ticking while the main loop is stuck produces a false "healthy" signal. The heartbeat should be evidence that work is progressing.
- **Record the current step, not just a timestamp.** A heartbeat row that includes the file being processed and the stage turns a "stuck" alert into a diagnosis.

## Per-user installation

If administrator rights are unavailable, the installer must not require elevation. A per-user installation writes everything under the user's profile directory and registers nothing machine-wide. This is a documented installation mode on Windows and is supported by standard installer tooling.

What a per-user installer should do:

- Place all binaries, scripts, and data under the user's application data directory.
- Register any scheduled task under the current user, not under a system account.
- Set environment variables at the user scope only.
- Avoid writing to protected locations such as the program files directory or the machine registry hive.
- Verify at the end of installation that the agent can start in a standard user session.

The last point is the one teams skip. An installer that reports success but produces an agent that cannot activate its COM objects has not solved the problem. A startup self-check that instantiates each required object and logs the result converts a mysterious field failure into an immediate, actionable error.

For scripted dependencies, a per-user package installation achieves the same effect without an installer. The constraint is identical: nothing may write outside the user's profile.

## A comparison of automation strategies

| Strategy | Needs admin? | Works with screen locked? | Drives legacy desktop apps? | Notes |
|---|---|---|---|---|
| Input simulation | No | No | Yes | Coordinate- and focus-dependent; breaks on layout changes |
| Object-model automation (COM) | No | Yes | Yes | Windows only; interface may be version-specific or policy-blocked |
| Browser automation | Sometimes | Yes | No | Requires the app to be a web app; needs a browser runtime |
| Remote cloud service | No | N/A | Depends | Requires connectivity; may conflict with data-residency policy |
| Local OCR over screenshots | No | No | Yes | Accuracy-sensitive; needs confidence thresholds and quarantine |

The table is a starting filter, not a decision. The right question is which of these can satisfy the constraint list for the specific target machines, and the answer is often a combination: object-model automation for the applications that support it, with a quarantined fallback for the ones that do not.

## Worked example: sizing the daily run

Suppose the requirement is 50 documents per day per machine, 10 fields each, and the extraction path takes 10 seconds per document. That is 500 seconds, or about 8 minutes of agent runtime per machine per day. Across 20 machines, that is 160 minutes of total daily runtime.

Two conclusions follow from that arithmetic, and they are the reason to do it:

- The workload is small relative to a day, so the agent does not need to run continuously. A periodic trigger — hourly, or on a file-arrival check — is sufficient and simpler to reason about.
- The margin is large enough that a retry policy is affordable. If a document fails and is retried three times with backoff, the worst case adds well under an hour of runtime. A design that cannot afford retries is a sign the sizing was never done.

The arithmetic also exposes the real risk: it is not throughput, it is correctness. At 50 documents per day, a 5% silent error rate produces roughly two to three wrong records daily, and those errors are far more expensive than a slow run. This is the argument for quarantining low-confidence extractions rather than writing them.

## Instrumenting instead of guessing

Claims about reliability should come from measurement, and the measurement is straightforward to set up.

- **Instrument per-document timing.** Record a start and end timestamp around each document's processing, and write both to the log. From that log, compute the mean and the 95th percentile with a one-line aggregation. The percentile matters more than the mean: a mean of 12 seconds with a 95th percentile of 22 seconds tells you the slow tail is where the timeouts will occur.
- **Instrument outcomes by category.** Count successes, extraction failures, empty extractions, write failures, and timeouts separately. A single "failure rate" number hides which stage is degrading.
- **Replay a fixed corpus.** Keep a set of known documents with known correct outputs and run the agent against it after every change. This is the only way to detect a regression introduced by an application update or a policy change.
- **Watch the heartbeat gap distribution.** The maximum observed gap between heartbeats is a direct measure of how long the agent can stall before a monitor should fire.

None of this requires a metrics service. A log file and a short aggregation script are sufficient, and they work with no connectivity.

## Failure modes to design against

Each of the following has a specific mitigation, and each is worth a line in the design document.

- **Policy blocks the automation interface.** A group policy can disable COM activation for a specific application. Mitigation: instantiate every required object at startup and fail loudly if any is unavailable.
- **Profile redirection.** The user's application data directory may be redirected to a network share with slow or unreliable writes. Mitigation: write state locally where possible, and treat a slow write as a timeout rather than a hang.
- **Scheduled task does not run.** Task scheduling behavior differs by OS version and by whether a user is logged in. Mitigation: verify task execution empirically on a target machine, and add a second trigger path (for example, a startup-folder script) as a fallback.
- **Poison document.** One malformed file blocks the queue if retried in place. Mitigation: bounded retries with backoff, then quarantine.
- **Partial write.** The process dies after opening an output file but before closing it. Mitigation: write to a temporary name and rename into place, so consumers never observe a partial file.
- **False health signal.** A heartbeat written by a thread that is not doing work. Mitigation: write the heartbeat from the main loop and include the current step.
- **Silent wrong data.** OCR or parsing returns a plausible but incorrect value. Mitigation: confidence thresholds, field-level validation, and a quarantine path for anything that fails validation.

## Decision checklist

Before writing code, answer these for the target machines:

1. What is the exact OS version and patch level?
2. Does the deploying account have local administrator rights? If not, can any component require elevation?
3. Which applications must the agent drive, and does each expose a documented automation interface?
4. Is that interface blocked by policy on the target machines?
5. Does the agent need to run while the screen is locked? If yes, input simulation is eliminated.
6. What is the network availability pattern, and what must work while it is unavailable?
7. What are the RAM, CPU, and disk limits, and does the chosen runtime fit within them?
8. Where does state live, and what happens if that location is redirected or unavailable?
9. How will a stuck agent be detected, and who is alerted?
10. What is the quarantine path for documents that fail, and who reviews them?

If any answer is "unknown," that is the next task — not the next feature.

## Testing in the real environment

Development machines and target machines differ in ways that are invisible until deployment. A short pilot on real hardware, with a real user logged in and doing real work, catches the failures that no virtual machine reproduces. The issues that surface in that setting tend to be environmental: a policy that disables an interface, a profile path that is redirected, a task that does not fire under a particular session state.

The practical sequence is: install as a standard user, run the agent through a full day of normal use including a screen lock, kill the agent mid-document to verify recovery, and confirm that a quarantined document is visible to the person who needs to fix it. Each of those is a five-minute test that prevents a category of production incident.

## The broader lesson

The choice between scripting languages or between automation strategies is secondary. The primary decision is whether the constraints of the target environment are treated as requirements or as obstacles to work around. Teams that design for the constraints ship agents that keep running after the developer has moved on. Teams that assume a flexible environment ship agents that are blocked before they reach the field.

Concretely, that means: prefer the application's object model over simulated input, install per user, keep state local, prove liveness with a heartbeat written by the working loop, move files atomically, quarantine anything that fails validation, and measure the tail rather than the mean.

## Take the next 30 minutes

On a target machine, log in as a standard user with no administrator rights. Create a small script that instantiates the automation object for the application you intend to drive, and run it with the console host:

```cmd
cscript //nologo check.vbs
```

where `check.vbs` contains:

```vbs
On Error Resume Next
Dim o
Set o = CreateObject("Excel.Application")
If Err.Number <> 0 Then
    WScript.Echo "FAIL: " & Err.Description
Else
    WScript.Echo "OK: object model available"
    o.Quit
End If
```

If it prints `OK`, the object model is available to a standard user on that machine and the COM-based design is viable. If it prints `FAIL`, read the error description — a policy block and a missing component produce different messages — and adjust the design before writing any agent logic. Repeat the check for every application the agent must drive.
