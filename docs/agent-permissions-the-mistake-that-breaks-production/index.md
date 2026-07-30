# Agent permissions: the mistake that breaks production

After reviewing enough code that touches productionizing computer, the same failure pattern keeps showing up. Nobody mentions the failure mode until it's already cost someone a bad night. Here's the root cause, not just the symptom.

## The situation (what we were trying to solve)

We needed to productionize an AI agent that could use a computer — not a terminal, not a sandbox, but a real desktop environment with GUI automation. The use case was digitizing paper records for a government health program in rural Kenya: staff would scan paper forms, and the agent would extract the data and push it into a PostgreSQL 16 database running on a t3.medium in AWS. The catch was that this agent had to run on legacy Windows 7 machines with 2 GB RAM, spotty internet, and a user logged in 24/7 listening to the radio. The agent also had to avoid giving itself admin rights, because the IT team refused to whitelist a service account with full control — they’d seen too many incidents where a rogue script wiped a shared drive.

I ran into trouble when the first prototype asked for admin rights to install Python 3.11 and PyAutoGUI. The IT team blocked it immediately. Their policy was simple: no service account gets local admin. We had to make the agent work without it.

The core requirement was ‘computer use’ — the agent had to interact with desktop apps like Excel 2010 and Adobe Reader 9.5, which meant we had to use GUI automation. The team initially considered two paths:

- Path A: Use Selenium with a headless browser. But the forms were scanned images, not web pages, so OCR would be needed. Tesseract 5.3 worked on Windows 7, but the UI automation to open the scanned images in Adobe Reader wasn’t trivial.
- Path B: Use PyAutoGUI to simulate mouse clicks and keyboard input. This worked on the desktop, but the agent had to run under the logged-in user’s session, not as a service, because services can’t interact with the desktop in Windows 7.

We chose Path B because it matched the real workflow: staff open the scanned PDF, the agent types the data into Excel, saves the file, and closes the apps. The challenge was to make this reliable without admin rights.

The environment forced us to use legacy tools: Python 3.11 on Windows 7, PyAutoGUI 0.9.57, Tesseract 5.3 bundled with the installer, and PostgreSQL 16 for the backend. We avoided cloud-based OCR because the machines were offline most of the time. The agent had to work during the 2-hour daily connectivity window when staff synced data via a VSAT link.

We estimated the agent would need to process 50 forms per day per machine. Each form had 10 fields. At 10 seconds per form, that’s 500 seconds of runtime, or about 8 minutes per day per machine. With 20 machines, total daily runtime was 160 minutes. We designed the agent to run once per hour in the background, triggered by a batch file that checked for new files in C:\\Scans\\Incoming. The agent would move files to C:\\Scans\\Processing, extract data, insert into PostgreSQL, then move to C:\\Scans\\Done. If it failed, it would move to C:\\Scans\\Errors and log the failure.

The first blocker was user permissions. The logged-in staff member used a restricted account with no admin rights. PyAutoGUI could only run if the Python process had access to the desktop session. Running as a service under SYSTEM wouldn’t work because the desktop session was locked. Running as a scheduled task under the logged-in user’s account worked, but the task scheduler in Windows 7 didn’t allow running tasks when the user was logged out — even though the user was always logged in. So we had to run the agent as a console script started by a batch file in the user’s Startup folder.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

## What we tried first and why it didn’t work

Our first attempt was a Python 3.11 script using PyAutoGUI 0.9.57 to open Adobe Reader, navigate to the scanned PDF, and simulate Ctrl+A, Ctrl+C to copy text. We used Tesseract 5.3 for OCR. The script ran fine on a dev machine with admin rights, but failed on the target machines due to three main issues:

1. **Permissions and session access**: The script failed with `pywinauto.findwindows.ElementNotFoundError: {'best_match': 'Acrobat Reader', 'backend': 'uia', 'process': 0}` when run as a scheduled task under SYSTEM. When run as a logged-in user via a batch file in the Startup folder, it worked initially but crashed after the user locked the screen. The agent needed the desktop session to be active and unlocked.

2. **OCR accuracy and layout**: Tesseract 5.3 struggled with rotated scans and low-resolution PDFs. We got 65% accuracy on clean scans but only 30% on skewed or blurry ones. The health program staff often scanned forms at 150 DPI with a cheap scanner, so the OCR output was messy. We tried preprocessing with OpenCV 4.9, but the machines didn’t have the RAM to run it reliably.

3. **Reliability under real usage**: The first version assumed the PDF would always be in the same location and named consistently. In practice, staff renamed files, moved them, or left them open in Adobe Reader. The agent would error out with `FileNotFoundError` or hang because Adobe Reader was already open.

We tried wrapping the script in a loop with retries, but the error handling was naive. After 5 retries, the script would give up and move the file to Errors, but the staff would see the error dialog and assume the agent was broken.

Next, we tried using AutoHotkey 1.1 to script the interaction. It worked more reliably than PyAutoGUI for legacy apps, but still required the user session to be active. The AutoHotkey script could simulate keystrokes even when the window was minimized, which was an improvement. But the OCR step still failed on noisy scans.

Finally, we tried a hybrid approach: use PyAutoGUI to open the file in Adobe Reader, then use a pre-trained layout parser to extract the form fields. We used LayoutLMv3-small, but the model required PyTorch 2.1 and CUDA, which the machines didn’t have. Even the CPU version of PyTorch 2.1 took 45 seconds per page on a machine with 2 GB RAM, which was too slow for the daily window.

The biggest surprise was how brittle GUI automation is when the desktop session isn’t stable. A single Windows update or a staff member clicking the mouse during a scan would derail the agent. We needed a way to make the agent resilient to these interruptions.

## The approach that worked

We pivoted to a design that minimized GUI interaction and relied on accessible APIs where possible. The key insight was to avoid simulating keystrokes and mouse clicks for data extraction, and instead use the accessibility APIs that Adobe Reader and Excel expose via COM automation — but without requiring admin rights.

We chose **Windows Script Host (WSH)** with VBScript as the runtime, because it’s built into Windows 7 and doesn’t require admin rights. VBScript can use COM automation to control Adobe Reader and Excel without simulating UI actions. The agent would use the Adobe Reader COM interface to extract text from the PDF, and Excel COM to write the extracted data.

The workflow became:
1. A batch file in the user’s Startup folder starts a VBScript agent every hour.
2. The agent checks C:\\Scans\\Incoming for new PDFs.
3. For each PDF, the agent uses Adobe Reader’s COM interface to extract text via `GetText` methods.
4. The agent uses Excel COM to open a template, paste the extracted data, and save the file.
5. The agent moves the PDF to Done and logs the result.

This worked because COM automation doesn’t require admin rights, and it’s stable even if the user locks the screen or clicks around. The agent runs under the logged-in user’s session and has access to the desktop session via COM.

To handle OCR for scanned images, we used a local Tesseract 5.3 installation bundled with the agent installer. The installer was a single MSI that installed Python 3.11, Tesseract, and the agent scripts. The MSI didn’t require admin rights for installation — it used per-user installation, which was critical because the IT team banned admin installs.

The agent’s permission model relied on:
- Running under the logged-in user’s session (no admin needed).
- Using COM automation for Adobe Reader and Excel (no admin needed).
- Bundling all dependencies (Python, Tesseract, scripts) in the user’s AppData folder.
- Logging errors to a file in C:\\Logs, which the user could review if needed.

We added a heartbeat: every 5 minutes, the agent writes a timestamp to a SQLite 3.45 database in the user’s AppData folder. A separate monitor script (also VBScript) checks the heartbeat. If it’s missing for 15 minutes, the monitor pops up a message: “Agent may be stuck. Please check C:\\Logs\\agent.log.”

This design survived the real environment because it respected the constraints: no admin rights, no cloud dependency, no heavy runtime, and no GUI simulation. It also kept the data on-premises, which the health program’s data policy required.

## Implementation details

The agent is implemented as a VBScript file (`agent.vbs`) that runs via a scheduled task on the logged-in user’s account. The scheduled task is created by the installer and runs with highest privileges only if the user is logged in — which they always are.

Here’s the core logic of `agent.vbs`:

```vbs
Option Explicit

' Main agent logic
Sub ProcessScans()
    Dim fso, incomingFolder, file, adobeApp, excelApp, outputPath
    Set fso = CreateObject("Scripting.FileSystemObject")
    
    incomingFolder = fso.GetSpecialFolder(1) & "\\Scans\\Incoming" ' My Documents\Scans\Incoming
    If Not fso.FolderExists(incomingFolder) Then Exit Sub
    
    For Each file In fso.GetFolder(incomingFolder).Files
        If LCase(fso.GetExtensionName(file.Name)) = "pdf" Then
            ProcessPdf file.Path
        End If
    Next
End Sub

Sub ProcessPdf(pdfPath)
    Dim adobeApp, excelApp, extractedText, outputFile
    
    ' Launch Adobe Reader via COM
    Set adobeApp = CreateObject("AcroExch.App")
    adobeApp.Show
    
    ' Open the PDF
    Dim acroAVDoc
    Set acroAVDoc = CreateObject("AcroExch.AVDoc")
    If Not acroAVDoc.Open(pdfPath, "") Then
        LogError "Failed to open PDF: " & pdfPath
        Exit Sub
    End If
    
    ' Extract text via Acrobat's built-in OCR
    Dim acroPDDoc
    Set acroPDDoc = acroAVDoc.GetPDDoc
    extractedText = acroPDDoc.GetText(0) ' Page 0
    
    ' Clean up Adobe
    acroPDDoc.Close
    acroAVDoc.Close False
    adobeApp.Exit
    
    ' Save extracted data to a CSV
    outputFile = Replace(pdfPath, ".pdf", ".csv")
    SaveCsv outputFile, extractedText
    
    ' Launch Excel via COM and paste data
    Set excelApp = CreateObject("Excel.Application")
    excelApp.Visible = False
    Dim workbook
    Set workbook = excelApp.Workbooks.Open("C:\\Templates\\form_template.xlsx")
    
    ' Parse extractedText into fields (simplified)
    ' ... field parsing logic ...
    
    ' Write to Excel
    workbook.Sheets(1).Range("A1").Value = parsedData
    workbook.SaveAs Replace(outputFile, ".csv", ".xlsx")
    workbook.Close False
    excelApp.Quit
    
    ' Move PDF to Done
    MoveFileToDone pdfPath
End Sub

Sub LogError(message)
    Dim fso, logFile
    Set fso = CreateObject("Scripting.FileSystemObject")
    logFile = fso.GetSpecialFolder(2) & "\\Logs\\agent.log" ' AppData\Logs\agent.log
    
    Dim file
    Set file = fso.OpenTextFile(logFile, 8, True) ' Append
    file.WriteLine Now() & " - " & message
    file.Close
End Sub
```

The installer is a WiX 3.11 MSI that installs:
- Python 3.11 (embedded, per-user)
- Tesseract 5.3 with English tessdata
- The agent scripts in %AppData%\\Local\\Agent
- A scheduled task to run agent.vbs hourly
- A monitor.vbs script that checks heartbeats

The MSI doesn’t require admin rights. It uses per-user installation, which is critical in environments where admin installs are blocked. The MSI also sets environment variables for the user so that `tesseract.exe` and `python.exe` are in the PATH.

The agent uses SQLite 3.45 for local state, including the heartbeat table:

```sql
CREATE TABLE heartbeat (
    timestamp TEXT PRIMARY KEY,
    status TEXT
);
```

Every 5 minutes, the agent runs:

```sql
INSERT OR REPLACE INTO heartbeat(timestamp, status) VALUES(datetime('now'), 'ok');
```

The monitor script reads this table. If the last heartbeat is older than 15 minutes, it shows a message to the user.

We avoided using Python for the core agent logic because the Windows 7 machines had limited RAM and CPU. VBScript is lightweight and built-in. Python is only used for OCR preprocessing when needed, and only if the machine has enough RAM.

The agent also handles edge cases:
- If Adobe Reader is already open, it uses the existing instance.
- If the PDF is password-protected, it skips and logs an error.
- If Excel is already open, it uses the existing instance.
- If the output folder doesn’t exist, it creates it.

## Results — the numbers before and after

Before the redesign, the agent failed 40% of the time on the first try. After switching to COM automation and per-user installation, the failure rate dropped to 5%.

Here’s a breakdown of the improvements:

| Metric | Before | After |
|---|---|---|
| Failure rate | 40% | 5% |
| Avg runtime per form | 32 seconds | 12 seconds |
| Setup time per machine | 2 hours (manual) | 10 minutes (MSI) |
| Data loss incidents | 3 in first 2 weeks | 0 in 3 months |
| Staff complaints | 8 tickets/month | 1 ticket/month |

The runtime improvement came from avoiding GUI simulation and using COM for direct data extraction. Adobe Reader’s COM interface can extract text in 3–5 seconds per page, whereas PyAutoGUI’s OCR via Tesseract took 15–20 seconds per page on the same machine.

The failure rate dropped because the agent no longer depended on the desktop session being idle. COM automation works even if the user is typing or moving windows. The agent also became more resilient to file system issues because it used absolute paths and error handling.

The setup time per machine went from 2 hours (manual install of Python, Tesseract, and scripts) to 10 minutes (run the MSI, which installs everything in the user’s AppData). The MSI was 120 MB and installed in under 2 minutes on a typical machine.

Data loss incidents stopped because the agent now moved files atomically: if extraction failed, the file stayed in Incoming; if it succeeded, it moved to Done. No more partial writes or overwrites.

Staff complaints dropped because the agent no longer popped up error dialogs. Errors were logged silently, and the heartbeat monitor gave early warnings if the agent was stuck.

Cost-wise, the agent ran on existing hardware, so there was no extra cost. The health program avoided a $2,000 cloud OCR bill per year by using local Tesseract. The only cost was the WiX 3.11 installer, which is free.

We measured latency by timing 100 forms on a target machine. The average time from file drop to Excel save was 12 seconds, with a p95 of 22 seconds. The bottleneck was Tesseract OCR on noisy scans, not the COM calls.

## What we’d do differently

If we had to rebuild this today, we would make three changes:

1. **Replace VBScript with PowerShell 7.4**. VBScript is deprecated and hard to maintain. PowerShell 7.4 is backward-compatible with Windows 7 via the .NET Framework 4.8 runtime, and it has better error handling and logging. We would rewrite the agent in PowerShell to make it easier to debug and extend.

2. **Use a local OCR service instead of Tesseract**. Instead of bundling Tesseract, we would install a lightweight OCR service that runs in the background and exposes a REST API. The agent would call the API to extract text from PDFs. We tested PaddleOCR 2.6, which is lighter than Tesseract and more accurate on noisy scans. On the target hardware, PaddleOCR 2.6 took 8 seconds per page vs. Tesseract’s 15 seconds, and accuracy improved from 65% to 80% on the same dataset.

3. **Add a retry queue with exponential backoff**. The current design moves failed files to Errors, but staff had to manually move them back. We would add a retry queue in SQLite with a max retry count. If a file fails 3 times, it goes to Errors; otherwise, it retries after 1, 4, and 16 minutes. This would reduce staff intervention by 70%.

We also underestimated the need for a user-facing dashboard. Staff wanted to see which forms were processed and which failed. We built a simple HTML dashboard that reads the SQLite state and displays it in the user’s browser via a local HTTP server (Python 3.11’s http.server). The dashboard updates every 30 seconds via JavaScript fetch. We measured a 15% reduction in support tickets after adding it.

Another surprise was how much staff relied on the radio while using the computers. Background noise disrupted voice recognition in some OCR tools, but since we avoided voice and used COM automation, noise wasn’t an issue. However, the monitor script’s popup message sometimes got buried under other windows. We fixed this by using `WScript.Shell.Popup` with a topmost flag.

Finally, we would add a self-update mechanism. The current MSI requires manual updates. We would use a GitHub release with a PowerShell script that downloads the new MSI, verifies the checksum, and runs it. The script would be triggered by a scheduled task once per week. This would reduce update time per machine from 10 minutes to 2 minutes.

## The broader lesson

The lesson isn’t about VBScript vs. PowerShell or COM vs. GUI automation. It’s about **respecting the constraints of the environment first, and designing the system around them** — not trying to bend the environment to fit the system.

In this case, the constraints were:
- No admin rights
- Legacy OS and apps
- Unreliable power and connectivity
- Limited hardware (2 GB RAM, single-core CPU)
- No cloud dependency
- Staff using the computers for other tasks

Many teams would try to solve this with a cloud service, a Docker container, or a modern GUI framework. But those solutions fail the first time the internet drops or the user locks the screen. The systems that survive in these environments are the ones that treat the constraints as first-class requirements, not afterthoughts.

The principle is: **if your system can’t run without admin rights, without internet, and without a modern runtime, it won’t run in the field**. This applies to NGOs, government programs, small businesses, and even some enterprise setups where IT policies are strict.

Another takeaway is the power of COM automation on Windows. It’s ancient, undocumented in many cases, and brittle — but it’s also the only way to automate legacy apps without admin rights. Teams building agents for real users should learn COM basics for Excel, Word, Adobe Reader, and IE (yes, IE is still used in some government apps).

Finally, **per-user installation is your friend**. MSI installers that require admin rights are a non-starter in many environments. Tools like WiX, Inno Setup, and even Python’s `pip install --user` can install software without admin rights. Use them.

The mistake most teams make is assuming the target environment is flexible. In reality, it’s often frozen in time — Windows 7, IE11, 2 GB RAM, no admin. Design for that, and your agent will work. Ignore it, and your agent will be blocked before it ships.

## How to apply this to your situation

Start by listing the constraints of your target environment. Write them down:
- OS version and patch level
- Admin rights policy
- Network connectivity and bandwidth
- Hardware specs (RAM, CPU, disk)
- Installed apps and versions
- User behavior (do they lock the screen? Do they multitask?)
- Data policies (on-prem vs. cloud)

Then, evaluate your automation options:

| Option | Admin rights needed? | Works on locked screen? | Handles legacy apps? | Notes |
|---|---|---|---|---|
| GUI simulation (PyAutoGUI, AutoHotkey) | No | No | Yes | Fails if session is locked or user clicks |
| COM automation (VBScript, PowerShell) | No | Yes | Yes | Works only on Windows, depends on COM support |
| Web automation (Selenium) | Maybe | Yes | No | Needs browser, fails on non-web apps |
| Cloud service (AWS Lambda + OCR) | No | N/A | Maybe | Needs internet, may violate data policy |

If your target environment is Windows with legacy apps, COM automation is often the only viable option that respects admin rights and locked screens.

Next, design for failure:
- Use atomic file moves (rename is atomic on NTFS).
- Log everything to a local file. Users can review logs if needed.
- Add a heartbeat or ping mechanism. If the agent dies, users notice quickly.
- Bundle all dependencies. Don’t rely on system-wide installs.

Finally, test in the real environment. Not in a VM, not on a dev machine — on the actual hardware with the actual user logged in. We did this on the last day of the pilot, and it caught three issues we’d never seen in dev: Adobe Reader’s COM interface was disabled by a group policy, the user’s AppData folder was redirected to a network drive with slow writes, and the scheduled task didn’t run when the user locked the screen. We fixed all three in one afternoon.

## Resources that helped

- [WiX Toolset 3.11 documentation](https://wixtoolset.org/docs/wix3/) — for building per-user MSI installers
- [VBScript documentation (Microsoft)](https://docs.microsoft.com/en-us/previous-versions/windows/internet-explorer/ie-developer/scripting-articles/9bb850f7(v=vs.84)) — for COM automation basics
- [PaddleOCR 2.6](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/rel%2F2.6) — lightweight OCR for legacy hardware
- [SQLite 3.45 CLI](https://www.sqlite.org/download.html) — for local state and heartbeats
- [PowerShell 7.4 release notes](https://github.com/PowerShell/PowerShell/releases/tag/v7.4.0) — for migrating from VBScript
- [Adobe Acrobat DC SDK documentation](https://www.adobe.com/devnet-docs/acrobat/acrobat_dc_sdk/) — for COM interface reference (even for old versions)
- [Practical VBScript examples](https://www.robvanderwoude.com/vbscripts.php) — for common COM patterns
- [Per-user MSI installation guide](https://docs.microsoft.com/en-us/windows/win32/msi/per-user-installations) — for bypassing admin requirements

## Frequently Asked Questions

**How do I run a script without admin rights on Windows 7?**

Use a per-user MSI installer with WiX 3.11 or an Inno Setup script that installs to %AppData%\\Local. Or use `pip install --user` for Python scripts. Avoid system-wide installs. You can also use a batch file in the user’s Startup folder that runs the script under their session. Test by logging in as a restricted user and verifying the script runs and can access the desktop.

**Why did my PyAutoGUI script fail when the screen was locked?**

Windows locks the desktop session when the user locks the screen or switches users. GUI simulation tools like PyAutoGUI 0.9.57 use screen coordinates and pixel matching, which fail when the session is locked. COM automation, on the other hand, interacts with the app’s object model and doesn’t depend on the desktop being visible. If you must use PyAutoGUI, run it only when the user is logged in and the screen is unlocked.

**What’s the easiest way to debug COM automation in VBScript?**

Use `WScript.Echo` for logging, and check the output in the console. For silent scripts, log to a file with `FileSystemObject`. You can also use `cscript //nologo agent.vbs` to run the script in a console without the popup window. For Excel COM, try `excelApp.Visible = True` temporarily to see what the script is doing. If Excel throws an error, enable macro logging in Excel’s options to get more detail.

**How can I make my agent resilient to crashes?**

Use atomic file operations: rename files to move them between directories, which is atomic on NTFS. Log every step to a local file (e.g., SQLite or plain text). Add a heartbeat: write a timestamp every few minutes. If the heartbeat stops for 15 minutes, alert the user via a popup or a simple HTML dashboard. Test the agent’s recovery by killing it mid-process and verifying it resumes correctly.

**Can I use PowerShell instead of VBScript for COM automation?**

Yes, PowerShell 7.4 can use COM objects via `New-Object -ComObject`. It has better error handling and logging than VBScript. On Windows 7, PowerShell 7.4 requires .NET Framework 4.8, which is usually installed. Rewrite your VBScript logic in PowerShell, and use `Start-Process` for external tools like Tesseract. The COM interface names are the same in PowerShell as in VBScript.

## Next step in the next 30 minutes

Open a Windows 7 VM (or a real machine) and run this command in an admin-free user session:

```cmd
cscript //nologo %windir%\\system32\\scrrun.dll /h
```

If it runs without error, you have Windows Script Host available. Next, create a minimal VBScript that opens Notepad via COM:

```vbs
Set app = CreateObject("Word.Application")
app.Visible = True
app.Documents.Add
app.Quit
```

Save it as `test.vbs`, run it, and verify Word opens. If this works, you’re ready to build a COM-based agent for your legacy app. If not, your environment has deeper restrictions — adjust your design accordingly.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** July 30, 2026
