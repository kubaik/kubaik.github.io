/**
 * Interactive Code Runner for Static Blog
 * Supports: Python (via Pyodide), JavaScript
 * Works on GitHub Pages (fully client-side)
 */

(function () {
  "use strict";

  let pyodideInstance = null;
  let pyodideLoading = false;
  let pyodideReady = false;
  const pyodideCallbacks = [];

  // ── Detect language from <code> class ─────────────────────────
  function detectLanguage(codeEl) {
    const cls = codeEl.className || "";
    if (/language-python|language-py\b/.test(cls)) return "python";
    if (/language-javascript|language-js\b/.test(cls)) return "javascript";
    return null;
  }

  // ── Load Pyodide lazily ────────────────────────────────────────
  function loadPyodide(callback) {
    if (pyodideReady) return callback(null, pyodideInstance);
    pyodideCallbacks.push(callback);
    if (pyodideLoading) return;
    pyodideLoading = true;

    const script = document.createElement("script");
    script.src =
      "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js";
    script.onload = async () => {
      try {
        pyodideInstance = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/",
          stdout: () => {},
          stderr: () => {},
        });
        pyodideReady = true;
        pyodideCallbacks.forEach((cb) => cb(null, pyodideInstance));
      } catch (e) {
        pyodideCallbacks.forEach((cb) => cb(e));
      }
    };
    script.onerror = () => {
      pyodideCallbacks.forEach((cb) =>
        cb(new Error("Failed to load Pyodide"))
      );
    };
    document.head.appendChild(script);
  }

  // ── Run Python ─────────────────────────────────────────────────
  async function runPython(code, outputEl, runBtn) {
    outputEl.textContent = "⏳ Loading Python runtime…";
    outputEl.className = "cr-output cr-loading";
    outputEl.style.display = "block";

    loadPyodide(async (err, pyodide) => {
      if (err) {
        outputEl.textContent = "❌ Could not load Python: " + err.message;
        outputEl.className = "cr-output cr-error";
        runBtn.disabled = false;
        runBtn.textContent = "▶ Run";
        return;
      }

      outputEl.textContent = "⏳ Running…";

      let captured = [];

      // Redirect stdout/stderr
      pyodide.setStdout({ batched: (s) => captured.push(s) });
      pyodide.setStderr({ batched: (s) => captured.push("⚠ " + s) });

      try {
        const result = await pyodide.runPythonAsync(code);
        const out = captured.join("\n");
        const display =
          out.trim() ||
          (result !== undefined && result !== null
            ? String(result)
            : "(no output)");
        outputEl.textContent = display;
        outputEl.className = "cr-output cr-success";
      } catch (e) {
        outputEl.textContent = "❌ " + e.message;
        outputEl.className = "cr-output cr-error";
      } finally {
        runBtn.disabled = false;
        runBtn.textContent = "▶ Run";
      }
    });
  }

  // ── Run JavaScript ─────────────────────────────────────────────
  function runJavaScript(code, outputEl, runBtn) {
    outputEl.style.display = "block";
    let captured = [];
    const origLog = console.log;
    const origWarn = console.warn;
    const origError = console.error;

    console.log = (...args) => captured.push(args.map(String).join(" "));
    console.warn = (...args) =>
      captured.push("⚠ " + args.map(String).join(" "));
    console.error = (...args) =>
      captured.push("❌ " + args.map(String).join(" "));

    try {
      // Wrap in async IIFE so await works
      const asyncCode = `(async () => { ${code} })()`;
      const result = eval(asyncCode);

      if (result && typeof result.then === "function") {
        outputEl.textContent = "⏳ Running…";
        outputEl.className = "cr-output cr-loading";
        result
          .then((val) => {
            console.log = origLog;
            console.warn = origWarn;
            console.error = origError;
            const out =
              captured.join("\n") ||
              (val !== undefined ? String(val) : "(no output)");
            outputEl.textContent = out;
            outputEl.className = "cr-output cr-success";
          })
          .catch((e) => {
            console.log = origLog;
            console.warn = origWarn;
            console.error = origError;
            outputEl.textContent = "❌ " + e.message;
            outputEl.className = "cr-output cr-error";
          })
          .finally(() => {
            runBtn.disabled = false;
            runBtn.textContent = "▶ Run";
          });
        return;
      }

      const out =
        captured.join("\n") ||
        (result !== undefined ? String(result) : "(no output)");
      outputEl.textContent = out;
      outputEl.className = "cr-output cr-success";
    } catch (e) {
      outputEl.textContent = "❌ " + e.message;
      outputEl.className = "cr-output cr-error";
    } finally {
      console.log = origLog;
      console.warn = origWarn;
      console.error = origError;
      runBtn.disabled = false;
      runBtn.textContent = "▶ Run";
    }
  }

  // ── Build the runner widget around a <pre><code> block ─────────
  function buildRunner(pre, codeEl, lang) {
    const originalCode = codeEl.textContent;

    // Wrapper
    const wrapper = document.createElement("div");
    wrapper.className = "cr-wrapper";

    // Header bar
    const header = document.createElement("div");
    header.className = "cr-header";

    const langBadge = document.createElement("span");
    langBadge.className = "cr-lang cr-lang-" + lang;
    langBadge.textContent = lang === "python" ? "Python" : "JavaScript";

    const actions = document.createElement("div");
    actions.className = "cr-actions";

    const resetBtn = document.createElement("button");
    resetBtn.className = "cr-btn cr-reset";
    resetBtn.title = "Reset to original";
    resetBtn.innerHTML =
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg> Reset';

    const runBtn = document.createElement("button");
    runBtn.className = "cr-btn cr-run";
    runBtn.innerHTML =
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg> Run';

    actions.appendChild(resetBtn);
    actions.appendChild(runBtn);
    header.appendChild(langBadge);
    header.appendChild(actions);

    // Editable textarea
    const textarea = document.createElement("textarea");
    textarea.className = "cr-editor";
    textarea.value = originalCode;
    textarea.spellcheck = false;
    textarea.autocomplete = "off";
    textarea.setAttribute("autocorrect", "off");
    textarea.setAttribute("autocapitalize", "off");

    // Auto-resize
    function resizeTextarea() {
      textarea.style.height = "auto";
      textarea.style.height = textarea.scrollHeight + "px";
    }
    textarea.addEventListener("input", resizeTextarea);

    // Tab key support
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const s = textarea.selectionStart;
        const end = textarea.selectionEnd;
        textarea.value =
          textarea.value.substring(0, s) +
          "    " +
          textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = s + 4;
      }
    });

    // Output area
    const outputEl = document.createElement("pre");
    outputEl.className = "cr-output";
    outputEl.style.display = "none";

    // Wire up buttons
    runBtn.addEventListener("click", () => {
      runBtn.disabled = true;
      runBtn.textContent = "Running…";
      const code = textarea.value;
      if (lang === "python") {
        runPython(code, outputEl, runBtn);
      } else {
        runJavaScript(code, outputEl, runBtn);
      }
    });

    resetBtn.addEventListener("click", () => {
      textarea.value = originalCode;
      resizeTextarea();
      outputEl.style.display = "none";
      outputEl.textContent = "";
      outputEl.className = "cr-output";
    });

    // Assemble
    wrapper.appendChild(header);
    wrapper.appendChild(textarea);
    wrapper.appendChild(outputEl);

    // Replace original <pre> with wrapper
    pre.parentNode.replaceChild(wrapper, pre);

    // Initial sizing
    requestAnimationFrame(resizeTextarea);
  }

  // ── Inject styles ──────────────────────────────────────────────
  function injectStyles() {
    if (document.getElementById("cr-styles")) return;
    const style = document.createElement("style");
    style.id = "cr-styles";
    style.textContent = `
      .cr-wrapper {
        border-radius: 10px;
        overflow: hidden;
        margin: 1.5rem 0;
        border: 1.5px solid #e2e8f0;
        background: #0f1117;
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
        box-shadow: 0 4px 24px rgba(0,0,0,0.13);
      }
      .cr-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        background: #1a1d2e;
        border-bottom: 1px solid #2d3148;
        min-height: 38px;
      }
      .cr-lang {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 20px;
      }
      .cr-lang-python  { background: #3b4fc4; color: #b8cbff; }
      .cr-lang-javascript { background: #7c6000; color: #fde68a; }
      .cr-actions { display: flex; gap: 8px; }
      .cr-btn {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 13px;
        border-radius: 6px;
        border: none;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.15s, transform 0.1s;
        font-family: inherit;
      }
      .cr-btn:hover:not(:disabled) { opacity: 0.85; transform: scale(1.03); }
      .cr-btn:disabled { opacity: 0.45; cursor: not-allowed; }
      .cr-run   { background: #22c55e; color: #fff; }
      .cr-reset { background: #3b4563; color: #94a3b8; }
      .cr-editor {
        display: block;
        width: 100%;
        padding: 16px 18px;
        background: #0f1117;
        color: #e2e8f0;
        border: none;
        resize: none;
        font-family: inherit;
        font-size: 13.5px;
        line-height: 1.7;
        tab-size: 4;
        outline: none;
        box-sizing: border-box;
        min-height: 60px;
        caret-color: #60a5fa;
      }
      .cr-editor:focus { background: #12151f; }
      .cr-output {
        margin: 0;
        padding: 12px 18px;
        font-family: inherit;
        font-size: 13px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
        border-top: 1px solid #2d3148;
      }
      .cr-loading { color: #94a3b8; background: #0f1117; }
      .cr-success { color: #86efac; background: #0d1f14; }
      .cr-error   { color: #fca5a5; background: #1c0a0a; }
    `;
    document.head.appendChild(style);
  }

  // ── Main: find all runnable code blocks ────────────────────────
  function init() {
    injectStyles();
    const blocks = document.querySelectorAll(
      ".post-content pre code, article.blog-post pre code"
    );
    blocks.forEach((codeEl) => {
      const lang = detectLanguage(codeEl);
      if (!lang) return;
      const pre = codeEl.closest("pre");
      if (!pre) return;
      buildRunner(pre, codeEl, lang);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();