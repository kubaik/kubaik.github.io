# Secure file uploads: where pipelines break

File uploads are the part of an application where untrusted bytes cross a trust boundary. Most teams get the UI right — drag-and-drop, progress bars, retry logic — and then discover, usually after a security review, that the pipeline behind it trusts the client far more than it should. The symptom is rarely a crash. It's a finding in a penetration test report, a bucket with public read enabled, or a file that was "image/jpeg" on the way in and an HTML document on the way out.

This post walks through the failure modes that show up across unrelated codebases: the same root causes behind different error messages. It's a diagnostic guide, not a tutorial. The goal is to help you self-triage an existing upload path before a reviewer does it for you.

## The symptom: a file that passes validation but shouldn't

The exact symptom varies by stack, but the pattern is consistent: a file is accepted by the application, stored somewhere, and later served or processed in a way that violates an assumption the team thought was enforced. Common error strings and log lines that indicate this class of problem:

- `Content-Type` mismatch warnings from a virus scanner or image processor: `expected image/jpeg, got text/html`
- A 200 response from your upload endpoint for a file whose magic bytes don't match its extension
- S3 `AccessDenied` on an object that should be private, or the opposite: an object that is readable by anyone with the URL
- `413 Request Entity Too Large` from a proxy that disagrees with your application's limit
- `Multipart: Boundary not found` or `Unexpected end of form` from a body parser when the request is malformed or truncated
- A filename in your storage layer containing `../` or a null byte

If any of these appear in your logs, the pipeline is doing less validation than the team believes. The rest of this post is about which layer is failing and how to confirm it.

## Cause 1: trusting the client's Content-Type header

The most common cause is that the application uses the `Content-Type` header from the multipart request as the file's type. That header is set by the client. A browser will usually set it correctly based on the file extension, but an attacker controls it completely. A file named `invoice.pdf` can be uploaded with `Content-Type: image/png`, or with no type at all.

### How to confirm

Check whether your validation code reads `file.mimetype` (Express/Multer), `UploadFile.content_type` (FastAPI/Starlette), or the equivalent in your framework. If the only check is against that value, you have this cause. A quick test: upload a text file with `Content-Type: image/jpeg` using `curl -F 'file=@test.txt;type=image/jpeg' https://your-app/upload`. If it's accepted, the pipeline trusts the header.

### The fix

Validate the file's actual content, not its declared type. For images, parse the header bytes. The magic numbers for common formats are stable and short:

```python
# Python: check magic bytes for common image types
def sniff_image_type(header: bytes) -> str | None:
    if header.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'image/gif'
    if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'image/webp'
    return None

# In your handler:
header = await file.read(16)
await file.seek(0)
detected = sniff_image_type(header)
if detected not in ALLOWED_TYPES:
    raise HTTPException(status_code=415, detail='Unsupported file type')
```

For formats beyond images — PDF, Office documents, ZIP archives — use a library that parses the container rather than a hand-rolled header check. `python-magic` wraps libmagic and is the usual choice in Python; on Node.js, the `file-type` package reads magic bytes for a wide range of formats. Neither is a security boundary on its own: a polyglot file can be a valid JPEG and a valid HTML document at the same time. The point is to stop trusting the header, then add a second layer (see Cause 3).

## Cause 2: path traversal and filename injection

The second most common cause is using the client-supplied filename directly in a filesystem path or object key. `../../etc/passwd` as a filename is the classic example, but the more realistic problem is overwriting an existing object or writing outside the intended prefix. On object storage the equivalent is a key like `uploads/../other-tenant/file.png`, which some SDKs and proxies will normalize differently.

### How to confirm

Search your codebase for the upload handler and check whether the original filename is used in any path construction. In Node.js, look for `path.join(uploadDir, file.originalname)`. In Python, `os.path.join(UPLOAD_DIR, file.filename)`. If the filename is concatenated or joined without sanitization, you have this cause. Also check whether the filename is used in a database query or a `Content-Disposition` header — both have their own injection risks.

### The fix

Generate the storage key yourself and never use the client filename for anything except display. A UUID or a content hash is sufficient. If you need the original name for the user's benefit, store it as metadata and sanitize it on display.

```javascript
// Node.js: generate a safe key, keep original name as metadata
const crypto = require('crypto');
const path = require('path');

function safeKey(originalName) {
  const ext = path.extname(originalName).toLowerCase();
  const allowed = new Set(['.jpg', '.jpeg', '.png', '.gif', '.webp']);
  if (!allowed.has(ext)) throw new Error('extension not allowed');
  return `uploads/${crypto.randomUUID()}${ext}`;
}
```

Notice that the extension is still taken from the client filename. That's acceptable only because it's checked against an allowlist and the rest of the name is discarded. If you want to be stricter, derive the extension from the sniffed type instead. The key point is that the path is constructed entirely from server-controlled values.

## Cause 3: storing and serving user content from the same origin

A pipeline can validate type and sanitize filenames and still be exploitable if uploaded files are served from the application's own origin. An HTML file uploaded as `report.html` and served with `Content-Type: text/html` runs with the same origin as your app. That gives it access to cookies, local storage, and any same-origin API. This is stored XSS with extra steps.

### How to confirm

Request an uploaded file and inspect the response headers. If `Content-Type` reflects the uploaded file's type and there is no `Content-Disposition: attachment`, the file is being rendered. If the file is served from the same hostname as your application (not a separate domain or a CDN with a different origin), the risk is present. Check whether `X-Content-Type-Options: nosniff` is set; without it, some browsers will sniff the content type and render HTML even when you send `text/plain`.

### The fix

Serve user content from a separate origin, ideally a domain that shares no cookies with the application. On AWS, that usually means an S3 bucket behind CloudFront with a distinct hostname, not a path on the app's domain. Set `Content-Disposition: attachment` for anything that isn't an image you intend to render inline, and always send `X-Content-Type-Options: nosniff`. If you must serve from the same origin, use a strict Content Security Policy that disallows inline scripts and restricts `script-src` to your own bundle.

A related failure: presigned URLs with excessively long expiry. A presigned S3 URL is a bearer token. If it leaks — through logs, referrer headers, or a shared link — anyone can use it until it expires. Keep expiry short (minutes, not days) and generate a new URL per request rather than storing one.

## Cause 4: size limits enforced in only one layer

The `413 Request Entity Too Large` error is usually a sign that two layers disagree about the maximum size. A common setup is an application-level limit of 10 MB and a reverse proxy (nginx, ALB, API Gateway) with a lower default. The request is rejected before your handler runs, so your application-level error handling never fires and the user sees a generic proxy error.

### How to confirm

Check the limits at every hop: the client, the CDN, the load balancer, the reverse proxy, the application framework, and the storage layer. For nginx, `client_max_body_size` defaults to 1 MB. For AWS API Gateway, the payload limit is 10 MB for REST APIs. Application Load Balancers have their own limits. If any of these is lower than your application's limit, the effective limit is the lowest one, and the error will come from that layer, not yours.

### The fix

Decide on a single maximum size and configure every layer to match or exceed it, with the application enforcing the final limit. Return a structured error from the application so clients can distinguish "too large" from "wrong type." If you're uploading directly to S3 with presigned URLs, enforce the size limit in the presigned policy using a `content-length-range` condition; otherwise the client can upload anything up to S3's own limits.

## Cause 5: processing untrusted files in-process

Image resizing, PDF text extraction, and archive decompression are all operations that parse attacker-controlled input. Running them in the same process as your web server means a parser bug or a decompression bomb takes down the application. A zip bomb is the simplest example: a small compressed file that expands to gigabytes when decompressed.

### How to confirm

Look at where your processing code runs. If `sharp`, ImageMagick, `pdftotext`, or an archive library is called directly in a request handler, you have this cause. Check whether there are limits on decompressed size, pixel count, or processing time. ImageMagick in particular has a history of vulnerabilities when processing untrusted input; the project's own security policy recommends policy.xml restrictions.

### The fix

Move processing to an isolated worker with resource limits: a separate container with CPU and memory caps, a Lambda function with a timeout, or a queue-backed worker. Set explicit limits on output size and processing time. For images, configure the decoder to reject images above a pixel count. For archives, check the uncompressed size before extracting and refuse anything above a threshold. The web request should only validate and store the raw bytes; processing happens asynchronously.

## Cause 6: missing antivirus and content disarm

If the pipeline accepts documents that will later be opened by other users, malware scanning is part of the trust boundary. The symptom is usually indirect: a user reports a suspicious attachment, or a security review asks where scanning happens and the answer is "nowhere."

### How to confirm

Trace the lifecycle of an uploaded document. If it goes from upload to storage to download without passing through a scanner, there is no scanning. Check whether the scanner result is enforced — a scan that runs but doesn't block download is logging, not prevention.

### The fix

Scan asynchronously after upload, tag the object as `pending`, `clean`, or `infected`, and only allow download when the tag is `clean`. For formats that support it, content disarm and reconstruction (CDR) is more robust than signature-based scanning: it rebuilds the file from parsed content, discarding anything that isn't part of the document structure. ClamAV is the usual open-source scanner; commercial CDR services exist but the mechanism is the same. The important design decision is that scanning is a gate, not an alert.

## A decision framework for the pipeline

When designing or reviewing an upload path, the useful question is not "is this secure?" but "which layer enforces which invariant?" The table below maps each invariant to the layer that should own it. If two layers claim the same invariant, one of them is probably not enforcing it.

| Invariant | Enforced by | Typical mistake |
|---|---|---|
| File type is allowed | Application, via magic-byte sniffing | Trusting `Content-Type` header |
| File size is bounded | Every hop, with application as final gate | Only application limit set |
| Storage key is safe | Application, generated server-side | Using client filename in path |
| Content is served safely | CDN/object storage with separate origin | Serving from app origin |
| Content is free of malware | Async scanner with download gate | Scanning but not blocking |
| Processing is isolated | Worker with resource limits | Decoding in request handler |

## Prevention: the checklist to run before the next review

A secure upload pipeline is mostly a matter of not trusting the client at any layer. The checklist below is ordered by how often each item is missing in practice.

1. **Sniff content, don't read the header.** Validate magic bytes for every accepted format. Reject anything that doesn't match an allowlist.
2. **Generate storage keys server-side.** Never use the client filename in a path or object key. Keep it as metadata only.
3. **Serve from a separate origin.** Use a distinct hostname for user content, set `Content-Disposition: attachment` and `X-Content-Type-Options: nosniff`.
4. **Align size limits across every hop.** Proxy, load balancer, application, and storage should agree. Enforce the final limit in the presigned policy if uploading directly.
5. **Scan asynchronously and gate download.** A scan that doesn't block is telemetry, not security.
6. **Process in isolation.** Decode images, PDFs, and archives in a worker with CPU, memory, and time limits.
7. **Log the decisions.** Record the sniffed type, the size, the scanner result, and the generated key. When something goes wrong, the log is how you reconstruct which layer failed.

## The next 30 minutes: check your upload handler

Open the file that handles uploads in your application and search for the client-supplied filename and `Content-Type`. If either appears in a path, a storage key, or a validation check, you've found the first thing to fix. Replace the filename with a server-generated UUID and add a magic-byte check before the file is written anywhere. That single change closes the two most common findings in upload pipelines, and it's small enough to ship today.

## Frequently Asked Questions

**Why is checking the file extension not enough for secure uploads?**
Because the extension is part of the client-supplied filename and can be anything. A file named `photo.jpg` can contain HTML, a ZIP archive, or an executable. The extension tells you what the client claims the file is, not what it is. Magic-byte sniffing reads the actual content header, and for formats with complex containers, parsing the structure is the only reliable check.

**Should I store uploads in S3 or on the local filesystem?**
For most applications, object storage is the better default because it separates the storage layer from the application process and makes it easier to serve content from a different origin. The security properties depend on configuration: block public access, use presigned URLs with short expiry, and enable versioning so an overwrite doesn't destroy the original. Local filesystem storage is workable for small deployments but requires you to handle path traversal, disk quotas, and backup yourself.

**How do I prevent zip bombs and decompression attacks?**
Check the uncompressed size before extracting, and refuse archives that exceed a threshold relative to their compressed size. Run extraction in a worker with a memory limit and a timeout so a bomb fails the worker rather than the host. The same principle applies to image decoding: set a maximum pixel count and reject images above it before allocating buffers.

**What is the difference between antivirus scanning and content disarm?**
Antivirus scanning matches known signatures and heuristics; it catches known malware but misses new or targeted files. Content disarm and reconstruction parses the document and rebuilds it from the structural elements it understands, discarding anything else — macros, embedded scripts, external references. CDR is more robust against unknown threats but can alter the document's appearance or functionality. Many pipelines use both: scan first, then disarm.

## Official documentation

- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
