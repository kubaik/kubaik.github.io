# Changelog: indexing hygiene (2026-10-21)

Fixes the Search Console "Page indexing" failures for https://kubaik.github.io
(26 × 404, 8 × Page with redirect, 1 × noindex) and adds guards so they don't recur.

## Root causes

| GSC item                 | Cause                                                                                                                   | Fix                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Page with redirect (8)   | `visibility_automator.py` / `tweet_entropy.py` published slash-less URLs (`/slug`). GitHub Pages 301s them to `/slug/`. | `utilsIndexing/url_utils.public_post_url()` used everywhere |
| Not found 404 (26)       | Posts were published, crawled, then deleted by bulk commits that bypassed the tombstone logic.                          | `utilsIndexing/retire_registry.py` ledger + tombstones      |
| noindex (1)              | `track-ai-agent-costs-in-fractal-workflows` was quarantined Oct 3–4, restored Oct 4. GSC data is stale.                 | No code change. Click **Validate fix** in GSC               |
| Crawled, not indexed (4) | URLs not provided.                                                                                                      | Not addressed. Send the URLs for a targeted audit           |

Also found by the new gate: `/tag/` was in `sitemap.xml` but renders `noindex`.

## Changed files (replace in place)

| File                                    | Change                                                                                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `static_site_generator.py`              | Calls `utilsIndexing.retire_registry.sync_ledger_and_tombstones()` after the 404 page is written (wrapped in try/except, so it can never break a build). Removed `/tag/` from the sitemap. |
| `tweet_entropy.py`                      | Imports `public_post_url` from `utilsIndexing.url_utils`; post links now end in `/`.                                                                                                       |
| `visibility_automator.py`               | Same, in 8 places.                                                                                                                                                                         |
| `scripts/generate_sitemap.py`           | Removed `/tag/` from `_REQUIRED_STATIC_PATHS`.                                                                                                                                             |
| `.github/workflows/blog-automation.yml` | New step "Indexing hygiene gate", placed after the noindex/sitemap audit and before OG image generation.                                                                                   |

## New files (in `utilsIndexing/`, except the seed JSON)

Create the folder `utilsIndexing/` at the repo root with these four files: `__init__.py`, `url_utils.py`, `retire_registry.py`, `check_indexing_hygiene.py`. Imports use the package path (`from utilsIndexing.url_utils import public_post_url`), so run scripts from the repo root, as the workflow already does.

### `utilsIndexing/url_utils.py`

- `public_post_url(slug, base_url=...)` returns `https://kubaik.github.io/<slug>/`.
- `normalize_url(url)` adds a trailing slash to extensionless paths.
- Use these for any new code that emits a post URL (tweets, IndexNow, RSS, emails).

### `utilsIndexing/retire_registry.py`

Runs automatically on every `blog_system.py build`. It keeps two files, both written to the working directory (run from the repo root):

- `published_slugs.json`: every slug that has ever been live.
- `retired_slugs.json`: slugs in the ledger that no longer exist, with `retired_at`.

Behaviour:

- A slug in the ledger with no `docs/<slug>/post.json` is marked retired, however it was deleted (script, `git rm`, by hand).
- Retired slugs younger than `tombstone_days` (default 180) get a `200` page with `noindex, follow`. It is never listed in the sitemap.
- Older retired slugs are left as plain 404s.
- A slug that comes back to life is removed from the retired list.

Optional `config.yaml` keys:

```yaml
tombstone_retired: true # false = leave deleted URLs as plain 404s
tombstone_days: 180
```

Import the rest of the 404s from GSC:

1. GSC → Indexing → Pages → _Not found (404)_ → **Export** → CSV.
2. Run `python utilsIndexing/retire_registry.py import-gsc "Not found (404).csv"` from the repo root.
3. Commit `retired_slugs.json` and `published_slugs.json`. The next build creates the tombstones.

Limits:

- The ledger starts empty. Posts deleted before today are only known if you import them (the 8 seeded slugs are in `retired_slugs.json`).
- Tombstones are `200` + `noindex`, so Google may report them as soft 404s or "Excluded by noindex" instead of 404. Both are harmless. Set `tombstone_retired: false` if you prefer plain 404s.

### `utilsIndexing/check_indexing_hygiene.py`

Read-only gate. Exit code 1 on any error.

```bash
python blog_system.py build
python utilsIndexing/check_indexing_hygiene.py --docs ./docs --base https://kubaik.github.io
# add --strict to fail on warnings (slash-less internal links) too
```

It checks:

1. Every sitemap URL ends in `/`, points at a real `docs/<slug>/index.html`, and is not `noindex`, a meta-refresh stub or a tombstone.
2. No sitemap URL is in `retired_slugs.json`.
3. Every internal link in built HTML resolves to a real page.
4. Canonicals have a trailing slash; slash-less internal links produce warnings.

It is wired into the workflow and will block deploys when it fails. To make it non-blocking while you get to a clean state, add `continue-on-error: true` to that step, then remove it later.

### `retired_slugs.json` (repo root)

Seed list: the 8 slugs visible in the GSC 404 screenshot.

## Deploy checklist

1. Copy the changed files over the originals and the `utilsIndexing/` folder and `retired_slugs.json` into the repo root.
2. Run locally: `python blog_system.py build && python utilsIndexing/check_indexing_hygiene.py`. Expect 0 errors (the old `docs/sitemap.xml` error about `/tag/` disappears after the rebuild).
3. Commit and push. The workflow runs the gate on every run.
4. Import the full GSC 404 export (see above) and commit the updated JSON files.
5. In GSC, click **Validate fix** on the noindex, 404 and redirect items. Re-crawling takes weeks.
6. Send the 4 "Crawled – currently not indexed" URLs for a content audit.

## Rollback

Revert the five changed files. Delete the new files if wanted. Nothing else depends on them: the registry hook in the generator is wrapped in try/except, and the workflow gate is a standalone step.
