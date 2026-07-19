# Design Variant C — Full Polish (A + B + Actions)

**Variant:** C
**Branch:** `design/perf-full-polish`
**Scope:** `web/templates/run_detail.html`, `web/static/style.css`, plus small hooks in `web/app.py`, `web/templates/perf.html`, `web/templates/data.html`.

## From A (readability / layout)

1. **Reply-first / side-by-side Prompt | Reply.** Each example on the detail page now renders in an `.io-grid` with two flex columns: Instruction + Input on the left ("Prompt"), Output on the right ("Reply"). At `>= 760px` the columns sit side by side; below that they stack with Output shown **first** (via CSS `order: -1` on `.io-col-reply`, reset to `order: 0` at `>= 760px`).
2. **Capped Input height.** The Input field is wrapped in `.io-clamp` (`max-height: 12rem`, bottom gradient fade). A `Show more` / `Show less` button (`.io-clamp-toggle`) is revealed by JS only when the content actually overflows, and toggles an `.expanded` class that removes the cap.
3. **Collapsed Howto.** On the detail page only, the shared `howto.html` partial is now wrapped in `<details class="howto-collapsed"><summary>How to use this page</summary>...</details>` so it's collapsed by default. `perf.html` / `data.html` keep the always-visible howto (out of scope for this page-specific change).
4. **Truncation badge.** A `possibly truncated` badge (`.badge-truncated`) appears next to the Output label whenever `result.metrics.output_tokens_total >= max_new_tokens` and `max_new_tokens > 0` (read from `result.metadata.max_new_tokens`). Includes a `title` tooltip with the exact numbers.

## From B (compact metrics / header)

5. **Compact header pills.** The old "Summary" card (benchmark/model/device/config as big `.metric` blocks) is replaced by a single `.pill-row` of small rounded pills: benchmark · model · device · config · tok/s · latency. tok/s and latency are highlighted (`.pill-accent`).
6. **Cleaner metrics strip.** The Metrics section now renders `.metric-chip` items in a `.metrics-strip`: values are rounded to 1 decimal for floats (latency/tok-per-sec), friendly labels replace raw field names (e.g. `latency_ms_p50` → "P50 latency (ms)"), and `latency_ms_p50` / `latency_ms_p95` are hidden entirely when `samples == 1` (mean/p50/p95 are identical and redundant with a single sample).
7. **Metadata behind `<details>`.** Raw run metadata (everything except the `examples` payload, which is rendered separately) is now tucked behind `<details class="advanced-details"><summary>Advanced (metadata)</summary>...</details>`, using the same compact chip styling.

## Actions (C only)

8. **Copy buttons.** Each Instruction / Input / Output block has a small `Copy` button (`data-copy-target="<id>"`) wired up by a tiny inline `<script>` using `navigator.clipboard.writeText`. Button label flips to `Copied!` for ~1.2s as feedback.
9. **Re-run link.** Each example card now has a `Re-run` action linking to `/perf?instruction=...&input=...` (URL-encoded). `GET /perf` in `web/app.py` now reads `instruction` / `input` query params and passes them to `perf.html` as `defaults`, which are used as the form's `value=`/textarea content. A green notice ("Prefilled from a previous run's Re-run link…") appears when prefilled.
10. **Save as example link.** Each example card also has a `Save as example` action linking to `/data?instruction=...&input_text=...&output=...` (the model's output becomes the proposed gold `output`). `GET /data` in `web/app.py` now reads those same-shaped query params and passes them as `prefill` to `data.html`, prefilling the Add-training-example form. A matching notice is shown. This is the "GET prefill into `data.html` fields" option — no new POST route was needed.

## Files changed

- `web/templates/run_detail.html` — full restructure: pills header, collapsed howto, side-by-side prompt/reply with clamp + copy + truncation badge, compact metrics strip, `<details>` metadata, re-run/save-as-example actions, inline JS for copy + clamp toggle.
- `web/static/style.css` — new styles for pills, collapsible howto/advanced `<details>`, metrics strip/chips, io-grid side-by-side layout + mobile reorder, copy buttons, input clamp, truncation badge, `.btn-sm`, `.io-actions`.
- `web/app.py` — `GET /perf` and `GET /data` now read prefill query params (`instruction`/`input` and `instruction`/`input_text`/`output` respectively) and pass them to the templates as `defaults` / `prefill`.
- `web/templates/perf.html` — Instruction/Input fields use `defaults.*` instead of hardcoded values; notice banner when prefilled via Re-run.
- `web/templates/data.html` — Instruction/Input/Output fields use `prefill.*`; notice banner when prefilled via Save as example.

## Verification

Rendered all four routes (`/`, `/perf`, `/perf?instruction=...&input=...`, `/data`, `/data?instruction=...&input_text=...&output=...`, `/perf/<run_id>`) against a live `uvicorn` instance with a synthetic benchmark JSON fixture (long Input to trigger the clamp, `output_tokens_total == max_new_tokens` to trigger the truncation badge, `samples == 1` to verify p50/p95 are hidden). All returned `200` with no server errors, and the rendered HTML was inspected directly to confirm pills, collapsed howto, clamp markup, truncation badge, compact metric chips, `<details>` metadata, and correctly URL-encoded Re-run / Save-as-example links.
