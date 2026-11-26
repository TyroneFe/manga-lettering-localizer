# Localize a first page

Choose a matching color page and English donor with ordinary black lettering on a flat white balloon. Use upright, opaque 8-bit pages for this workflow. The pilot opens ZIP or CBZ archives containing PNG, JPEG or WebP images. The inventory tool also recognizes loose images and can inspect CBR files with a compatible local extractor.

## 1. Inventory the originals

Use this layout under the checkout. Place your archives in the two source folders and copy the example JSON files into `my-study`.

```text
workspace/
  source/
    color/edition.cbz
    english/edition.cbz
my-study/
  pilot.json
  regions.json
```

```powershell
.\.venv\Scripts\python.exe -m manga_localizer.inventory workspace/source/color workspace/source/english --report workspace/cache/inventory.json --proxies
```

The inventory lists archive member names, their order, hashes and decode metadata. Use those names when preparing the study. Keep the inventory on the machine that created it because it records local paths.

## 2. Register the donor

In `my-study/pilot.json`, set `color_archive` to `color/edition.cbz` and `english_archive` to `english/edition.cbz`. Set `candidate_range` to the one-based inclusive range of color image members to search, in natural filename order. Choose an `english_contains` substring that matches exactly one donor member. Give the view a short ID such as `page-one`. Keep `donor_fraction` as `[0, 1]` for a single page. [Example configuration](../examples/pilot.json)

```powershell
.\.venv\Scripts\python.exe -m manga_localizer.pilot my-study/pilot.json --root workspace
```

Inspect `workspace/review/pilot/page-one-alignment.png` and `registration.json`. Compare the panel boundaries, faces and details near the intended balloon. If the identity or alignment is ambiguous, resolve it before selecting a lettering region.

## 3. Mark a balloon and render

In `my-study/regions.json`, replace the example view ID with `page-one`. Set `expected_color_member` to the selected color member's filename, including its extension. Both rectangles use native **color-page coordinates**. Each is `[left, top, right, bottom]`, with right and bottom excluded.

- `erase` encloses the original lettering and a small clean paper margin.
- `ink` encloses the English lettering as projected onto the color page. Leave room for punctuation and antialiased edges.
- `retained` records lettering the page still contains. A partial page should remain identifiable.

Keep both rectangles inside the balloon. The renderer can hold an ordinary text region when it encounters clipped ink, a nonwhite perimeter or artwork underneath the proposed placement. [Example annotation](../examples/regions.json)

```powershell
.\.venv\Scripts\python.exe -m manga_localizer.render my-study/pilot.json my-study/regions.json --root workspace
.\.venv\Scripts\python.exe -m manga_localizer.verify my-study/pilot.json my-study/regions.json --root workspace
```

Open the native PNG under `workspace/output/pilot/`. Under `workspace/review/pilot/`, inspect the comparison, authorization mask, difference overlay and per-region crops. Check `results.json` for applied and held regions. The verification report records the saved file hashes and protected-pixel counts. A passing pixel check establishes that edits stayed inside their authorized geometry. Artwork and lettering inside that geometry still need visual inspection.
