# Contributing

Chapter 1 visual acceptance is the current milestone. The maintainer reviews the private manga study. Anyone can help test the engine with the public radio demo or with original synthetic fixtures. [Project status and priorities](README.md#what-needs-doing)

## Test the demo on Windows

1. Follow the [Windows setup](README.md#try-it-on-windows) in a fresh checkout. If a step fails, record the command and its first error message.
2. Open the generated comparison. The result should read **ECHO!** with the same punctuation and lettering as the black-and-white donor. Inspect the native master at 100% for old strokes or a damaged balloon outline.
3. Report your Windows and Python versions, the commit or release you tried, the final pixel-check line, and what you saw. For a visible defect, name the feature and its location. The original radio demo is suitable for public screenshots. Keep manga pages, extracted lettering, archives and personal filesystem paths out of public reports. [Demo report](https://github.com/TyroneFe/manga-lettering-localizer/issues/new?template=demo-report.md)

## Work on an open task

- **Texture repairs.** Start with the outlined-letter test and the mirrored tile branch. The issue describes the fixture and the pixel assertions. [Issue #2](https://github.com/TyroneFe/manga-lettering-localizer/issues/2) · [Effect tests](tests/test_effects.py) · [Renderer for effects](manga_localizer/effects.py)
- **Page matching.** Create original pages with a known answer, then run the matcher without giving it the answer. Report how often it picks the correct page and what happens when no match exists. [Issue #3](https://github.com/TyroneFe/manga-lettering-localizer/issues/3) · [Pilot matcher](manga_localizer/pilot.py) · [Registration](manga_localizer/registration.py)
- **Documentation.** If the one-page walkthrough leaves you unsure which coordinates or image to use, point to the exact step and describe what you expected. A wording fix or small illustrated example based on the radio drawing makes a good pull request. [One-page walkthrough](docs/one-page-study.md)

For a code change, fork the repository and work on a branch. Use generated or original artwork you can share. Run the synthetic suite from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

In the pull request, link the task, describe the case you exercised, and give the command and result. For a rendering change, include a native-size synthetic comparison and the number of changed pixels outside the approved area. For a matching change, include a case where the tool should send the pair for review.

## Code map

| Area | Entry points |
| --- | --- |
| Source inventory and page alignment | [Inventory](manga_localizer/inventory.py) · [Registration](manga_localizer/registration.py) |
| Balloon masks, paper and over-art effects | [Lettering](manga_localizer/lettering.py) · [Paper repair](manga_localizer/paper.py) · [Effects](manga_localizer/effects.py) |
| Master PNGs and checks | [Render](manga_localizer/render.py) · [Verify](manga_localizer/verify.py) · [Reading index](manga_localizer/reading_index.py) |
| Demo inputs and regression fixtures | [Example regions](examples/regions.json) · [Tests](tests/) |
