# Manga Lettering Localizer

This Windows tool transfers the English lettering from a black-and-white manga edition onto the matching official color artwork. It keeps the color page at its original resolution and takes the letter shapes, punctuation and emphasis from the English page. The current prototype runs locally on the CPU.

**The next milestone is a visually approved first chapter of Gantz.** Five pilot scenes passed human review. All 40 story canvases in chapter 1 have draft PNGs, with 231 lettering groups transferred and one split spread in reading order. The reviewer chose to omit the English-only platform announcement on page 20. The chapter still needs lettering repairs and a full-page review before a volume trial can begin. [Chapter 1 checklist](https://github.com/TyroneFe/manga-lettering-localizer/issues/1)

## What needs doing

| When | Work | What will show it is ready |
| --- | --- | --- |
| Now | Finish chapter 1 lettering. All six running effects on page 20 are in English. The lower lifting sound still crosses detailed coat folds. Page 33 has a train screech across the platform. | Check each effect against the official English donor at native size. Record any unresolved effect. Inspect every dialogue balloon for remaining Japanese text. [Chapter checklist](https://github.com/TyroneFe/manga-lettering-localizer/issues/1) |
| Now | Inspect the repaired floor lines on page 29, the two upper captions on page 31 and the six groups flagged by the residual-ink audit. | Record the native-size visual check for each item and correct any remaining strokes or seams. |
| In parallel | Test lettering repairs over patterned artwork. | An original striped fixture catches a reflected seam that bends the wrong way and rejects sampling from erased pixels. [Texture repair task](https://github.com/TyroneFe/manga-lettering-localizer/issues/2) |
| In parallel | Measure when page matching, alignment and balloon detection need a reviewer. | Measure wrong matches, missed regions and cases held for review. Start with repeated art, missing counterparts and split spreads. [Page-matching benchmark](https://github.com/TyroneFe/manga-lettering-localizer/issues/3) |
| After chapter approval | Run and review one volume before attempting a collection. | Account for every source page, reproduce the results after an interrupted run and verify reading order, spreads and color coverage. |
| Later | Build verified CBZ export and the Windows review editor. | Reopen the CBZ and compare every page with its approved PNG. In the editor, correct a mask, undo the edit, save and reopen the project, then resume the run. |

## What works today

- **Import and alignment.** The engine inventories and hashes the originals, proposes matching pages and estimates how to align the English donor with the color artwork. An operator reviews uncertain matches and lettering regions.
- **Lettering and PNG output.** The renderer handles flat balloons, reviewed captions and selected outlined effects. It uses separate masks for removing old ink and adding English lettering. The chapter verifier reopened 35 chapter renders and checked all 37 original archive hashes. It found zero changed pixels outside authorized regions. The reading index includes those 35 renders and five pilot masters.
- **Recovery and testing.** The chapter runner passed an interruption and restart test. The public release passed 43 synthetic regression tests. The example below exercises the transfer and saved-PNG checks using original artwork anyone can share.

## How to help

**Without coding:** run the Windows demo below and inspect the balloon at 100%. A report that pinpoints a clipped letter, stray stroke or confusing setup step is enough to start investigating. [Report a demo result](https://github.com/TyroneFe/manga-lettering-localizer/issues/new?template=demo-report.md)

**With Python or image-processing experience:** the texture repair test is a small first contribution. The matching benchmark is a larger task. Both use original synthetic artwork and can be developed offline. [Contribution guide](CONTRIBUTING.md) · [Open tasks](https://github.com/TyroneFe/manga-lettering-localizer/issues)

## Try it on Windows

Use 64-bit Python 3.12. From the repository directory, run these commands in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe demo.py
```

The demo generates two miniature CBZs and writes `demo-output/comparison.png`. Compare the exclamation mark in the English donor with the one in the result. Open `demo-output/output/pilot/001-radio.png` to inspect the native master. The last console line should read `Changed pixels outside approved regions: 0`.

![Original color radio drawing, black-and-white English donor, and localized radio with ECHO! in the balloon](docs/radio-comparison.png)

To try your own editions, start with one known matching pair. The walkthrough covers inventory, registration, region coordinates and saved-PNG verification. [Localize a first page](docs/one-page-study.md)
