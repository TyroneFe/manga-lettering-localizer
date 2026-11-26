"""Run a complete one-page study on two original synthetic editions."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from manga_localizer.chapter import load_json
from manga_localizer.inventory import atomic_bytes, inventory
from manga_localizer.pilot import contact, prepare
from manga_localizer.render import render
from manga_localizer.verify import verify


HERE = Path(__file__).resolve().parent


def make_editions():
    """Draw the equivalent illustration once, then supply different source lettering."""
    canvas = Image.new("RGB", (420, 290), "#b7e3de")
    draw = ImageDraw.Draw(canvas)
    ink = "#1b3540"
    draw.ellipse((280, 16, 353, 89), fill="#f2ac59")
    draw.polygon([(0, 172), (103, 85), (177, 149), (238, 102), (348, 181), (420, 138),
                  (420, 290), (0, 290)], fill="#709eaa", outline=ink, width=3)
    draw.polygon([(0, 198), (98, 146), (181, 192), (295, 134), (420, 190), (420, 290),
                  (0, 290)], fill="#426a72", outline=ink, width=3)
    draw.polygon([(0, 251), (420, 211), (420, 290), (0, 290)], fill="#d6ac76", outline=ink, width=3)
    for x in range(4, 420, 19):
        y = 257 - x // 11
        draw.line((x, y+3, x+7, y), fill="#937451", width=2)
    draw.line((286, 156, 344, 82), fill=ink, width=5)
    draw.ellipse((341, 78, 348, 85), fill=ink)
    draw.rounded_rectangle((129, 149, 324, 254), radius=13, fill="#dc8153", outline=ink, width=5)
    draw.rounded_rectangle((142, 162, 246, 240), radius=9, fill="#eacb91", outline=ink, width=3)
    for y in range(173, 236, 9):
        for x in range(154+(y % 2)*4, 236, 9):
            draw.ellipse((x-2, y-2, x+2, y+2), fill=ink)
    draw.rectangle((255, 164, 311, 181), fill="#e9e7c0", outline=ink, width=2)
    for x in range(262, 309, 6):
        draw.line((x, 169, x, 173+(x % 4)), fill=ink)
    draw.line((282, 166, 282, 179), fill="#b04736", width=2)
    draw.ellipse((268, 192, 300, 224), fill=ink)
    draw.line((284, 205, 290, 197), fill="#eacb91", width=3)
    draw.polygon([(124, 90), (161, 128), (153, 91)], fill="white", outline=ink, width=3)
    draw.ellipse((18, 15, 205, 105), fill="white", outline=ink, width=3)
    draw.rectangle((1, 1, 418, 288), outline=ink, width=3)

    color = canvas.copy()
    source = ImageDraw.Draw(color)
    for stroke in [[(64, 44), (84, 44)], [(74, 45), (74, 65)], [(60, 66), (87, 66)],
                   [(95, 44), (116, 44), (116, 66), (95, 66)], [(127, 55), (151, 55)]]:
        source.line(stroke, fill="black", width=4)
    donor = ImageOps.grayscale(canvas).convert("RGB")
    ImageDraw.Draw(donor).text((110, 56), "ECHO!", fill="black", anchor="mm",
                               font=ImageFont.load_default(size=29))
    return color, donor


def write_archive(path, page):
    png = BytesIO()
    page.save(png, "PNG")
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr(ZipInfo("001.png"), png.getvalue())
    atomic_bytes(path, payload.getvalue())


def main():
    root = HERE / "demo-output"
    study = HERE / "examples"
    config = load_json(study / "pilot.json")
    color, donor = make_editions()
    color_dir, donor_dir = root / "source/color", root / "source/english"
    write_archive(color_dir / "radio.cbz", color)
    write_archive(donor_dir / "radio.cbz", donor)
    inventory([color_dir, donor_dir], root / "cache/inventory.json")
    prepare(root, config)
    reports = render(root, config, load_json(study / "regions.json"), study)
    if reports[0]["regions"][0]["status"] != "applied":
        raise RuntimeError("The demo lettering was held for review")
    sealed = verify(root, study / "pilot.json", study / "regions.json", study_dir=study)
    master = root / "output/pilot" / sealed["masters"][0]["output"]
    with Image.open(master) as output:
        contact(root / "comparison.png", [("Color source", np.asarray(color)),
                ("English donor", np.asarray(donor)), ("Verified result", np.asarray(output))])
    print(f"Master: {master}\nComparison: {root / 'comparison.png'}")
    print(f"Changed pixels outside approved regions: {sealed['masters'][0]['outside_changed']}")


if __name__ == "__main__":
    main()
