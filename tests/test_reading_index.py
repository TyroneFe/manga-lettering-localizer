import numpy as np
import pytest

from manga_localizer.chapter import write_json
from manga_localizer.chapter_lettering import write_lettering_index
from manga_localizer.inventory import file_hash
from manga_localizer.pilot import save_png
from manga_localizer.reading_index import check_index


def test_index_checks_both_spread_halves_current_links_and_saved_hashes(tmp_path):
    root, study = tmp_path / "project", tmp_path / "study"
    chapter = root / "review" / "chapters" / "trial"
    pilot = root / "review" / "pilot"
    output = root / "output" / "chapters" / "trial" / "lettering"
    write_json(study / "chapter.json", dict(id="trial", first_story_page=7,
                                            last_story_page=8, color_number_offset=-4))
    job = dict(id="p007-p008", english="spread.png", kind="story",
               views=[dict(japanese="003.png"), dict(japanese="004.png")])
    write_json(chapter / "report.json", dict(chapter="trial", complete=True, errors={}, pending=[],
                                             total_jobs=1, pages=[job]))

    def edition(review, directory, pairs):
        reports, seals = [], []
        for stem, scene, value in pairs:
            rgb = np.full((8, 9, 3), value, np.uint8)
            name = f"{stem}-{scene}.png"
            save_png(directory / name, rgb)
            save_png(review / f"{scene}-comparison.png", rgb)
            reports.append(dict(id=scene, english="spread.png", japanese=stem+".png",
                                output=name, regions=[], retained=[], changed=0, state="art_only"))
            seals.append(dict(id=scene, output=name, file_sha256=file_hash(directory / name),
                              changed=0, held_changed=0, outside_changed=0))
        write_json(review / "results.json", reports)
        write_json(review / "verification.json", dict(masters=seals))
        return reports

    edition(pilot, root / "output" / "pilot", [("003", "A", 110), ("004", "B", 115)])
    current = edition(chapter / "lettering" / "rendered", output, [("004", "revised", 130)])
    write_lettering_index(root, chapter, output, current)
    args = root, study, "trial"
    assert check_index(*args) == (2, 0, 1)
    index = chapter / "lettering" / "LETTERING.md"
    original = index.read_bytes()
    lines = original.rsplit(b"\n", 3)
    index.write_bytes(b"\n".join([lines[0], lines[2], lines[1], lines[3]]))
    with pytest.raises(ValueError, match="out-of-order"):
        check_index(*args)
    index.write_bytes(original.replace(b"004-revised.png", b"004-B.png"))
    with pytest.raises(ValueError, match="links differ"):
        check_index(*args)
    index.write_bytes(original)
    master = output / "004-revised.png"
    master.write_bytes(master.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        check_index(*args)
    assert index.read_bytes() == original
