from zipfile import ZipFile

import pytest

from manga_localizer.chapter import chapter_jobs


def test_chapter_accounts_for_front_matter_and_right_to_left_spread(tmp_path):
    japanese = tmp_path / "color.cbz"
    english = tmp_path / "english.cbz"
    plan = dict(english_selector="c001 - ", first_story_page=7, last_story_page=9,
                color_number_offset=-4, front_matter={"p000": ["001"], "p001": []},
                color_only=["002"])
    with ZipFile(japanese, "w") as archive:
        for n in range(1, 7):
            archive.writestr(f"volume/{n:03d}.jpg", b"")
    with ZipFile(english, "w") as archive:
        for locator in ("p008-p009", "p001", "p007", "p000"):
            archive.writestr(f"c001 - {locator} [English].png", b"")
    jobs = chapter_jobs(japanese, english, plan)
    assert [job[0] for job in jobs] == ["p000", "p001", "p007", "p008-p009", "jp-002"]
    assert jobs[1][2:] == ("english_only", [])
    assert jobs[3][2:] == ("story", ["volume/004.jpg", "volume/005.jpg"])
    assert jobs[4][1:] == (None, "color_only", ["volume/002.jpg"])

    plan["last_story_page"] = 10
    with pytest.raises(ValueError, match="missing or out-of-order"):
        chapter_jobs(japanese, english, plan)
