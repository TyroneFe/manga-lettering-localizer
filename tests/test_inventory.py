import os
import subprocess
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest
from PIL import Image, ImageCms

from manga_localizer.inventory import file_hash, inventory, read_image, safe_member, scan_file


@pytest.mark.parametrize("failures", [1, 5])
def test_atomic_replace_while_windows_reader_holds_output(tmp_path, monkeypatch, failures):
    import manga_localizer.inventory as inv

    path = tmp_path / "comparison.png"
    path.write_bytes(b"old")
    replace = os.replace
    attempts = []

    def reader_holds_file(temp, destination):
        attempts.append(temp)
        assert destination.read_bytes() == b"old"
        if len(attempts) <= failures:
            error = PermissionError("destination in use")
            error.winerror = 5
            raise error
        replace(temp, destination)

    monkeypatch.setattr(inv.os, "replace", reader_holds_file)
    monkeypatch.setattr(inv, "sleep", lambda delay: None)
    if failures == 1:
        inv.atomic_bytes(path, b"new")
        assert path.read_bytes() == b"new"
        assert len(attempts) == 2
    else:
        with pytest.raises(PermissionError):
            inv.atomic_bytes(path, b"new")
        assert path.read_bytes() == b"old"
        assert len(attempts) == 5
    assert list(tmp_path.iterdir()) == [path]


def picture(fmt="PNG", mode="RGB", **meta):
    f=BytesIO()
    Image.new(mode,(12,17)).save(f,fmt,**meta)
    return f.getvalue()


@pytest.mark.parametrize("name",["../p.jpg","C:/a.png","a/../b.jpg","a/CON.png","a. /b","//host/a","a:x.png"])
def test_unsafe(name):
    assert not safe_member(name)


def test_loose_and_zip(tmp_path):
    src=tmp_path/"files"
    src.mkdir()
    original=picture()
    for name in ("02.png","10.png","2.png","14.png"):
        (src/name).write_bytes(original)
    (src/"sub").mkdir()
    (src/"sub"/"2.png").write_bytes(original)
    (src/"表紙.webp").write_bytes(picture("WEBP"))
    archive=src/"nested.cbz"
    with ZipFile(archive,"w") as z:
        z.writestr("10.jpg",picture("JPEG"))
        z.writestr("2.png",original)
        with pytest.warns(UserWarning):
            z.writestr("2.png",original)
        z.writestr("../escape.png",original)
        z.writestr("truncated.jpg",picture("JPEG")[:-80])
        z.writestr("ComicInfo.xml","<ComicInfo/>")
    hashes={p:file_hash(p) for p in src.rglob("*") if p.is_file()}
    def run():
        return inventory([src],tmp_path/"inventory.json",extract=tmp_path/"extracted")["editions"][0]
    first=run()
    assert first==run()
    assert {p:file_hash(p) for p in hashes}==hashes
    assert [v["source"] for v in first["files"][:4]]==["02.png","2.png","10.png","14.png"]
    arc=next(a for a in first["files"] if a["source"]=="nested.cbz")
    assert sorted(r["status"] for r in arc["pages"])==sorted(["ok"]*3+["unsafe member","non-image","unreadable"])
    assert len(arc["name_collisions"]["2.png"])==2 and first["duplicates"] and first["possible_gaps"]
    assert not (tmp_path/"escape.png").exists()
    assert all(file_hash(f) in f.name for f in (tmp_path/"extracted").rglob("*") if f.is_file())
    with pytest.raises(ValueError):
        inventory([src],src/"result.json")


def test_damaged_sibling_and_limits(tmp_path,monkeypatch):
    import manga_localizer.inventory as inv
    arc=tmp_path/"bad.zip"
    with ZipFile(arc,"w",compression=ZIP_STORED) as z:
        a,b=picture(),picture("JPEG")
        z.writestr("good.png",a)
        z.writestr("bad.jpg",b)
    data=bytearray(arc.read_bytes())
    data[data.index(b)+len(b)//2] ^= 1
    arc.write_bytes(data)
    rec=scan_file(arc,tmp_path)
    assert [p["status"] for p in rec["pages"]]==["unreadable","ok"]
    monkeypatch.setattr(inv,"MAX_MEMBER",1)
    assert all("limit" in p["status"] for p in scan_file(arc,tmp_path)["pages"])
    arc.write_bytes(b"broken")
    assert scan_file(arc,tmp_path)["issues"]


def test_metadata_and_empty(tmp_path):
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    assert read_image(picture(icc_profile=profile),False)["icc_sha256"]
    assert read_image(picture(mode="P",transparency=0),False)["transparency"]
    assert read_image(picture(mode="I;16"),False)["bit_depth"]==16
    exif=Image.Exif()
    exif[274]=6
    assert read_image(picture("JPEG",exif=exif),False)["orientation"]==6
    target=tmp_path/"empty"
    target.mkdir()
    assert inventory([target],tmp_path/"report")["editions"][0]["files"]==[]


def test_compressed_cbr(tmp_path,monkeypatch):
    create=Path(os.environ.get("ProgramFiles","C:/Program Files"))/"WinRAR"/"Rar.exe"
    if not create.is_file():
        pytest.skip("fixture creation needs WinRAR")
    f=tmp_path/"public.png"
    raw=BytesIO()
    Image.new("RGB",(240,330),(91,140,21)).save(raw,"PNG",compress_level=0)
    f.write_bytes(raw.getvalue())
    rar=tmp_path/"example.rar"
    subprocess.run([str(create),"a","-inul","-ep","-m5",str(rar),str(f)],check=True)
    cbr=rar.rename(tmp_path/"example.cbr")
    import rarfile
    with rarfile.RarFile(cbr) as packed:
        assert packed.infolist()[0].compress_size < len(f.read_bytes()) // 2
    result=inventory([cbr],tmp_path/"report")["editions"][0]["files"][0]["pages"]
    assert len(result)==1 and result[0]["status"]=="ok" and result[0]["sha256"]==file_hash(f)
    def missing(*args,**kwargs): raise rarfile.RarCannotExec("backend missing")
    monkeypatch.setattr(rarfile.RarFile,"open",missing)
    assert "backend missing" in scan_file(cbr,tmp_path)["pages"][0]["reason"]
