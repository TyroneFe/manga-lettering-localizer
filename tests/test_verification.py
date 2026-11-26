import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from manga_localizer.inventory import atomic_bytes
from manga_localizer.pilot import save_png
from manga_localizer.render import render
from manga_localizer.verify import verify


def test_readback_rejects_damage_and_unaccounted_pages(tmp_path):
    root=tmp_path/"project"
    (root/"source").mkdir(parents=True)
    cfg={"color_archive":"color.cbz","english_archive":"english.cbz"}
    sections={"pair":{"expected_color_member":"01.png","retained":[],"regions":[
        {"label":"flat","erase":[6,6,22,30],"ink":[27,6,47,30],
         "paper_marks":[[17,12,21,18]]}]}}
    original=np.full((60,80,3),255,np.uint8)
    english=original.copy()
    original[13:25,11:15]=0
    original[15,19]=239
    english[13:25,30:35]=0
    for path,data in (("color.cbz",original),("english.cbz",english)):
        buf=BytesIO()
        Image.fromarray(data).save(buf,format="PNG")
        with ZipFile(root/"source"/path,"w") as z:
            z.writestr("01.png",buf.getvalue())
    pilot=root/"review"/"pilot"
    for path,data in [(pilot/"registration.json",[dict(id="pair",english="01.png",member="01.png",matrix=[[1.,0,0],[0,1.,0]])]),
                      (tmp_path/"config.json",cfg),(tmp_path/"annotations.json",sections)]:
        atomic_bytes(path,json.dumps(data).encode())
    render(root,cfg,sections)
    args=(root,tmp_path/"config.json",tmp_path/"annotations.json")
    clean=verify(*args,check_sources=False)
    assert clean["masters"][0]["changed"]>0
    master=root/"output"/"pilot"/"01-pair.png"
    with Image.open(master) as im:
        damaged=np.array(im)
    # A mark inside the erasure window stays protected after an independent readback.
    assert np.all(damaged[15,19] == 239)
    damaged[15,19,0]^=1
    save_png(master,damaged)
    with pytest.raises(ValueError,match="protected"):
        verify(*args,check_sources=False)
    atomic_bytes(pilot/"results.json",b"[]")
    with pytest.raises(ValueError,match="accounting"):
        verify(*args,check_sources=False)
