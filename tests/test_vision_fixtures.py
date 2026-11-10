import io
import numpy as np
from PIL import Image
from scarx.evaluation.vision_fixtures import scenes,render,expected,eligible,image_uri
from scarx.app.pilots import PilotLab
from scarx.neural.input_events import InputEvents


def test_rendered_public_truth_and_actual_encoder_colorcollision_rgb_preservation():
    fixtures=scenes();assert len(fixtures)==12
    kinds=('single_color','single_shape','object_count','left_color')
    assert sum(8*sum(eligible(s,k) for k in kinds) for s in fixtures)==240
    assert expected(fixtures[0],'single_color','vi')=='đỏ' and expected(fixtures[10],'object_count','en')=='3'
    for i in range(0,12,2):
        first=render(fixtures[i]);second=render(fixtures[i+1]);assert first!=second
        a,_=PilotLab.pixels(image_uri(first));b,_=PilotLab.pixels(image_uri(second));np.testing.assert_array_equal(a,b)
        e=InputEvents(22);assert not np.array_equal(e.image(first),e.image(second))
        assert len(first)<10000 and Image.open(io.BytesIO(first)).size==(128,128)
