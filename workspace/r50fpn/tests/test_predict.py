"""predict.py's compact(): rounded boxes stay inside their image, the zip is compressed and still passes the checks."""
import json
import zipfile

from check_submission import check
from make_submission import write_submission
from predict import compact

IMAGES = [{'id': 1, 'file_name': 'images/00001.jpg', 'width': 256, 'height': 200},
          {'id': 2, 'file_name': 'images/00002.jpg', 'width': 512, 'height': 512}]


def test_compact_keeps_boxes_inside_and_passes_the_checks(tmp_path):
    # XYXY on the 512 input: a box on the right and bottom edges, a box 0.001 px wide, a box with long decimals
    preds = {'images/00001.jpg': (3, [[500.0, 500.0, 512.0, 512.0], [10.0, 10.0, 10.001, 20.0]], [3, 3], [0.9, 0.5]),
             'images/00002.jpg': (0, [[1.234567, 2.345678, 100.987654, 200.123456]], [0], [0.123456789])}
    out = tmp_path / 'sub.zip'
    write_submission(IMAGES, preds, out=out)
    compact(out, IMAGES)

    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ['classification.csv', 'detection.json']
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in z.infolist())
        assert z.read('classification.csv').decode() == 'image_id,label\n1,3\n2,0\n'
        detections = json.loads(z.read('detection.json'))
    assert len(detections) == 2  # the 0.001 px box rounds to zero width and is dropped
    size = {im['id']: (im['width'], im['height']) for im in IMAGES}
    for d in detections:
        x, y, w, h = d['bbox']
        assert 0 <= x and 0 <= y and w > 0 and h > 0
        assert x + w <= size[d['image_id']][0] and y + h <= size[d['image_id']][1]
        assert all(round(v, 2) == v for v in d['bbox'])
    assert detections[1]['score'] == 0.123457
    problems, _ = check(out, IMAGES)
    assert not problems
