"""The submission check accepts what make_submission.write_submission writes and rejects broken zips."""
import zipfile
from pathlib import Path

import pytest

from make_submission import write_submission
from predict import check_submission

SAMPLE = Path(__file__).resolve().parents[3] / 'dataset' / 'sample_submission_val.zip'


def test_check_accepts_write_submission_output_and_rejects_broken_zips(tmp_path):
    images = [{'id': 1, 'file_name': 'images/00001.jpg', 'width': 256, 'height': 256},
              {'id': 2, 'file_name': 'images/00002.png', 'width': 512, 'height': 512}]
    preds = {'images/00001.jpg': (7, [[10., 20., 30., 60.]], [7], [0.9]), 'images/00002.png': (0, [], [], [])}
    good = tmp_path / 'good.zip'
    write_submission(images, preds, out=good)
    summary = check_submission(good, images, SAMPLE if SAMPLE.exists() else None)
    assert summary == {'images': 2, 'detections': 1, 'images_with_detections': 1}
    with zipfile.ZipFile(good) as z:
        det = z.read('detection.json')
    assert b'"category_id": 8' in det and b'[5.0, 10.0, 10.0, 20.0]' in det  # label + 1, rescaled to 256 px
    broken = tmp_path / 'broken.zip'
    with zipfile.ZipFile(broken, 'w') as z:
        z.writestr('classification.csv', 'image_id,label\n1,7\n')  # an image missing, no detection.json
    with pytest.raises(AssertionError):
        check_submission(broken, images)
