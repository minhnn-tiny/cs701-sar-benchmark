"""check_submission.py on a good zip (written by make_submission.write_submission) and on broken ones."""
import json
import zipfile

import pytest

from check_submission import check
from make_submission import write_submission

IMAGES = [{'id': 1, 'file_name': 'images/00001.jpg', 'width': 512, 'height': 512},
          {'id': 2, 'file_name': 'images/00002.png', 'width': 256, 'height': 256},
          {'id': 3, 'file_name': 'images/00003.jpg', 'width': 300, 'height': 200}]


@pytest.fixture
def good_zip(tmp_path):
    """Boxes XYXY on the 512 input, classes 0..8, as predict.py hands them to write_submission."""
    preds = {'images/00001.jpg': (0, [[10, 20, 30, 40], [0, 0, 512, 512]], [0, 0], [0.9, 0.05]),
             'images/00002.png': (8, [[100, 100, 104, 106]], [8], [1.0]),
             'images/00003.jpg': (4, [], [], [])}  # an image without detections is valid
    out = tmp_path / 'good.zip'
    write_submission(IMAGES, preds, out=out)
    return out


def rewrite(src, dst, classification=None, detections=None, extra=None):
    """A copy of the zip src with classification.csv and/or detection.json replaced, or an extra file added."""
    with zipfile.ZipFile(src) as z:
        files = {name: z.read(name) for name in z.namelist()}
    if classification is not None:
        files['classification.csv'] = classification.encode()
    if detections is not None:
        files['detection.json'] = json.dumps(detections).encode()
    if extra:
        files[extra] = b''
    with zipfile.ZipFile(dst, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    return dst


def detections_of(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read('detection.json'))


def test_a_zip_written_by_write_submission_passes(good_zip):
    problems, summary = check(good_zip, IMAGES)
    assert problems == {}
    assert summary['images'] == summary['classification rows'] == 3 and summary['detections'] == 3
    full_image = detections_of(good_zip)[1]['bbox']  # [0, 0, 512, 512] on the input -> the whole original image
    assert full_image == [0, 0, 512, 512]


@pytest.mark.parametrize('classification, expected', [
    ('image_id,label\n1,0\n2,8\n', 'an image of images.json has no classification row'),
    ('image_id,label\n1,0\n2,8\n3,4\n3,4\n', 'an image has more than one classification row'),
    ('image_id,label\n1,0\n2,9\n3,4\n', 'classification.csv label outside 0-8'),
    ('image_id,label\n1,0\n2,8\n7,4\n', 'classification.csv has an id that is not in images.json'),
    ('id,label\n1,0\n2,8\n3,4\n', 'classification.csv header must be image_id,label'),
    ('image_id,label\n1,0\n2,ship\n3,4\n', 'classification.csv row is not two integers'),
])
def test_broken_classification_is_caught(good_zip, tmp_path, classification, expected):
    problems, _ = check(rewrite(good_zip, tmp_path / 'bad.zip', classification=classification), IMAGES)
    assert expected in problems


@pytest.mark.parametrize('change, expected', [
    ({'category_id': 0}, 'a detection has a category_id outside 1-9'),
    ({'category_id': 10}, 'a detection has a category_id outside 1-9'),
    ({'score': 1.5}, 'a detection score is not a finite number in [0, 1]'),
    ({'score': float('nan')}, 'a detection score is not a finite number in [0, 1]'),
    ({'bbox': [500, 10, 20, 20]}, 'a bbox [x, y, w, h] is not inside its image (original pixels)'),  # x + w > 512
    ({'bbox': [10, 10, 0, 5]}, 'a bbox has w or h <= 0'),
    ({'bbox': [10, 10, 5]}, 'a bbox is not 4 finite numbers'),
    ({'image_id': 99}, 'a detection has an image_id that is not in images.json'),
    ({'extra': 1}, "a detection must have exactly the keys ['bbox', 'category_id', 'image_id', 'score']"),
])
def test_broken_detections_are_caught(good_zip, tmp_path, change, expected):
    detections = detections_of(good_zip)
    detections[0].update(change)
    problems, _ = check(rewrite(good_zip, tmp_path / 'bad.zip', detections=detections), IMAGES)
    assert expected in problems


def test_a_box_outside_a_non_square_image_is_caught(good_zip, tmp_path):
    """Image 3 is 300 x 200: a box in 512-input pixels that was not rescaled ends below the image."""
    detections = detections_of(good_zip) + [{'image_id': 3, 'category_id': 5, 'bbox': [10, 150, 20, 60],
                                             'score': 0.5}]
    problems, _ = check(rewrite(good_zip, tmp_path / 'bad.zip', detections=detections), IMAGES)
    assert 'a bbox [x, y, w, h] is not inside its image (original pixels)' in problems


def test_an_extra_file_in_the_zip_is_caught(good_zip, tmp_path):
    problems, _ = check(rewrite(good_zip, tmp_path / 'bad.zip', extra='notes.txt'), IMAGES)
    assert 'the zip must hold exactly classification.csv and detection.json' in problems


def test_predict_deflates_the_zip_and_keeps_its_files(good_zip):
    """write_submission stores the files; predict.py compresses them, to stay under the leaderboard's 25 MB."""
    from predict import deflate
    with zipfile.ZipFile(good_zip) as z:
        files = {name: z.read(name) for name in z.namelist()}
    deflate(good_zip)
    with zipfile.ZipFile(good_zip) as z:
        assert {name: z.read(name) for name in z.namelist()} == files
        assert {info.compress_type for info in z.infolist()} == {zipfile.ZIP_DEFLATED}
    assert check(good_zip, IMAGES)[0] == {}


def test_a_zip_over_the_size_limit_is_caught(good_zip):
    """The leaderboard refuses a zip over 25 MB; a limit below the good zip's size stands in for a too-large zip."""
    problems, _ = check(good_zip, IMAGES, max_bytes=100)
    assert set(problems) == {'the zip is larger than the leaderboard accepts'}
