import seqfishtools
import numpy as np
import tifffile as tif

import itertools
import pytest

DTYPE = np.uint16
DTYPEMIN = np.iinfo(DTYPE).min
DTYPEMAX = np.iinfo(DTYPE).max

RANDIM_SHAPES = [
    (2048, 2048),
    (1, 2048, 2048),
    (2, 2048, 2048),
    (3, 2048, 2048),
    (4, 2048, 2048),
    (5, 2048, 2048),
]

for a, b in itertools.product(range(1, 6), repeat=2):
    RANDIM_SHAPES.append((a, b, 2048, 2048))


@pytest.fixture(params=RANDIM_SHAPES)
def random_nd_image(request):
    return np.random.randint(
        DTYPEMIN,
        DTYPEMAX,
        request.param,
        dtype=DTYPE
    )


def test_hash_read_write_nd_image(tmp_path, random_nd_image):
    random_nd_hashed = hash(random_nd_image.tobytes())
    orig_shape = random_nd_image.shape
    imname = tmp_path / 'hash_im.tif'

    seqfishtools.hash_imwrite(imname, random_nd_image)

    del random_nd_image

    read_nd_image = seqfishtools.hash_imread(imname)

    read_nd_hashed = hash(read_nd_image.tobytes())
    read_shape = read_nd_image.shape

    assert orig_shape == read_shape
    assert random_nd_hashed == read_nd_hashed


def test_tifffile_read_write_nd_image(tmp_path, random_nd_image):
    random_nd_hashed = hash(random_nd_image.tobytes())
    orig_shape = random_nd_image.shape
    imname = tmp_path / 'tif_im.tif'

    tif.imwrite(imname, random_nd_image)

    read_nd_image = tif.imread(imname)

    read_nd_hashed = hash(read_nd_image.tobytes())
    read_shape = read_nd_image.shape

    assert orig_shape == read_shape
    assert random_nd_hashed == read_nd_hashed

    del random_nd_image, read_nd_image