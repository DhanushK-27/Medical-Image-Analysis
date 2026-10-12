import numpy as np
import pytest

from medical_image import (
    gaussian_filter,
    median_filter,
    histogram_equalization,
    clahe_enhancement,
    thresholding,
    edge_detection,
    process_image,
)


@pytest.fixture
def sample_image():
    """Create a synthetic grayscale image for testing."""
    image = np.zeros((128, 128), dtype=np.uint8)
    image[32:96, 32:96] = 180
    image[48:80, 48:80] = 255
    return image


@pytest.mark.parametrize(
    "algorithm",
    [
        gaussian_filter,
        median_filter,
        histogram_equalization,
        clahe_enhancement,
        thresholding,
        edge_detection,
    ],
)
def test_algorithm_output(algorithm, sample_image):
    result = algorithm(sample_image)

    assert result is not None
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert np.all(result >= 0)
    assert np.all(result <= 255)


def test_gaussian_filter(sample_image):
    result = gaussian_filter(sample_image)

    assert not np.array_equal(result, sample_image)


def test_thresholding_output(sample_image):
    result = thresholding(sample_image)

    assert set(np.unique(result)).issubset({0, 255})


def test_process_image(sample_image):
    results = process_image(sample_image)

    assert len(results) == 7
    assert "Original" in results
    assert "CLAHE" in results
    assert "Edge Detection" in results