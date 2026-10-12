import cv2
import numpy as np


def load_image(image_path):
    """Load a medical image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image


def gaussian_filter(image, kernel_size=5):
    """Reduce image noise using Gaussian filtering."""
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), 0
    )


def median_filter(image, kernel_size=5):
    """Reduce impulse noise using median filtering."""
    return cv2.medianBlur(image, kernel_size)


def histogram_equalization(image):
    """Improve global image contrast."""
    return cv2.equalizeHist(image)


def clahe_enhancement(image):
    """Enhance local contrast using CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


def thresholding(image):
    """Separate brighter and darker regions."""
    _, result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def edge_detection(image):
    """Detect image boundaries using Canny."""
    return cv2.Canny(image, 100, 200)


def process_image(image):
    """Apply all implemented algorithms."""
    return {
        "Original": image,
        "Gaussian Filter": gaussian_filter(image),
        "Median Filter": median_filter(image),
        "Histogram Equalization": histogram_equalization(image),
        "CLAHE": clahe_enhancement(image),
        "Thresholding": thresholding(image),
        "Edge Detection": edge_detection(image),
    }