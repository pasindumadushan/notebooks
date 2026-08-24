import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np


def detect_bright_region(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    threshold = np.percentile(gray, 99)

    _, thresh = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    h, w = gray.shape

    if cx < w/2 and cy < h/2:
        quadrant = "Q1"
    elif cx >= w/2 and cy < h/2:
        quadrant = "Q2"
    elif cx < w/2 and cy >= h/2:
        quadrant = "Q3"
    else:
        quadrant = "Q4"

    return cx, cy, quadrant


def calculate_illumination_score(image_path):
    """
    Calculate illumination uniformity score using std dev of Gaussian background
    on the L channel of LAB color space.
    Higher value = more uneven illumination.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, _, _ = cv2.split(lab)

    background = cv2.GaussianBlur(
        l_channel,
        (101, 101),
        sigmaX=0
    )

    return float(background.std())


def calculate_contrast_score(image_path):
    """
    Calculate contrast score using standard deviation of grayscale pixel intensity.
    Lower value = lower contrast.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(gray.std())


def calculate_blur_score(image_path):
    """
    Calculate blur score using Variance of Laplacian.
    Lower value = blurrier image.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def analyze_dataset(
        output_csv,
        raw_dir,
        target_size,
        blur_threshold,
        contrast_threshold,
        illumination_threshold
):
    """
    Analyze dataset and generate CSV report.
    """

    records = []
    total_images = 0

    for root, dirs, files in os.walk(raw_dir):

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            img_path = os.path.join(root, file)

            try:

                with Image.open(img_path) as img:

                    width, height = img.size

                    img_array = np.array(img)

                    min_pixel = int(
                        img_array.min()
                    )

                    max_pixel = int(
                        img_array.max()
                    )

                    class_label = os.path.basename(
                        os.path.dirname(img_path)
                    )

                    # Blur analysis
                    blur_score = calculate_blur_score(
                        img_path
                    )

                    is_blurry = (
                        blur_score is not None and
                        blur_score < blur_threshold
                    )

                    # Contrast analysis
                    contrast_score = calculate_contrast_score(
                        img_path
                    )

                    needs_contrast_enhancement = (
                        contrast_score is not None and
                        contrast_score < contrast_threshold
                    )

                    # Illumination analysis
                    illumination_score = calculate_illumination_score(
                        img_path
                    )

                    needs_illumination_correction = (
                        illumination_score is not None and
                        illumination_score > illumination_threshold
                    )

                    # Bright region
                    bright_result = detect_bright_region(img_path)

                    if bright_result is not None:
                        bright_x, bright_y, bright_quadrant = bright_result
                    else:
                        bright_x, bright_y, bright_quadrant = None, None, None

                    # Resize check
                    is_resized = (
                        (width, height)
                        == target_size
                    )

                    # Normalization check
                    is_normalized = (
                        max_pixel <= 1
                    )

                    # Recommendations
                    needs_resize = (
                        not is_resized
                    )

                    needs_normalization = (
                        not is_normalized
                    )

                    records.append({

                        "file_path":
                            img_path,

                        "class_label":
                            class_label,

                        "width":
                            width,

                        "height":
                            height,

                        "min_pixel":
                            min_pixel,

                        "max_pixel":
                            max_pixel,

                        "blur_score":
                            round(blur_score, 2)
                            if blur_score is not None
                            else None,

                        "is_blurry":
                            is_blurry,

                        "bright_region_x":
                            bright_x,

                        "bright_region_y":
                            bright_y,

                        "bright_region_quadrant":
                            bright_quadrant,
                            
                        "contrast_score":
                            round(contrast_score, 2)
                            if contrast_score is not None
                            else None,

                        "illumination_score":
                            round(illumination_score, 2)
                            if illumination_score is not None
                            else None,

                        "needs_resize":
                            needs_resize,

                        "needs_illumination_correction":
                            needs_illumination_correction,

                        "needs_contrast_enhancement":
                            needs_contrast_enhancement,

                        "needs_normalization":
                            needs_normalization,

                        "recommended_action":
                            "REMOVE_BLURRY"
                            if is_blurry
                            else "PROCESS"

                    })

                    total_images += 1

            except Exception as e:

                records.append({
                    "file_path": img_path,
                    "error": str(e)
                })

    df = pd.DataFrame(records)

    df.to_csv(
        output_csv,
        index=False
    )

    print("=" * 60)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nTotal Images: {total_images}")
    print(f"CSV Saved To: {output_csv}")

    print(
        f"\nImages Requiring Resize: "
        f"{df['needs_resize'].sum()}"
    )

    print(
        f"Images Requiring Normalization: "
        f"{df['needs_normalization'].sum()}"
    )

    print(
        f"Blurry Images: "
        f"{df['is_blurry'].sum()}"
    )

    print(
        f"Images Requiring Contrast Enhancement: "
        f"{df['needs_contrast_enhancement'].sum()}"
    )

    print(
        f"Images Requiring Illumination Correction: "
        f"{df['needs_illumination_correction'].sum()}"
    )

    return df