import os
import cv2
import pandas as pd
import numpy as np

def generate_processed_path(
        image_path,
        raw_dir,
        processed_dir
):

    relative_path = os.path.relpath(
        image_path,
        raw_dir
    )

    return os.path.join(
        processed_dir,
        relative_path
    )

def flip_image(image):
    """
    Horizontally flip image.
    """
    return cv2.flip(image, 1)

def resize_image(
        image,
        target_size
):
    """
    Resize image to target dimensions.
    """

    return cv2.resize(
        image,
        target_size,
        interpolation=cv2.INTER_AREA
    )

def correct_illumination(image):
    """
    Correct uneven illumination using Gaussian background division
    (retinex-inspired), shade correction, and adaptive illumination
    normalization on the L channel of LAB color space.
    Only the circular fundus region is corrected; the black background
    is masked out and preserved as black throughout.
    """

    # Mask the circular fundus region - exclude black background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    # Estimate background illumination via large Gaussian blur
    background = cv2.GaussianBlur(
        l_channel,
        (101, 101),
        sigmaX=0
    ).astype(np.float32)

    l_float = l_channel.astype(np.float32)

    # Mean brightness computed only within fundus region (excludes black border)
    mean_brightness = np.mean(l_float[mask > 0])

    # Division-based normalization - preserves local contrast
    l_corrected = (l_float / (background + 1e-6)) * mean_brightness

    l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)

    # Adaptive illumination normalization
    l_normalized = cv2.normalize(
        l_corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Apply correction only within fundus region; force background to black
    l_result = np.zeros_like(l_channel)

    l_result[mask > 0] = l_normalized[mask > 0]

    lab = cv2.merge((l_result, a_channel, b_channel))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_contrast(image):
    """
    Enhance contrast using CLAHE on the L channel of LAB color space.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_image(image):
    """
    Normalize image to range 0-1.
    """

    image = image.astype(np.float32)

    return image / 255.0

def save_image(
        image,
        output_path,
        normalized=False
):
    """
    Save image.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    if normalized:

        image = (image * 255).astype(np.uint8)

    cv2.imwrite(
        output_path,
        image
    )


def process_image(
        image_path,
        output_path,        
        target_size,
        blur_threshold,
        bright_region_quadrant,
        needs_resize,
        needs_illumination_correction,
        needs_contrast_enhancement,
        needs_normalization,
):
    """
    Apply preprocessing steps to image.
    """

    image = cv2.imread(image_path)

    if image is None:
        return "FAILED"

    # -----------------------------------------
    # Flip if bright spot not in right side
    # -----------------------------------------

    if bright_region_quadrant not in ["Q2", "Q4"]:

        image = flip_image(image)

    # -----------------------------------------
    # Resize
    # -----------------------------------------

    if needs_resize:

        image = resize_image(
            image,
            target_size=target_size
        )

    # -----------------------------------------
    # Illumination Correction
    # -----------------------------------------

    if needs_illumination_correction:

        image = correct_illumination(image)

    # -----------------------------------------
    # Contrast Enhancement
    # -----------------------------------------

    if needs_contrast_enhancement:

        image = enhance_contrast(image)

    # -----------------------------------------
    # Normalize
    # -----------------------------------------

    normalized = False

    if needs_normalization:

        image = normalize_image(image)

        normalized = True

    save_image(
        image,
        output_path,
        normalized=normalized
    )

    return "PROCESSED"


# ----------------------------------------------------
# PROCESS DATASET
# ----------------------------------------------------

def preprocess_dataset(
        input_csv,
        raw_dir,
        processed_dir,
        target_size,
        blur_threshold
):

    df = pd.read_csv(input_csv)

    log_records = []

    processed_count = 0
    skipped_count = 0

    for _, row in df.iterrows():

        image_path = row["file_path"]

        # Skip blurry images
        if row["is_blurry"]:

            skipped_count += 1

            log_records.append({
                "file_path": image_path,
                "status": "SKIPPED_BLURRY"
            })

            continue

        output_path = generate_processed_path(
            image_path,
            raw_dir,
            processed_dir
        )

        status = process_image(
            image_path=image_path,
            output_path=output_path,
            target_size=target_size,
            blur_threshold=blur_threshold,
            bright_region_quadrant=row["bright_region_quadrant"],
            needs_resize=row["needs_resize"],
            needs_illumination_correction=row["needs_illumination_correction"],
            needs_contrast_enhancement=row["needs_contrast_enhancement"],
            needs_normalization=row["needs_normalization"]
        )

        processed_count += 1

        log_records.append({

            "file_path":
                image_path,

            "output_path":
                output_path,

            "status":
                status,

            "flipped":
                row["bright_region_quadrant"]
                not in ["Q2", "Q4"],

            "resized":
                row["needs_resize"],

            "illumination_corrected":
                row["needs_illumination_correction"],

            "contrast_enhanced":
                row["needs_contrast_enhancement"],

            "normalized":
                row["needs_normalization"]
        })

    report_df = pd.DataFrame(
        log_records
    )

    report_df.to_csv(
        "preprocessing_report.csv",
        index=False
    )

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nProcessed Images : {processed_count}")
    print(f"Skipped Images   : {skipped_count}")
    print("Report Saved     : preprocessing_report.csv")

    return report_df
