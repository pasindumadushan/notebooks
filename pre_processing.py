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


def create_split_directories(processed_dir, class_labels=None):
    """Create processed/train, processed/val, processed/test and class folders 0..4."""
    if class_labels is None:
        class_labels = [str(i) for i in range(5)]

    os.makedirs(processed_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for class_name in class_labels:
            os.makedirs(
                os.path.join(split_dir, str(class_name)),
                exist_ok=True
            )


def get_source_split_name(image_path, raw_dir):
    """Return the original raw split folder name when the dataset is already organized by train/val/test."""
    try:
        relative_path = os.path.relpath(image_path, raw_dir)
        first_part = relative_path.split(os.sep)[0]
        if first_part.lower() in {"train", "val", "test"}:
            return first_part.lower()
    except Exception:
        pass
    return "train"


def flip_image(image):
    """
    Horizontally flip image.
    """
    return cv2.flip(image, 1)


def create_fundus_mask(image_bgr):
    """Create a circular mask for the retinal fundus area."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    (x, y), radius = cv2.minEnclosingCircle(largest)
    radius = max(int(radius * 0.9), 1)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    return mask


def crop_fundus_roi(image_bgr):
    """Remove black borders and non-retina background while keeping a black outside area."""
    mask = create_fundus_mask(image_bgr)
    if mask is None:
        return image_bgr

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return image_bgr

    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())

    pad_x = max(10, int((x_max - x_min) * 0.08))
    pad_y = max(10, int((y_max - y_min) * 0.08))

    x_min = max(0, x_min - pad_x)
    x_max = min(image_bgr.shape[1], x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(image_bgr.shape[0], y_max + pad_y)

    cropped = np.zeros_like(image_bgr)
    cropped[y_min:y_max, x_min:x_max] = image_bgr[y_min:y_max, x_min:x_max]

    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    radius = max((x_max - x_min), (y_max - y_min)) // 2

    circular = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(circular, (center_x, center_y), radius, 255, -1)
    circular = cv2.bitwise_and(circular, mask)

    result = np.zeros_like(image_bgr)
    result[circular > 0] = cropped[circular > 0]
    return result


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

def ben_graham_enhancement(image):
    """
    Ben Graham style retinal enhancement.
    This emphasizes lesion-like structures by subtracting a blurred background
    and boosting local contrast while preserving the black background.
    """
    if image is None:
        return None

    image = crop_fundus_roi(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=gray.shape[1] / 30)
    enhanced = cv2.addWeighted(gray, 4, blur, -4, 128)
    enhanced = cv2.convertScaleAbs(enhanced)

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


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
        bright_region_quadrant,
        needs_resize,
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
    # Ben Graham style enhancement for retinal lesions
    # -----------------------------------------

    image = ben_graham_enhancement(image)

    # -----------------------------------------
    # Resize
    # -----------------------------------------

    if needs_resize:

        image = resize_image(
            image,
            target_size=target_size
        )

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
):

    # -----------------------------------------
    # Reset processed tree and create required split folders
    # -----------------------------------------

    if os.path.exists(processed_dir):
        import shutil
        shutil.rmtree(processed_dir)

    class_labels = [str(i) for i in range(5)]
    create_split_directories(processed_dir, class_labels=class_labels)

    df = pd.read_csv(input_csv)

    log_records = []

    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():

        image_path = row["file_path"]
        class_name = str(row.get("class_label", os.path.basename(os.path.dirname(image_path))))
        split_name = get_source_split_name(image_path, raw_dir)

        # Skip blurry images already identified during dataset analysis
        if row["is_blurry"]:
            skipped_count += 1
            log_records.append({
                "file_path": image_path,
                "status": "SKIPPED_BLURRY",
                "split": "skipped",
                "class_label": class_name,
            })
            continue

        output_path = os.path.join(
            processed_dir,
            split_name,
            class_name,
            os.path.basename(image_path)
        )

        status = process_image(
            image_path=image_path,
            output_path=output_path,
            target_size=target_size,
            bright_region_quadrant=row["bright_region_quadrant"],
            needs_resize=row["needs_resize"],
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

            "split":
                split_name,

            "class_label":
                class_name,

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

    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_df.to_csv(
        os.path.join(reports_dir, "preprocessing_report.csv"),
        index=False
    )

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nProcessed Images : {processed_count}")
    print(f"Skipped Images   : {skipped_count}")
    print(f"Report Saved     : {os.path.join(reports_dir, 'preprocessing_report.csv')}")

    return report_df
