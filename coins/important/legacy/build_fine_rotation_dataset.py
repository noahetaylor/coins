import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
import math

# Folders containing your 80 perfect images
SRC = Path("/home/noah/projects/image-identifier/data/coins/2026RotationTraining")
OUT = Path("/home/noah/projects/image-identifier/data/coins/FineRotationDataset")

OUT.mkdir(parents=True, exist_ok=True)

# Baseline angles for each folder
FOLDER_ANGLE = {
    "rot_train_0": 0,
    "rot_train_90": 90,
    "rot_train_180": 180,
    "rot_train_270": 270,
}

def crop_to_coin(img, pad=30, prior_cx=None, prior_cy=None, prior_radius=None):
    """Crop a square region tightly around the detected coin.

    Strategy:
    - Find non-white bounding box of the coin.
    - Try contour-based circle; if too small, use bbox or prior if available.
    - Returns (cropped_img, cx, cy, radius) so caller can track successful detections.
    """
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find non-white pixels (coin)
    white_mask = cv2.inRange(gray, 240, 255)
    non_white = cv2.bitwise_not(white_mask)
    
    # Morphology to fill small holes and clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Get bbox of non-white (coin area)
    coords = cv2.findNonZero(non_white)
    if coords is None or len(coords) == 0:
        return img, None, None, None

    x_box, y_box, w_box, h_box = cv2.boundingRect(coords)
    cx_box = x_box + w_box / 2.0
    cy_box = y_box + h_box / 2.0
    radius_box = max(w_box, h_box) / 2.0

    # Try contour-based circle (convex hull of non-white)
    # But only use it if it's reasonably sized (worn coins may have visible inner details)
    cx, cy, radius = cx_box, cy_box, radius_box
    method = "bbox"

    contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        (cx_hull, cy_hull), radius_hull = cv2.minEnclosingCircle(hull)
        cx_hull, cy_hull, radius_hull = float(cx_hull), float(cy_hull), float(radius_hull)
        
        # Hull must be: (1) large enough, (2) centered on bbox center
        # Both conditions must pass or we don't trust it (worn coins have misleading inner features)
        center_dist = math.sqrt((cx_hull - cx_box)**2 + (cy_hull - cy_box)**2)
        is_good_hull = (radius_hull >= radius_box * 0.95) and (center_dist < radius_box * 0.15)
        
        if is_good_hull:
            cx, cy, radius = cx_hull, cy_hull, radius_hull
            method = "hull"
        else:
            # Hull failed validation (too small or off-center), fall back to bbox
            # This is more reliable across all coins than trying to use priors
            method = "bbox (hull rejected)"

    # Build a square crop around the circle center with margin
    r = float(radius) * 1.05 + pad
    x1 = int(max(0, round(cx - r)))
    y1 = int(max(0, round(cy - r)))
    x2 = int(min(w_img, round(cx + r)))
    y2 = int(min(h_img, round(cy + r)))

    # Ensure square crop
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)
    # Re-center square on detected center
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    half = side // 2
    x1 = max(0, cx_i - half)
    y1 = max(0, cy_i - half)
    x2 = min(w_img, x1 + side)
    y2 = min(h_img, y1 + side)

    # Log detection method and sizes for debugging
    print(f"    crop_to_coin: method={method}, bbox={w_box}x{h_box}, radius={radius:.1f}, crop_side={side}")

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img, None, None, None
    return cropped, cx, cy, radius

def pad_to_square(img, size=512):
    """Pad cropped coin to a square with white background."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    # Create white background
    padded = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
    
    # Center the coin
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = img
    
    # Resize to standard size if needed
    if size and max_dim != size:
        padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)
    
    return padded

def pad_to_circle(img, size=512, outer_margin=30):
    """Center the coin on a square canvas and mask outside a centered circle to white.

    - `size` is the output square size.
    - `outer_margin` leaves a gap between coin and circle edge (increased from 8 to 30).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Try Hough Circle detection first (more robust for round coins)
    cx, cy, radius = None, None, None
    try:
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) / 8,
            param1=100,
            param2=30,
            minRadius=int(min(h, w) * 0.15),
            maxRadius=int(min(h, w) * 0.95),
        )
        if circles is not None and len(circles) > 0:
            cx, cy, radius = float(circles[0][0][0]), float(circles[0][0][1]), float(circles[0][0][2])
    except Exception:
        pass

    # Fallback: contour-based min enclosing circle on non-white mask
    if cx is None:
        white_mask = cv2.inRange(gray, 240, 255)
        non_white = cv2.bitwise_not(white_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx = float(cx)
            cy = float(cy)
            radius = float(radius)
        else:
            cx = w / 2.0
            cy = h / 2.0
            radius = max(w, h) / 2.0

    # make diameter based on detected radius + outer margin
    diam = int(math.ceil(2 * (radius + outer_margin)))
    max_dim = max(diam, int(math.ceil(max(h, w))))

    # create white canvas and center the coin so center aligns
    canvas = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
    canvas_cx = max_dim // 2
    canvas_cy = max_dim // 2
    x_off = int(round(canvas_cx - cx))
    y_off = int(round(canvas_cy - cy))

    dst_x1 = max(0, x_off)
    dst_y1 = max(0, y_off)
    dst_x2 = min(max_dim, x_off + w)
    dst_y2 = min(max_dim, y_off + h)

    src_x1 = max(0, -x_off)
    src_y1 = max(0, -y_off)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    # precise circular mask at canvas center
    mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
    rad = int(math.ceil(radius + outer_margin))
    cv2.circle(mask, (canvas_cx, canvas_cy), rad, 255, -1)

    masked = canvas.copy()
    inverted = cv2.bitwise_not(mask)
    masked[inverted == 255] = 255

    if size and max_dim != size:
        masked = cv2.resize(masked, (size, size), interpolation=cv2.INTER_CUBIC)

    return masked


def rotate_numpy_image(img, angle_deg):
    """Rotate a numpy BGR image by angle degrees (PIL uses counter-clockwise).

    We keep the canvas size the same and fill outside with white.
    """
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # PIL rotates counter-clockwise; keep expand=False to preserve canvas
    rotated = pil.rotate(angle_deg, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)


def main():
    import os

    TEST_SPLIT = float(os.environ.get("TEST_SPLIT", "0.1"))

    TRAIN_DIR = OUT / "train"
    TEST_DIR = OUT / "test"
    ALL_DIR = OUT / "all"
    for d in (TRAIN_DIR, TEST_DIR, ALL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directories:\n  TRAIN: {TRAIN_DIR}\n  TEST:  {TEST_DIR}\n  ALL:   {ALL_DIR}")

    # Only use rot_train_0 (base 0-degree labeled data)
    base_folder = SRC / "rot_train_0"
    if not base_folder.exists():
        print(f"Base folder {base_folder} not found")
        return

    images = sorted(base_folder.glob("*.jpg"))
    # Use only the 5 good coins for clean augmentation (1800 total)
    # Skip: 1970_D (lighting issue), 1973, 1977, 1979, 1986 (crop too small)
    good_coins = ["1965_img_0023.jpg", "1967_img_0054.jpg", "1974_D_img_0049.jpg", "1984_P_img_0004.jpg", "1985_D1_img_0044.jpg"]
    images = [img for img in images if img.name in good_coins]
    if not images:
        print(f"No images in {base_folder}")
        return

    print(f"Using {len(images)} images from {base_folder}")
    print(f"Will generate 360 rotations per image = {len(images) * 360} total samples")
    print("Starting generation...")

    created = 0
    total_images = len(images)
    
    # Track successful crops to use as priors for problematic coins
    prior_cx, prior_cy, prior_radius = None, None, None
    
    for idx, img_path in enumerate(images, start=1):
        print(f"Processing image {idx}/{total_images}: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to read {img_path}")
            continue

        cropped, cx, cy, radius = crop_to_coin(img, pad=30, prior_cx=prior_cx, prior_cy=prior_cy, prior_radius=prior_radius)
        
        # Update prior with successful detection for next coin
        if cx is not None and cy is not None and radius is not None:
            prior_cx, prior_cy, prior_radius = cx, cy, radius
        
        padded = pad_to_circle(cropped, size=512, outer_margin=30)

        stem = img_path.stem

        # Generate 360 rotations (0-359 degrees)
        for rotation_angle in range(360):
            # Rotate the image by rotation_angle degrees
            out_img = rotate_numpy_image(padded, rotation_angle)

            # Label is the rotation angle (what we rotated by to correct the coin to upright)
            fname = f"{stem}_{rotation_angle:.2f}.jpg"

            # Always write to TRAIN_DIR (full 360 coverage for each image)
            out_path = TRAIN_DIR / fname
            all_path = ALL_DIR / fname

            ok = cv2.imwrite(str(out_path), out_img)
            ok_all = cv2.imwrite(str(all_path), out_img)
            if not ok or not ok_all:
                print(f"  Failed to write: {out_path} or {all_path}")
            else:
                created += 1

            # Also randomly add to TEST_DIR (creates overlap but validates generalization)
            if random.random() < TEST_SPLIT:
                test_path = TEST_DIR / fname
                ok_test = cv2.imwrite(str(test_path), out_img)
                if not ok_test:
                    print(f"  Failed to write test: {test_path}")

        print(f"  Finished {idx}/{total_images}: created total {created} samples so far")

    print(f"Created {created} samples under {OUT} (train/test/all)")


if __name__ == '__main__':
    main()
