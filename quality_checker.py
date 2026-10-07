import cv2


def check_image_quality(image_path):
    """
    Check the quality of an uploaded MRI image.
    Evaluates blur, brightness, and resolution.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Quality assessment with 'is_valid', 'issues', and 'details'.
    """
    issues = []
    details = {}

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {
            "is_valid": False,
            "issues": ["Could not read the image file. Please upload a valid image."],
            "details": {}
        }

    # Check resolution
    height, width = img.shape[:2]
    details["resolution"] = f"{width}x{height}"
    if width < 64 or height < 64:
        issues.append(f"Image resolution too low ({width}x{height}). Minimum 64x64 required.")
    if width > 4096 or height > 4096:
        issues.append(f"Image resolution too high ({width}x{height}). Maximum 4096x4096 recommended.")

    # Convert to grayscale for blur and brightness checks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    details["blur_score"] = round(laplacian_var, 2)
    if laplacian_var < 50:
        issues.append(f"Image appears too blurry (score: {laplacian_var:.1f}). Please upload a clearer image.")

    # Check brightness
    mean_brightness = gray.mean()
    details["brightness"] = round(mean_brightness, 2)
    if mean_brightness < 30:
        issues.append(f"Image is too dark (brightness: {mean_brightness:.1f}). Please upload a brighter image.")
    elif mean_brightness > 240:
        issues.append(f"Image is too bright (brightness: {mean_brightness:.1f}). Please upload a properly exposed image.")

    # Check if image is mostly uniform (possibly blank)
    std_dev = gray.std()
    details["contrast"] = round(std_dev, 2)
    if std_dev < 10:
        issues.append("Image has very low contrast. It may be blank or corrupted.")

    is_valid = len(issues) == 0
    details["quality"] = "Good" if is_valid else "Poor"

    return {
        "is_valid": is_valid,
        "issues": issues,
        "details": details
    }
