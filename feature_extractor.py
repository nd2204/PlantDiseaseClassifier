import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew

def compute_color_moments(img_rgb, mask=None):
    """
    Tính các moment màu (mean, std, skewness) cho từng kênh R,G,B.
    Trả về: np.array length=9 (3 moment × 3 kênh)
    """
    if mask is not None:
        mask_bool = mask.astype(bool)
        pixels = img_rgb[mask_bool]
    else:
        pixels = img_rgb.reshape(-1, 3)

    feats = []
    for ch in range(3):
        vals = pixels[:, ch].astype(np.float32)
        if len(vals) == 0:
            mean_val = std_val = skew_val = 0.0
        else:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            skew_val = skew(vals) if len(vals) > 1 else 0.0
        feats.extend([mean_val, std_val, skew_val])
    return np.array(feats, dtype=np.float32)

def compute_glcm_features(gray, mask=None, levels=16, distances=[1], angles=[0]):
    """
    Tính đặc trưng GLCM (contrast, correlation, energy, homogeneity)
    cho vùng được chỉ định bởi mask (nếu có).
    """
    gray = gray.copy()
    if mask is not None:
        gray[mask == 0] = 0

    # Giảm số mức xám
    gray_quant = (gray / (256 / levels)).astype(np.uint8)

    glcm = graycomatrix(
        gray_quant,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    # Lấy trung bình trên tất cả distance/angle
    feats = []
    for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
        val = graycoprops(glcm, prop_name).mean()
        feats.append(val)
    return np.array(feats, dtype=np.float32)


def compute_lbp_features(gray, mask=None, radius=1, n_bins=10):
    """
    Tính đặc trưng LBP:
    - Tính LBP (uniform)
    - Lấy histogram (rút gọn còn n_bins giá trị đầu hoặc gộp lại)
    """
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    if mask is not None:
        lbp_vals = lbp[mask.astype(bool)]
    else:
        lbp_vals = lbp.flatten()

    max_val = int(lbp_vals.max()) + 1
    hist, bin_edges = np.histogram(lbp_vals, bins=max_val, range=(0, max_val), density=True)

    # Rút gọn: lấy n_bins đầu (hoặc bạn có thể gộp/resize histogram)
    if len(hist) >= n_bins:
        hist_feat = hist[:n_bins]
    else:
        # Nếu histogram ngắn hơn thì pad thêm 0
        hist_feat = np.pad(hist, (0, n_bins - len(hist)), mode='constant')

    return hist_feat.astype(np.float32)

def compute_shape_features(mask_lesion, mask_leaf):
    """
    Đặc trưng hình dạng:
    - Tổng diện tích vùng lá
    - Tổng diện tích vùng bệnh
    - Tỉ lệ bệnh / lá
    - Số vùng bệnh
    - Trung bình diện tích & circularity các vùng bệnh
    """
    mask_leaf_bin = (mask_leaf > 0).astype(np.uint8)
    mask_lesion_bin = (mask_lesion > 0).astype(np.uint8)

    leaf_area = int(mask_leaf_bin.sum())
    lesion_area = int(mask_lesion_bin.sum())

    ratio = lesion_area / leaf_area if leaf_area > 0 else 0.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_lesion_bin, connectivity=8)
    num_lesions = max(0, num_labels - 1)

    lesion_areas = []
    circularities = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        component_mask = np.zeros_like(mask_lesion_bin)
        component_mask[labels == label] = 255

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0.0

        lesion_areas.append(area)
        circularities.append(circularity)

    if len(lesion_areas) > 0:
        mean_area = float(np.mean(lesion_areas))
        mean_circ = float(np.mean(circularities))
    else:
        mean_area = 0.0
        mean_circ = 0.0

    # Ghép thành vector
    feats = np.array([
        float(leaf_area),
        float(lesion_area),
        float(ratio),
        float(num_lesions),
        mean_area,
        mean_circ
    ], dtype=np.float32)

    return feats

def extract_feature_vector(img_rgb, mask_leaf, mask_lesion):
    """
    Hàm tổng hợp để trích vector đặc trưng cuối cùng từ:
    - Ảnh RGB
    - mask lá
    - mask vùng bệnh
    """

    # Ảnh xám cho texture
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1) Color moments trên vùng lá
    color_feats = compute_color_moments(img_rgb, mask_leaf)

    # 2) GLCM trên vùng lá
    glcm_feats = compute_glcm_features(gray, mask_leaf, levels=16, distances=[1], angles=[0])

    # 3) LBP trên vùng lá
    lbp_feats = compute_lbp_features(gray, mask_leaf, radius=1, n_bins=10)

    # 4) Shape trên mask vùng bệnh
    shape_feats = compute_shape_features(mask_lesion, mask_leaf)

    # Ghép tất cả
    feature_vector = np.concatenate([color_feats, glcm_feats, lbp_feats, shape_feats])
    return feature_vector
