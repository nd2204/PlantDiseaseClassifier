import numpy as np
import cv2
from sklearn.cluster import KMeans

def segment_leaf_hsv(img_bgr, lower_green=[25, 40, 20], upper_green=[100, 255, 255]):
    """
    Phân đoạn lá dựa trên màu trong không gian HSV.
    - Giả sử lá có màu xanh, nền là xám / đen / màu khác.
    Trả về: mask_leaf (0/255)
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Ngưỡng "xanh lá" (có thể chỉnh tay tùy tập dữ liệu)
    lower_green = np.array(lower_green)
    upper_green = np.array(upper_green)

    mask_leaf = cv2.inRange(img_hsv, lower_green, upper_green)
    return mask_leaf


def clean_mask(mask, ksize=7, iterations_close=2, iterations_open=1):
    """
    Làm sạch mask nhị phân bằng phép toán hình thái học:
    - Đóng (closing) để lấp lỗ nhỏ
    - Mở (opening) để xóa nhiễu nhỏ
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=iterations_open)
    return mask_clean

def segment_lesion_lab_threshold(
    img_rgb,
    mask_leaf,
    channel="a",       # 'a' hoặc 'b'
    thresh=None,       # nếu None -> tự tính bằng Otsu trên vùng lá
    mode="greater"     # 'greater' hoặc 'less'
):
    """
    Phân đoạn vùng bệnh trên lá bằng ngưỡng trên kênh a* hoặc b* trong không gian Lab.

    Tham số:
    - img_rgb : ảnh RGB (đã resize / tiền xử lý)
    - mask_leaf : mask lá (0/255), chỉ vùng lá được xét là ứng viên bệnh
    - channel : 'a' hoặc 'b' (kênh trong Lab dùng để threshold)
    - thresh  : giá trị ngưỡng. Nếu None -> dùng Otsu để tự tìm ngưỡng trên pixel thuộc lá
    - mode    : 
        - 'greater' -> vùng bệnh = pixel có giá trị kênh >= thresh
        - 'less'    -> vùng bệnh = pixel có giá trị kênh <= thresh

    Trả về:
    - mask_lesion : mask vùng bệnh (0/255)
    - used_thresh : ngưỡng thực tế đã dùng
    """

    # 1. Chuyển RGB -> Lab
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)

    if channel == "a":
        ch = a
    elif channel == "b":
        ch = b
    else:
        raise ValueError("channel phải là 'a' hoặc 'b'")

    # 2. Lấy các giá trị kênh chỉ trong vùng lá
    mask_bool = mask_leaf.astype(bool)
    ch_leaf_values = ch[mask_bool]

    if ch_leaf_values.size == 0:
        # Không có pixel lá nào -> trả về mask rỗng
        return np.zeros_like(mask_leaf, dtype=np.uint8), None

    # 3. Xác định ngưỡng
    if thresh is None:
        # Dùng Otsu trên vector 1D (reshape thành Nx1 để cv2.threshold xử lý)
        vals = ch_leaf_values.reshape(-1, 1).astype(np.uint8)
        # Otsu cần maxval, nhưng ta chỉ cần ngưỡng trả về
        _, otsu_mask = cv2.threshold(
            vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # cv2.threshold trả về ngưỡng ở biến _ thứ nhất nếu dùng kiểu thông thường,
        # nhưng do ta đang gọi với biến '_' nên viết lại rõ:
        # (ret, otsu_mask) = cv2.threshold(...)
        # ở đây mình viết lại đúng:
        ret, _ = cv2.threshold(
            vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        used_thresh = ret
    else:
        used_thresh = float(thresh)

    # 4. Tạo mask vùng bệnh trên toàn ảnh, nhưng chỉ trong vùng lá
    mask_lesion = np.zeros_like(mask_leaf, dtype=np.uint8)

    if mode == "greater":
        cond = (ch >= used_thresh) & (mask_leaf == 255)
    elif mode == "less":
        cond = (ch <= used_thresh) & (mask_leaf == 255)
    else:
        raise ValueError("mode phải là 'greater' hoặc 'less'")

    mask_lesion[cond] = 255

    return mask_lesion, used_thresh

def segment_lesion_lab_kmeans(img_rgb, mask_leaf, k=3, lesion_cluster_idx=None):
    """
    Phân đoạn vùng bệnh bên trong lá bằng K-means trên không gian Lab.
    Bước:
    - Chuyển RGB -> Lab
    - Chỉ gom cụm các pixel thuộc vùng lá
    - K-means với k cụm
    - Chọn 1 cụm làm "vùng bệnh"

    Tham số:
    - k: số cụm màu
    - lesion_cluster_idx: nếu None -> dùng heuristic đơn giản để chọn,
                          nếu không -> dùng index đó (0..k-1)

    Trả về: mask_lesion (0/255), labels_full (ma trận nhãn cụm k cho toàn ảnh)
    """
    h, w, _ = img_rgb.shape
    leaf_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    leaf_lab_flat = leaf_lab.reshape(-1, 3)
    mask_flat = mask_leaf.flatten()

    # Chỉ pixel thuộc lá
    leaf_pixels = leaf_lab_flat[mask_flat == 255]

    if len(leaf_pixels) == 0:
        return np.zeros_like(mask_leaf), None

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(leaf_pixels)

    # Gán nhãn đầy đủ cho ảnh
    labels_full = np.full(mask_flat.shape, -1, dtype=np.int32)
    labels_full[mask_flat == 255] = kmeans.labels_
    labels_full = labels_full.reshape(h, w)

    # Nếu chưa biết cụm bệnh, chọn cụm có trung bình kênh L* thấp hơn (tối hơn)
    if lesion_cluster_idx is None:
        L, a, b = cv2.split(leaf_lab)
        means_L = []
        for c in range(k):
            L_vals = L[labels_full == c]
            if len(L_vals) > 0:
                means_L.append(np.mean(L_vals))
            else:
                means_L.append(255.0)
        # Giả định vùng bệnh thường tối hơn (chỉ là heuristic đơn giản)
        lesion_cluster_idx = int(np.argmin(means_L))

    mask_lesion = np.zeros_like(mask_leaf)
    mask_lesion[labels_full == lesion_cluster_idx] = 255

    return mask_lesion, labels_full

def clean_lesion_mask(mask_lesion, min_area=50, ksize=5):
    """
    Làm sạch mask vùng bệnh:
    - Dùng open/close
    - Lọc theo diện tích để bỏ vùng nhiễu quá nhỏ
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_clean = cv2.morphologyEx(mask_lesion, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Lọc theo diện tích
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    final_mask = np.zeros_like(mask_clean)
    for label in range(1, num_labels):  # 0 là nền
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == label] = 255

    return final_mask

