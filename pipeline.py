def process_image_pipeline(image_path, debug_visual=False):
    """
    Chạy toàn bộ pipeline cho 1 ảnh:
    - load & preprocess
    - segment leaf
    - segment lesion
    - clean masks
    - extract feature vector

    Trả về:
    - feature_vector (numpy array)
    - dict intermediate (chứa các ảnh/mask trung gian nếu cần dùng)
    """
    # 1. Load & preprocess
    img_bgr_raw, img_rgb_raw = load_image(image_path)
    img_bgr, img_rgb = preprocess_image(img_bgr_raw, target_size=(256, 256))

    # 2. Segment leaf (HSV) + clean
    mask_leaf = segment_leaf_hsv(img_bgr)
    mask_leaf_clean = clean_mask(mask_leaf, ksize=7)

    # 3. Segment lesion (Lab + K-means) + clean
    mask_lesion_raw, labels_full = segment_lesion_lab_kmeans(img_rgb, mask_leaf_clean, k=3)
    mask_lesion_clean = clean_lesion_mask(mask_lesion_raw, min_area=50, ksize=5)

    # 4. Extract features
    feature_vector = extract_feature_vector(img_rgb, mask_leaf_clean, mask_lesion_clean)

    # 5. Lưu các bước trung gian (để vẽ / debug / ghi vào báo cáo)
    intermediates = {
        "img_rgb_raw": img_rgb_raw,
        "img_rgb": img_rgb,
        "mask_leaf": mask_leaf_clean,
        "mask_lesion": mask_lesion_clean,
    }

    if debug_visual:
        # Tô overlay vùng bệnh lên ảnh
        overlay = img_rgb.copy()
        red_layer = np.zeros_like(img_rgb)
        red_layer[:, :, 0] = 255
        alpha = 0.5
        mask_bool = mask_lesion_clean.astype(bool)
        overlay[mask_bool] = (
            alpha * overlay[mask_bool] + (1 - alpha) * red_layer[mask_bool]
        ).astype(np.uint8)

        show_images(
            [img_rgb, mask_leaf_clean, mask_lesion_clean, overlay],
            ["Ảnh sau preprocess", "Mask lá", "Mask vùng bệnh", "Overlay vùng bệnh"],
            cols=4,
            figsize=(14, 4)
        )

    return feature_vector, intermediates
