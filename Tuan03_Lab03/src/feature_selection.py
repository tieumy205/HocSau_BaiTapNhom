from sklearn.feature_selection import SelectKBest, mutual_info_regression


RANDOM_STATE = 42


def mutual_info_score(X, y):
    """
    Wrapper cho mutual_info_regression.

    Phải khai báo ở cấp module (không dùng lambda)
    để Pipeline có thể được joblib/pickle lưu lại.
    """
    return mutual_info_regression(
        X,
        y,
        random_state=RANDOM_STATE
    )


def build_feature_selector(k="all"):
    """
    Tạo SelectKBest sử dụng Mutual Information.

    Parameters
    ----------
    k : int hoặc "all"
        Số lượng feature cần giữ lại.
    """

    return SelectKBest(
        score_func=mutual_info_score,
        k=k
    )


def get_selected_feature_names(pipeline):
    """
    Lấy tên các feature được SelectKBest chọn
    từ một Pipeline đã được fit.

    Pipeline cần có các step:
        - preprocessor
        - feature_selection
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["feature_selection"]

    # Tên feature sau preprocessing / OneHotEncoder
    feature_names = preprocessor.get_feature_names_out()

    # Mask True/False của các feature được chọn
    selected_mask = selector.get_support()

    # Lấy tên feature được giữ lại
    selected_features = feature_names[selected_mask]

    return selected_features