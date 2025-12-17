import joblib

teacher_te = joblib.load("models/teacher_id_te.pkl")
state_te = joblib.load("models/school_state_te.pkl")

TEACHER_MAP = teacher_te["mapping"]
STATE_MAP = state_te["mapping"]
GLOBAL_MEAN = teacher_te["global_mean"]

def target_encoding(features: dict) -> dict:
    encoded_features = {
        "teacher_id": TEACHER_MAP.get(features.get("teacher_id"), GLOBAL_MEAN),
        "school_state": STATE_MAP.get(features.get("school_state"), GLOBAL_MEAN),
    }
    return encoded_features