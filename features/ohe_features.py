import pandas as pd
import joblib

ohe_model = joblib.load("models/ohe_encoder.pkl")

def ohe_teacher_prefix_project_grade_category(features: dict) -> dict:

    ohe_features = ohe_model.transform(pd.DataFrame([features]))
    ohe_feature_names = ohe_model.get_feature_names_out()

    return dict(zip(ohe_feature_names, ohe_features[0]))