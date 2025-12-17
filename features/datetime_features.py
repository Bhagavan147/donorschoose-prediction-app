from datetime import datetime

def extract_datetime_features(features: dict) -> dict:
    project_submitted_datetime = features.get("project_submitted_datetime", "")

    datetime_features = {
        "project_submission_year": 2017,
        "project_submission_month": 0,
        "project_submission_day": 0,
        "project_submission_hour": 0,
    }

    if not project_submitted_datetime:
        return datetime_features
    
    project_submitted_datetime = datetime.strptime(project_submitted_datetime, "%d-%m-%Y %H:%M")

    datetime_features["project_submission_year"] = project_submitted_datetime.year
    datetime_features["project_submission_month"] = project_submitted_datetime.month
    datetime_features["project_submission_day"] = project_submitted_datetime.day
    datetime_features["project_submission_hour"] = project_submitted_datetime.hour

    return datetime_features