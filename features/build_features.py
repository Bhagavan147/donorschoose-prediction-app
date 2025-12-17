import numpy as np
import pandas as pd
from .target_encoding_features import target_encoding
from .resource_features import preprocess_resources
from .datetime_features import extract_datetime_features
from .title_features import extract_title_features, title_embeddings
from .essay_features import extract_essay_features, essay_embeddings
from .project_subcategory_features import project_subcategory_ohe
from .ohe_features import ohe_teacher_prefix_project_grade_category

def preprocess_input(data: dict) -> pd.DataFrame:

    features = {}

    # Target Encoding Features of teacher_id and school_state
    features.update(target_encoding({
        "teacher_id": data.get("teacher_id", ""),
        "school_state": data.get("school_state", "")
    }))

    # log transformation of teacher_number_of_previously_posted_projects
    features.update({
        "teacher_number_of_previously_posted_projects": np.log1p(data.get("teacher_number_of_previously_posted_projects", 0)),
    })

    # Resource Features
    features.update(preprocess_resources({
        "resources": data.get("resources", [])
    }))

    # Datetime Features
    features.update(extract_datetime_features({
        "project_submitted_datetime": data.get("project_submitted_datetime", "")
    }))

    # Title Features
    features.update(extract_title_features({
        "project_title": data.get("project_title", "")
    }))

    project_essay = data.get("project_essay_1", "") + " " + data.get("project_essay_2", "") + " " + data.get("project_essay_3", "") + " " + data.get("project_essay_4", "")

    # Essay Features
    features.update(extract_essay_features({
        "project_essay": project_essay
    }))

    # Project Subject Subcategory One-Hot Encoding
    features.update(project_subcategory_ohe({
        "project_subject_subcategories": data.get("project_subject_subcategories", "")
    }))

    # One-Hot Encoding for teacher_prefix and project_grade_category
    features.update(ohe_teacher_prefix_project_grade_category({
        "teacher_prefix": data.get("teacher_prefix", ""),
        "project_grade_category": data.get("project_grade_category", "")
    }))

    # Title Embeddings
    features.update(title_embeddings({
        "project_title": data.get("project_title", "")
    }))

    # Essay Embeddings
    features.update(essay_embeddings({
        "project_essay": project_essay
    }))

    return pd.DataFrame([features])