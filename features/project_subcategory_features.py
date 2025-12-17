def project_subcategory_ohe(features: dict) -> dict:
    project_subcategories = features.get("project_subject_subcategories", "")
    
    project_subcategory_features = {
        "Applied Sciences": 0,
        "Care & Hunger": 0,
        "Character Education": 0,
        "Civics & Government": 0,
        "College & Career Prep": 0,
        "Community Service": 0,
        "ESL": 0,
        "Early Development": 0,
        "Economics": 0,
        "Environmental Science": 0,
        "Extracurricular": 0,
        "Financial Literacy": 0,
        "Foreign Languages": 0,
        "Gym & Fitness": 0,
        "Health & Life Science": 0,
        "Health & Wellness": 0,
        "History & Geography": 0,
        "Literacy": 0,
        "Literature & Writing": 0,
        "Mathematics": 0,
        "Music": 0,
        "Nutrition Education": 0,
        "Other_Category": 0,
        "Parent Involvement": 0,
        "Performing Arts": 0,
        "Social Sciences": 0,
        "Special Needs": 0,
        "Team Sports": 0,
        "Visual Arts": 0,
        "Warmth": 0
    }

    if not project_subcategories:
        return project_subcategory_features
    
    subcategories = project_subcategories.split(", ")
    for subcat in subcategories:
        key = subcat if subcat in project_subcategory_features else "Other_Category"
        project_subcategory_features[key] = 1
    
    return project_subcategory_features