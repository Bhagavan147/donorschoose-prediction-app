import numpy as np
import joblib
import faiss
import re

sbert_model = joblib.load("models/sbert_model.pkl")
resources_pca = joblib.load("models/resources_pca.pkl")
index = faiss.read_index("models/resources_kmeans.index")

# clusters to categories
cluster_to_category = {
    0:"Books",10:"Books",11:"Books",20:"Books",27:"Books",29:"Books",
    1:"Stationery",2:"Stationery",9:"Stationery",14:"Stationery",18:"Stationery",24:"Stationery",
    7:"Classroom_aid",8:"Classroom_aid",13:"Classroom_aid",28:"Classroom_aid",
    5:"STEM",16:"STEM",21:"STEM",26:"STEM",
    4:"Electronics",6:"Electronics",12:"Electronics",25:"Electronics",
    17:"Sports_Fitness",19:"Sports_Fitness",22:"Sports_Fitness",23:"Sports_Fitness",
    15:"Food",
    3:"Subscriptions"
}

def sbert_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()                  # remove extra spaces at ends
    text = re.sub(r'\s+', ' ', text)     # collapse multiple spaces
    return text.lower()                  # lowercase since resources are not case-sensitive

def predict_cluster(embeddings):
    _, labels = index.search(embeddings.astype("float32"), 1)
    return labels.flatten()

def preprocess_resources(features: dict) -> dict:
    resources = features.get("resources", [])

    resource_features = {
        "Books": 0,
        "Classroom_aid": 0,
        "Electronics": 0,
        "Food": 0,
        "STEM": 0,
        "Sports_Fitness": 0,
        "Stationery": 0,
        "Subscriptions": 0,
        "total_resource_cost": 0.0
    }

    if not resources:
        return resource_features

    # Prepare descriptions for embedding
    resource_descriptions = [sbert_clean(resource["description"]) for resource in resources]

    # sbert embeddings for resource descriptions
    description_embeddings = sbert_model.encode(resource_descriptions)

    # PCA transformation of sbert embeddings
    description_pca_embeddings = resources_pca.transform(description_embeddings)

    # Predict clusters for each resource
    resource_clusters = predict_cluster(description_pca_embeddings)

    for i in range(len(resources)):
        cleaned_description = sbert_clean(resources[i].get("description", ""))

        if cleaned_description == "":
            continue
        elif "bk set" in cleaned_description:
            category = "Books"
        elif ("bean bag" in cleaned_description) or ("light" in cleaned_description) or ("filters" in cleaned_description):
            category = "Classroom_aid"
        else:
            category = cluster_to_category.get(resource_clusters[i])

        resource_features[category] += resources[i].get("quantity", 0)
        resource_features["total_resource_cost"] += resources[i].get("price", 0.0) * resources[i].get("quantity", 0)

    resource_features["total_resource_cost"] = np.log1p(resource_features["total_resource_cost"])

    return resource_features