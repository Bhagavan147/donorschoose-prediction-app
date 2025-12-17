# DonorsChoose Project Approval Prediction

## Overview

This project predicts whether a DonorsChoose project proposal will be approved, enabling **risk-based prioritization**, **faster reviews**, and **consistent decision-making**.
The problem is framed as a **binary classification task** with strong class imbalance.

---

## Target

* **Target variable:** `project_is_approved`
* **Classes:** Approved (1), Rejected (0)

---

## Data

* **projects.csv:** 109,248 rows × 16 columns
* **resources.csv:** 1,541,272 rows × 4 columns
* **Split:** 60% train / 20% validation / 20% test

---

## Metric Choice

The dataset is **~85% approved**, making accuracy and ROC-AUC misleading.

**Precision-Recall AUC (PR-AUC)** was used as the primary metric to better capture rejection risk.

---

## Feature Engineering

* **Teacher behavior:** K-Fold target encoding (`teacher_id`, `school_state`)
* **Categoricals:** One-Hot Encoding (`teacher_prefix`, grade category)
* **Resources:** SBERT embeddings + PCA + FAISS clustering
* **Text:** Readability, length, sentiment + SBERT embeddings
* **Numerical:** Log transforms for skewed variables
* **Temporal:** Submission year, month, day, hour

All encoders and models are **persisted for inference**.

---

## Modeling

* **Model:** XGBoost Classifier
* **Imbalance handling:** `scale_pos_weight ≈ 0.178`
* **Tuning:** Optuna (PR-AUC optimization)

### Final Performance

* **Train PR-AUC:** 0.9374
* **Validation PR-AUC:** 0.9373
* **Test PR-AUC:** 0.9363
* **Test ROC-AUC:** 0.7571

Consistent scores indicate **minimal overfitting**.

---

## Threshold Selection

Multiple thresholds were evaluated.
➡ **0.7** was chosen to balance false approvals and recall.

---

## Deployment

* **Flask REST API**
* Modular feature pipeline
* Persisted encoders, embeddings, and FAISS index
* Local deployment

**Sample Input**

```json
{
    "id": "p145979",
    "teacher_id": "3cbd43cbada3b2214c6abb4399f81eaf",
    "teacher_prefix": "Mrs.",
    "school_state": "MD",
    "project_submitted_datetime": "12-05-2016 09:15",
    "project_grade_category": "Grades 3-5",
    "project_subject_categories": "Math & Science",
    "project_subject_subcategories": "Mathematics",
    "project_title": "iPad Charging Station",
    "project_essay_1": "The biggest start-up successes -from Henry Ford to Bill Gates to Mark Zuckerberg -were pioneered by people from solidly middle-class backgrounds. These founders were not wealthy when they began. They were hungry for success, but knew they had a solid support system to fall back on if they failed.\"\r\n\r\n",
    "project_essay_2": "My school is a unique environment where we always put the needs of the children first.\r\nThe school motto is \" Kids Go to College.\" Each week the staff and students wear their college's colors to promote the belief that education is important. Our students are taught from the beginning of their schooling experience how vital education has become in today's society. My school is primarily composed of middle class families who do not have the benefit from government funded technology initiatives that many schools in the area utilize. We have worked very hard this year to expose the children to 21st Century technology in order for them to be college and career ready. My students enjoy expressing themselves in a variety of modalities. These iPads we have previously have been awarded will allow their creativity to soar! We are now in need of a way to ensure their longevity.",
    "project_essay_3": "In a previous Donor's Choose project, we were awarded 3 wonderful iPad for us to utilize in our classroom. We were also awarded the covers to keep them safe.  After many hours of use, we realized that having a central location for them to be charged would be a wonderful addition to our classroom.\r\n    \r\nThe iPad combination of cases and charging will allow students who are less engaged with textbooks and paper/pencil activities to show off their skills without the worry of destroying the technology. Having safe access to this kind of technology will allow students to remain on the forefront of advances in the classroom",
    "project_essay_4": ".\r\nThey will transition smoothly through school and eventually their career. I would like to create a classroom of inquisitive thinkers and problem solvers. Technology is one of our best options for this.",
    "project_resource_summary": "My students need a way to charge our recently donated iPads.  We have also included the various types of cables to insure safe charging during storms.",
    "teacher_number_of_previously_posted_projects": 2,
    "resources": [
        {
            "description": "CablesOnline 5-PACK 6 inch USB 2.0 A-Type Male to Micro-B Male Charge & Sync Cable, White (USB-1500W-5)",
            "quantity": 1,
            "price": 14.42
        },
        {
            "description": "GadgetsPRO Lightning to USB Cable for all Apple Lightning devices, Short 0.2m/8.5in (4-pack)",
            "quantity": 1,
            "price": 19.95
        },
        {
            "description": "Griffin PowerDock 5 - Multi-Charger Dock [Charges 5 USB devices] [For iPad, for iPhone, and for iPod]",
            "quantity": 1,
            "price": 67.24
        }
    ]
}
```

**Sample Response**

```json
{
  "id": "p145979",
  "predicted_status": "Approved",
  "approval_probability": 0.917
}
```

---

## Insights

* The dataset is **highly imbalanced**, making Precision–Recall AUC a more reliable evaluation metric than accuracy or ROC-AUC.
* Raw teacher identifiers show strong correlation with approval outcomes, but **direct usage causes leakage** and does not generalize.
* Aggregated teacher history features capture behavioral patterns while remaining **deployment-safe**.
* Resource requests are sparse and skewed toward a small number of dominant categories.
* Projects requesting **essential classroom materials** (e.g., books, stationery) show higher approval likelihood than high-cost or specialized resources.
* Total resource cost and quantity are **right-skewed**, benefiting from log transformations.
* Temporal features capture submission patterns but contribute **limited predictive power**.
* Text features (title and essay) exhibit high variance and sparsity, making **embeddings more effective than raw counts**.
* Readability and length features are statistically significant but have **small practical impact** when used in isolation.
* XGBoost handles heterogeneous feature types and sparsity effectively in this setting.
* Stable PR-AUC across train, validation, and test sets indicates **good generalization**.
* ROC-AUC is lower due to imbalance, reinforcing PR-AUC as the primary metric.
* Threshold selection has a **material impact on false approvals vs recall trade-offs**.

---

## Recommendations

* Use the model as a **ranking or risk-scoring system**, not a hard decision rule.
* Apply a **higher decision threshold (0.7)** to reduce false approvals while maintaining high recall.
* Route low-risk projects directly to fast-track review queues.
* Flag high-cost or complex resource requests for additional scrutiny.
* Avoid rules based solely on submission timing or seasonal patterns.
* Prefer **behavior-based aggregations** over identity-based features.
* Monitor PR-AUC and false approvals as primary performance indicators.
* Periodically retrain and recalibrate the model to reflect evolving teacher behavior and funding patterns.
* Extend the pipeline with model explainability for reviewer transparency.

---
