# System Specification: California House Grouping & Classification System

## 1. Overview & Objective
The system is designed to automatically categorize geographical census block groups in California into regions based on their location (**Latitude**, **Longitude**) and socio-economic standing (**Median Income**). Because the source dataset lacks pre-defined category labels, the system implements a two-stage hybrid machine learning pipeline:
1. **Unsupervised Clustering (Pseudo-Labeling):** Automatically segments the spatial and income attributes into distinct, labeled clusters.
2. **Supervised Classification:** Trains a predictive model using the newly generated cluster labels as the target variable to accurately classify unseen locations.

---

## 2. Architectural Design & Pipeline Flow

The workflow operates sequentially as illustrated below:

```
┌────────────────────────┐
│  Raw Data Loading &    │ <── Link / Local housing.csv (Lat, Lon, MedInc)
│  Feature Selection     │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Train/Test Splitting  │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Unsupervised Phase    │ <── K-Means Clustering (K=6) on Train Set
│  (K-Means Engine)      │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│   Label Generation     │ <── Generates 'cluster' target column (0-5)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│   Supervised Phase     │ <── Classifier (e.g., Random Forest / XGBoost)
│ (Classifier Training)  │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│    Model Artifact     │ <── Serialization & Storage (K-Means & Classifier)
│      Persistence       │
└────────────────────────┘
```

---

## 3. Detailed Step-by-Step Functional Specifications

### Step 1: Data Ingestion & Splitting
* **Data Origin:** `housing.csv` via the live [BreatheCode Dataset URL](https://breathecode.herokuapp.com/asset/internal-link?id=439&path=housing.csv).
* **Feature Extraction:** Filter out all attributes except for:
  * `Latitude` (Float, geographic marker)
  * `Longitude` (Float, geographic marker)
  * `MedInc` (Float, median income of the block group)
* **Data Partitioning:** Professionally segment the dataset into **Training** and **Testing** sets using a standard ratio (e.g., 80/20 or 70/30) with a fixed random seed to maintain reproducibility.

### Step 2: Unsupervised Grouping (K-Means Engine)
* **Hyperparameters:** Set the number of clusters ($K$) strictly to **6**.
* **Fit Operations:** Fit the K-Means algorithm using exclusively the training set features (`Latitude`, `Longitude`, `MedInc`).
* **Feature Engineering:** Apply predictions back to the training dataset, storing the result in a new, properly structured column named `cluster`.
* **Data Transformation:** Ensure the `cluster` output is treated explicitly as a categorical factor or integer label.
* **Visualization (Training):** Construct a 2D scatter plot (mapping spatial coordinates or income distribution) color-coded by the 6 generated cluster IDs.

### Step 3: Out-of-Sample Cluster Prediction
* **Test Application:** Pass the unseen test set through the trained K-Means model to map every point to its closest cluster centroid.
* **Verification Plotting:** Overlay the predicted test cluster points onto the original training scatter plot to visually confirm geometric fit and spatial consistency.

### Step 4: Supervised Learning Framework
* **Objective:** Learn the boundaries defined by the K-Means clustering algorithm so the system can evaluate new coordinates instantly without re-clustering.
* **Target Feature:** The generated `cluster` label column.
* **Predictor Input:** `Latitude`, `Longitude`, and `MedInc`.
* **Evaluation Metrics:** Generate comprehensive classification statistics, including an Accuracy Score, Precision, Recall, F1-Score, and a Confusion Matrix against the test partition labels.

### Step 5: Model Serialization & Persistence
* **Export Requirement:** Serialize the state of both independent models:
  1. The trained **K-Means clustering engine**
  2. The trained **supervised classification model**
* **Format:** Save using reliable serialization frameworks (`pickle` or `joblib`) into their designated storage directories.

---

## 4. Technical Stack Requirements
* **Language:** Python 3.8+
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning Library:** `scikit-learn`
* **Visualization:** `matplotlib` or `seaborn`
* **Artifact Persistence:** `joblib` or `pickle`