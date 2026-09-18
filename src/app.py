"""Train and use the California house grouping system.

The command line interface downloads the source data when needed, trains the
K-Means pseudo-labeler and classifier, writes metrics/plots, and persists both
models for later predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
	from utils import DEFAULT_DATA_URL, FEATURE_COLUMNS, load_housing_data
except ImportError:  # Allows ``python src/app.py`` from the repository root.
	from src.utils import DEFAULT_DATA_URL, FEATURE_COLUMNS, load_housing_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "housing.csv"
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"


def train_system(
	data_path: Path = DEFAULT_DATA_PATH,
	model_dir: Path = DEFAULT_MODEL_DIR,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
	data_url: str = DEFAULT_DATA_URL,
	test_size: float = 0.2,
	random_state: int = 42,
) -> dict:
	"""Train both stages and return evaluation metrics."""
	data = load_housing_data(data_path, data_url)
	features = data.loc[:, FEATURE_COLUMNS]
	x_train, x_test = train_test_split(
		features, test_size=test_size, random_state=random_state
	)

	# Scaling is essential because income and coordinates have very different
	# numeric ranges. Keep it inside each persisted pipeline.
	kmeans = Pipeline(
		[
			("scaler", StandardScaler()),
			("kmeans", KMeans(n_clusters=6, random_state=random_state, n_init=10)),
		]
	)
	train_labels = kmeans.fit_predict(x_train)
	test_labels = kmeans.predict(x_test)

	classifier = Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"classifier",
				RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1),
			),
		]
	)
	classifier.fit(x_train, train_labels)
	predictions = classifier.predict(x_test)

	metrics = {
		"accuracy": float(accuracy_score(test_labels, predictions)),
		"precision_weighted": float(
			precision_score(test_labels, predictions, average="weighted", zero_division=0)
		),
		"recall_weighted": float(
			recall_score(test_labels, predictions, average="weighted", zero_division=0)
		),
		"f1_weighted": float(
			f1_score(test_labels, predictions, average="weighted", zero_division=0)
		),
		"confusion_matrix": confusion_matrix(test_labels, predictions).tolist(),
		"classification_report": classification_report(
			test_labels, predictions, output_dict=True, zero_division=0
		),
		"n_samples": int(len(data)),
		"features": list(FEATURE_COLUMNS),
		"n_clusters": 6,
		"random_state": random_state,
	}

	model_dir.mkdir(parents=True, exist_ok=True)
	output_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(kmeans, model_dir / "kmeans.joblib")
	joblib.dump(classifier, model_dir / "classifier.joblib")
	(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

	training_plot = output_dir / "clusters.png"
	plt.figure(figsize=(10, 7))
	plt.scatter(x_train["Longitude"], x_train["Latitude"], c=train_labels, s=8, cmap="tab10", alpha=0.55, label="Train")
	plt.scatter(x_test["Longitude"], x_test["Latitude"], c=test_labels, s=16, cmap="tab10", marker="x", alpha=0.8, label="Test")
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.title("California housing groups (K-Means, k=6)")
	plt.legend()
	plt.tight_layout()
	plt.savefig(training_plot, dpi=150)
	plt.close()

	# Save the labeled complete dataset for analysts and downstream systems.
	labeled = data.copy()
	labeled["cluster"] = kmeans.predict(features).astype(int)
	labeled.to_csv(output_dir / "housing_labeled.csv", index=False)
	return metrics


def predict_one(latitude: float, longitude: float, medinc: float, model_dir: Path = DEFAULT_MODEL_DIR) -> int:
	"""Classify one new location using the persisted supervised model."""
	model = joblib.load(model_dir / "classifier.joblib")
	row = pd.DataFrame([[latitude, longitude, medinc]], columns=FEATURE_COLUMNS)
	return int(model.predict(row)[0])


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	train = subparsers.add_parser("train", help="download/load data and train models")
	train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
	train.add_argument("--url", default=DEFAULT_DATA_URL)
	train.add_argument("--test-size", type=float, default=0.2)
	train.add_argument("--random-state", type=int, default=42)
	predict = subparsers.add_parser("predict", help="classify one new location")
	predict.add_argument("latitude", type=float)
	predict.add_argument("longitude", type=float)
	predict.add_argument("medinc", type=float)
	args = parser.parse_args()

	if args.command == "train":
		metrics = train_system(args.data, data_url=args.url, test_size=args.test_size, random_state=args.random_state)
		print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
	else:
		print(json.dumps({"cluster": predict_one(args.latitude, args.longitude, args.medinc)}))
	return 0


if __name__ == "__main__":
	sys.exit(main())
