from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "machine_sensor_sample.csv"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Dataset shape:", df.shape)
print("\nMissing values by column:")
print(df.isna().sum())

features = ["temperature_c", "vibration_mm_s", "pressure_bar", "operating_hours"]
X = df[features]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Predictive Maintenance - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
plt.close()

results = X_test.copy()
results["actual_failure"] = y_test
results["predicted_failure"] = predictions
results.to_csv(OUTPUTS / "test_predictions.csv", index=False)
print(f"\nSaved results to: {OUTPUTS}")
