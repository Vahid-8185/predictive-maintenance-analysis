# Predictive Maintenance Using Machine Learning

> Learning portfolio project using a synthetic dataset. It is not real factory, Siemens, or equipment data.

## Goal
Build a basic classification model to explore whether machine sensor readings can help flag a possible machine failure.

## Tools
Python, pandas, scikit-learn, matplotlib.

## Dataset
`data/machine_sensor_sample.csv` contains generated example records with temperature, vibration, pressure, operating hours, and a binary failure label. The labels are artificially generated for learning and do not represent real equipment behaviour.

## Run
```bash
python -m pip install -r requirements.txt
python train_model.py
```

## Method
1. Load and inspect the synthetic sensor dataset.
2. Split the data into training and test sets.
3. Scale numeric features and train a logistic regression classifier.
4. Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Save example test predictions.

## Outputs
Running the script creates:
- `outputs/confusion_matrix.png`
- `outputs/test_predictions.csv`

## Results
Run the script and add the actual evaluation metrics printed in your terminal. Do not claim that the model predicts real machine failures.

## Limitations
This is a small synthetic learning dataset. Its labels are artificially generated, so its metrics are only for practising a machine-learning workflow. A real predictive-maintenance system would require validated equipment data, careful testing, and domain expertise.
