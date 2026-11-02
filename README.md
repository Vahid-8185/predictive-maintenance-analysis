# Predictive Maintenance Using Machine Learning

A Python-based machine learning project that explores how machine sensor readings can be used to identify possible equipment failures.

> **Note:** This is a learning portfolio project using a synthetic dataset. It does not use real factory, Siemens, or equipment data.

## Project Objective
Build a classification model using machine sensor data to practise a basic predictive maintenance workflow.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Logistic Regression

## Dataset
The dataset contains 200 synthetic records with:
- Temperature
- Vibration
- Pressure
- Operating hours
- Failure label

The failure labels are artificially generated and do not represent real equipment behaviour.

## Methodology
1. Loaded and inspected the dataset.
2. Checked for missing values.
3. Split the data into training and testing sets.
4. Scaled numerical features using StandardScaler.
5. Trained a Logistic Regression model.
6. Evaluated the model using accuracy, precision, recall, F1-score and a confusion matrix.

## Model Results
- **Test accuracy:** 66% (0.66)
- **Training records:** 150
- **Testing records:** 50

The model's results are for learning and demonstrating a machine learning workflow only. They should not be interpreted as real-world predictive performance.

## Project Structure
```text
predictive-maintenance-analysis/
├── data/
│   └── machine_sensor_sample.csv
├── outputs/
│   ├── confusion_matrix.png
│   └── test_predictions.csv
├── train_model.py
├── requirements.txt
└── README.md
```

## How to Run

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the model:

```bash
python train_model.py
```

## Limitations
This project uses a small synthetic dataset with artificially generated labels. A real predictive maintenance solution would require validated industrial sensor data, domain expertise and further model testing.
