# Diabetes Risk Predictor (Flask)

Predicts the probability of diabetes from clinical metrics using a Logistic Regression model trained on the Pima Indians Diabetes dataset.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python model/train_model.py
flask run
