# House Price Predictor — Quick Reference

## Overview
Predict house prices using a **Feedforward Neural Network (FNN)**.  
Includes data preprocessing, model training, evaluation, saving/loading, and new data prediction.

---

## Setup

**Python Version:** 3.8+  
**Install dependencies:**
```bash
pip install pandas numpy scikit-learn tensorflow
```

**Files Needed:**
- `Housing.csv` – Dataset  
- `house_price_predictor.py` – Script

---

## Usage

Run the script:
```bash
python house_price_predictor.py
```

**Outputs:**
- Dataset preview and headers  
- Preprocessing summary (numeric/categorical features)  
- Model training progress  
- **R² value** and **MSE**  
- Saved model: `houseprice_fnn_model.keras`  
- Predicted price for example house

---

## Example: New Data Prediction
```python
new_data = pd.DataFrame({
    'area': [9960],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['yes'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [2],
    'prefarea': ['yes'],
    'furnishingstatus': ['semi-furnished']
})
```
Output:
```
Predicted Price: 1234567.89
```

---

## Function Summary

| Function | Purpose |
|----------|---------|
| `load_data()` | Load CSV dataset |
| `name_of_headers()` | Get all column names |
| `feature_header_of_dataset()` | Get feature columns |
| `target_header_of_dataset()` | Get target column |
| `preprocessing_pipeline()` | Scale numeric + encode categorical features |
| `fnn_model_pipeline()` | Build, train, evaluate FNN |
| `save_model()` | Save trained model |
| `load_model()` | Load model from disk |
| `predict_new_data()` | Predict house price for new data |
| `main()` | Full pipeline execution |

---

## Notes
- Early stopping prevents overfitting.  
- Ensure new data has the **same columns** as training data.  
- Adjust `feature_header_of_dataset()` if dataset columns change.