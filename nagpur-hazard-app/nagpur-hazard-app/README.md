# Nagpur Electrical Hazard Predictor

AI-based prediction of public electrical hazard risk across Nagpur localities, with a
pole-level asset registry that accepts any CSV/Excel upload.

**Live app:** _paste your Streamlit link here_

## Features
- Risk map of Nagpur localities (XGBoost model, relative HIGH / MEDIUM / LOW levels)
- Live Nagpur weather (Open-Meteo) or manual weather scenarios
- Pole registry: upload CSV/Excel, auto column matching, validation warnings,
  maintenance status (OVERDUE / DUE SOON / OK), asset age, filters, CSV download
- Pole details page with map and locality risk

## Data
| Data | Status |
|---|---|
| Weather (training and live) | Real (Open-Meteo / ERA5) |
| Locality names and coordinates | Real (OpenStreetMap) |
| Demo pole records, infrastructure attributes, hazard labels | Synthetic |

Academic project. Not affiliated with MSEDCL, Nagpur Municipal Corporation or any government body.
It is not a safety alert service.

## Repository layout
```
app.py                 Streamlit app
requirements.txt
generate_poles.py      creates synthetic demo poles (data/poles.csv)
data/areas.csv         localities used by the model   (required)
data/poles.csv         demo poles                      (optional)
models/hazard_model.pkl trained model                  (required)
sample_data/           test files for the upload tab
```

## Run locally
```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repository.
2. share.streamlit.io -> Create app -> pick the repo, branch `main`, file `app.py`.
3. Deploy.
