# OnTime - Spatio-Temporal Traffic Forecasting

A compact pipeline for short-term traffic forecasting that fuses historical graph signals with optional camera-derived inputs and supports explainability.

Pre-trained weights are included. You can run the forecast API and dashboard immediately without retraining.

## Repository Layout (Current)

```text
/OnTime
├── data/
│   ├── mappings/
│   │   ├── cam_to_sensor.json
│   │   └── la_boundary.geojson
│   └── raw/
│       ├── adj_METR-LA.pkl
│       ├── METR-LA.h5
│       └── weather_CA_2019.csv
├── frontend-nextjs/
│   ├── public/
│   ├── src/
│   ├── next-env.d.ts
│   ├── next.config.js
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.ts
│   └── tsconfig.json
├── models/
│   ├── shap_explainer.py
│   ├── stgcn_model.py
│   └── stgcn_weights.pt
├── reports/
│   └── metrics.csv
├── scripts/
│   ├── generate_dashboard_data.py
│   └── generate_sensor_locations.py
├── src/
│   ├── api.py
│   ├── data_loader.py
│   ├── map_matcher.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated folders (not tracked): frontend-nextjs/.next, frontend-nextjs/node_modules, Python __pycache__

## Quick Start

### 1. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Start the backend API

```powershell
uvicorn src.api:app --reload
```

### 3. Start the frontend dashboard

```powershell
cd frontend-nextjs
npm install
npm run dev
```

## Optional: Retrain the model

```powershell
python -u src\train.py --epochs 30 --batch 64 --window 12 --horizon 3
```

## Notes

- The API uses models/stgcn_weights.pt by default.
- Raw data files live under data/raw.
