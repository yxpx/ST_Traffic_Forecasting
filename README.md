# ST_Traffic_Forecasting

Short-term traffic forecasting with a spatio-temporal graphical convolutional network (ST-GCN) model, a FastAPI backend, and a Next.js dashboard. The backend fuses historical graph signals and optional camera inputs. Pre-trained weights are included.

## Repo layout

- backend for the API and training
- frontend for the Next.js dashboard
- data/raw for dataset files
- models for weights and model code
- scripts for data generation helpers
- assets/vision for the CV tools

## Setup

### 1. Python env (uv)

```powershell
uv venv .venv
uv pip install -r requirements.txt
```

### 2. Download METR-LA data

Dataset: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset

```powershell
uv pip install kagglehub
python scripts\download_metr_la.py
```

This script places the dataset files into data/raw.

### 3. Weather data

Default path is data/raw/weather_CA_2019.csv. The backend reads this through the WEATHER_PATH environment variable and training uses --weather.

If your weather file does not align with the METR-LA time range, it will bias the features. For clean experiments, either provide a matched file or disable weather by setting WEATHER_PATH to an empty value.

### 4. Start the backend API

```powershell
uvicorn backend.api:app --reload
```

### 5. Start the frontend dashboard

```powershell
cd frontend
pnpm install
pnpm dev
```

## Weights

- models/stgcn_weights.pt is used by default for forecasting
- models/yolov8n.pt is used for CCTV detection

## Data generation scripts

- scripts/generate_timeseries_data.py writes frontend/public/timeseries-data.json
- scripts/generate_heatmap_data.py writes frontend/public/heatmap-data.json
- scripts/generate_sensor_locations.py writes frontend/public/sensor-locations.json
- scripts/generate_dashboard_data.py writes frontend-observable/src outputs

## Training

```powershell
python -u backend\train.py --epochs 30 --batch 64 --window 12 --horizon 3
```

## License

MIT
