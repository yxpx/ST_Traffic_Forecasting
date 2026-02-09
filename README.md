# ST Traffic Forecasting

This project implements a traffic forecasting system using a Spatio-Temporal Graph Convolutional Network (ST-GCN). It features a FastAPI backend for serving predictions and an interactive Next.js dashboard for visualization. The system is capable of fusing historical traffic graph signals with optional real-time CCTV camera inputs for enhanced accuracy.

This project was developed at Datathon 2026, organized by [DataZen](https://www.datazen.in/) at K. J. Somaiya School of Engineering (KJSSE), where it was awarded First Prize in the Core ML Category.

**Demo Video:** [https://youtu.be/3evyjJfxDd8](https://youtu.be/3evyjJfxDd8)

## Features

- Spatio-Temporal Graph Convolutional Network for traffic speed forecasting
- FastAPI backend with real-time prediction endpoints
- Interactive Next.js dashboard with traffic visualizations
- Optional CCTV integration using YOLOv8n for vehicle detection
- Weather data fusion for improved prediction accuracy
- Pre-trained model weights included for immediate use

## Repository Structure

```
ST_Traffic_Forecasting/
├── backend/                 # FastAPI backend and training code
├── frontend/                # Next.js dashboard application
├── models/                  # Model definitions and weights
├── scripts/                 # Data generation and utility scripts
├── data/                    # Raw datasets and mappings
├── assets/                  # Computer vision utilities
├── reports/                 # Training metrics
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
```

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- uv
- pnpm

## Backend Setup

```powershell
# Clone the repository
git clone https://github.com/yxpx/ST_Traffic_Forecasting.git
cd ST_Traffic_Forecasting

# Create and activate virtual environment
uv venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
uv pip install -r requirements.txt

# Download the METR-LA dataset
python scripts/download_metr_la.py

# Start the backend server
uvicorn backend.api:app --reload
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

**Dataset Source:** [METR-LA Dataset on Kaggle](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)

## Frontend Setup

```powershell
cd frontend
pnpm install
pnpm dev
```

The dashboard will be available at `http://localhost:3000`.

## Weather Data

The system supports weather data fusion for improved predictions. The default path is `data/raw/weather_CA_2019.csv`. You can configure this via the `WEATHER_PATH` environment variable or disable it by setting it to an empty value.

If your weather data does not align with the METR-LA time range, the features may be biased. For clean experiments, provide a temporally matched file or disable weather integration.

## Model Weights

The repository includes pre-trained weights that are loaded automatically:

- `models/stgcn_weights.pt` for traffic forecasting
- `models/yolov8n.pt` for CCTV vehicle detection

## Data Generation Scripts

Run these scripts to regenerate dashboard visualizations:

```powershell
python scripts/generate_timeseries_data.py
python scripts/generate_heatmap_data.py
python scripts/generate_sensor_locations.py
```

## Model Training

```powershell
python -u backend/train.py --epochs 30 --batch 64 --window 12 --horizon 3
```

- `--epochs`: Number of training iterations over the full dataset.
- `--batch`: Number of samples processed before updating model weights.
- `--window`: Number of past time steps used as input for prediction.
- `--horizon`: Number of future time steps to predict.
- `--weather`: Flag to include weather features during training.

Training metrics are saved to `reports/metrics.csv`.
