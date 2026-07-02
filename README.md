# StockSense — Python ML API (Phase 2)

## Folder location in your workspace
```
stocksense-workspace/
├── stocksense-backend/     ← Spring Boot (Phase 1)
└── stocksense-ml-api/      ← THIS folder
```

## 1. Open in VS Code
VS Code is already open at `stocksense-workspace/`. The `stocksense-ml-api/` folder is already visible in the Explorer panel on the left.

## 2. Create virtual environment
Open VS Code terminal (Ctrl+`) and run:
```bash
cd stocksense-ml-api
python -m venv venv
```
Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

You'll see `(venv)` in your terminal prompt.

## 3. Install dependencies
```bash
pip install -r requirements.txt
```
First install takes 3–5 minutes (PyTorch is large).

## 4. Create your .env file
Copy `.env.example` to `.env` and fill in your Neon DB URL:
```
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
MODEL_VERSION=v1.0
PORT=8000
```

## 5. Run locally
```bash
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/health` — should return `{"status":"UP","version":"v1.0"}`.

Test a prediction:
```bash
curl http://localhost:8000/predict/TCS.NS
```
First call trains the model (~10-30 seconds). Subsequent calls serve from cache.

## 6. Render deployment
- New Web Service → connect your `stocksense-ml-api` GitHub repo
- Runtime: Python 3.12
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add env var: `DATABASE_URL` (your Neon connection string)

## Commit sequence (do this after extracting zip)
```bash
cd stocksense-ml-api
git init
git remote add origin https://github.com/buildwithshree/stocksense-ml-api.git

git add requirements.txt .gitignore .env.example
git commit -m "chore: initialise Python ML API project"

git add main.py app/config.py
git commit -m "feat: FastAPI app entry point and settings config"

git add app/db/
git commit -m "feat(db): SQLAlchemy database layer with stock_cache read/write"

git add app/pipeline/
git commit -m "feat(data): yfinance OHLCV fetcher with Indian and US ticker support"

git add app/features/
git commit -m "feat(features): technical indicator engineering (RSI, MACD, EMA, ATR, BB, Stochastic)"

git add app/models/selector.py
git commit -m "feat(ml): dynamic model selector with TimeSeriesSplit — Ridge, RF, XGBoost, LSTM"

git add app/models/risk_scorer.py
git commit -m "feat(ml): mathematical risk score formula — volatility, drawdown, ATR, uncertainty, volume"

git add app/models/backtester.py
git commit -m "feat(ml): walk-forward backtesting engine with direction accuracy"

git add app/utils/
git commit -m "feat(cache): thread-safe in-memory model cache with 24h staleness"

git add app/api/
git commit -m "feat(api): FastAPI prediction and backtest routes with full response schema"

git push -u origin main
```
