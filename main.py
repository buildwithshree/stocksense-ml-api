import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("StockSense ML API starting — version %s", settings.model_version)
    yield
    logger.info("StockSense ML API shutting down")


app = FastAPI(
    title="StockSense ML API",
    version=settings.model_version,
    description="Stock price prediction and risk scoring microservice. "
            "For educational and research purposes only. "
            "Not financial advice. Do not use for real investment decisions.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to Spring Boot service URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
