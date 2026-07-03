from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database — same Neon instance as Spring Boot
    database_url: str

    # Data fetch
    default_period_years: int = 5
    min_rows_for_lstm: int = 500
    min_rows_for_xgboost: int = 200
    min_rows_for_rf: int = 100
    cache_stale_hours: int = 24

    # Model versioning
    model_version: str = "v1.0"

    # Server
    port: int = 8000
    workers: int = 1

    # Primary OHLCV + company-name data source.
    # Free tier: 800 requests/day, 8/min, full daily history since listing.
    twelve_data_key: str = ""

    # Deprecated — no longer called anywhere in the codebase after the
    # Twelve Data migration. Left here (unused) so Render's existing env
    # var doesn't cause a startup crash. Safe to delete once confirmed
    # nothing else references it.
    alpha_vantage_key: str = ""

settings = Settings()