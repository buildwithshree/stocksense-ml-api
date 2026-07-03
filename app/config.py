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

    alpha_vantage_key: str = "T6UUZET8RISCTHQ9"

settings = Settings()
