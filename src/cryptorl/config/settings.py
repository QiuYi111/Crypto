"""Core settings configuration using Pydantic."""

import os
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project settings
    project_name: str = "CryptoRL Agent"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Binance API settings
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    binance_futures_url: str = Field(
        default="https://testnet.binancefuture.com", env="BINANCE_FUTURES_URL"
    )

    # Database settings
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="cryptorl", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="market_data", env="INFLUXDB_BUCKET")

    postgresql_url: str = Field(default="", env="POSTGRESQL_URL")
    postgresql_host: str = Field(default="localhost", env="POSTGRESQL_HOST")
    postgresql_port: int = Field(default=5432, env="POSTGRESQL_PORT")
    postgresql_user: str = Field(default="cryptorl", env="POSTGRESQL_USER")
    postgresql_password: str = Field(default="", env="POSTGRESQL_PASSWORD")
    postgresql_db: str = Field(default="cryptorl", env="POSTGRESQL_DB")

    # LLM settings
    llm_model_name: str = Field(default="microsoft/DialoGPT-medium", env="LLM_MODEL_NAME")
    llm_device: str = Field(default="cuda", env="LLM_DEVICE")
    llm_max_tokens: int = Field(default=512, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    llm_batch_size: int = Field(default=8, env="LLM_BATCH_SIZE")
    llm_load_in_4bit: bool = Field(default=True, env="LLM_LOAD_IN_4BIT")
    llm_max_news_articles: int = Field(default=10, env="LLM_MAX_NEWS_ARTICLES")
    
    # LLM provider settings
    llm_provider: str = Field(default="local", env="LLM_PROVIDER")
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", env="DEEPSEEK_BASE_URL")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_cx: str = Field(default="", env="GOOGLE_CX")

    # Search API settings
    serpapi_key: str = Field(default="", env="SERPAPI_KEY")
    google_search_api_key: str = Field(default="", env="GOOGLE_SEARCH_API_KEY")
    google_search_cx: str = Field(default="", env="GOOGLE_SEARCH_CX")
    
    # China-compatible search APIs
    baidu_api_key: str = Field(default="", env="BAIDU_API_KEY")

    # RL settings
    rl_hidden_dim: int = Field(default=256, env="RL_HIDDEN_DIM")
    rl_num_layers: int = Field(default=4, env="RL_NUM_LAYERS")
    rl_use_mamba: bool = Field(default=True, env="RL_USE_MAMBA")
    rl_continuous_actions: bool = Field(default=True, env="RL_CONTINUOUS_ACTIONS")
    rl_initial_balance: float = Field(default=10000.0, env="RL_INITIAL_BALANCE")
    rl_max_position_size: float = Field(default=1.0, env="RL_MAX_POSITION_SIZE")
    rl_trading_fee: float = Field(default=0.001, env="RL_TRADING_FEE")
    rl_max_episode_length: int = Field(default=30, env="RL_MAX_EPISODE_LENGTH")
    rl_training_days: int = Field(default=365, env="RL_TRAINING_DAYS")
    rl_replay_buffer_size: int = Field(default=100000, env="RL_REPLAY_BUFFER_SIZE")
    rl_learning_rate: float = Field(default=3e-4, env="RL_LEARNING_RATE")
    rl_gamma: float = Field(default=0.99, env="RL_GAMMA")

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/cryptorl.log", env="LOG_FILE")

    # Trading settings
    trading_symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], env="TRADING_SYMBOLS")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    max_leverage: int = Field(default=10, env="MAX_LEVERAGE")
    risk_free_rate: float = Field(default=0.02, env="RISK_FREE_RATE")
    slippage_penalty: float = Field(default=0.001, env="SLIPPAGE_PENALTY")
    transaction_fee: float = Field(default=0.0004, env="TRANSACTION_FEE")

    # Derived settings
    @property
    def project_root(self) -> Path:
        """Return project root directory."""
        return Path(__file__).parent.parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Return data directory path."""
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """Return models directory path."""
        return self.project_root / "models"

    @property
    def logs_dir(self) -> Path:
        """Return logs directory path."""
        return self.project_root / "logs"

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()