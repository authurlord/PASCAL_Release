"""Centralized configuration.

Settings are loaded in this priority (highest wins):
  1. Environment variables (e.g. PATIENCE=6 python -m orchestrator.runner ...)
  2. .env file in project root (user-supplied, gitignored)
  3. Defaults defined below

The launcher scripts (scripts/run_eval.sh, scripts/start_vllm_*.sh)
export every variable they need.  The .env file is optional; create
one with GOOGLE_API_KEY=... if you prefer file-based config.
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so litellm/openai can read OPENAI_API_KEY etc.
load_dotenv()


# Release layout: <release_root>/src/shared/config.py → parent.parent.parent
# resolves to the release root that contains data/, dumps/, scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # LLM provider
    llm_provider: str = "litellm"

    # PostgreSQL
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "root"
    pg_password: str = "123123"
    pg_minconn: int = 1
    pg_maxconn: int = 64

    # Service ports
    system_agent_port: int = 6000
    user_sim_port: int = 6001
    db_env_port: int = 6002

    # Models. The PASCAL anchor uses Qwen3.6-35B-A3B-FP8 via local vLLM
    # for the system agent and Gemini 2.5 Flash Lite for the user
    # simulator. See docs/MODEL_CARDS.md and scripts/run_eval.sh.
    user_sim_model: str = "gemini/gemini-2.5-flash-lite"
    system_agent_model: str = "openai/qwen3.6-35b"

    # LiteLlm proxy (optional — set if using a LiteLlm proxy server)
    litellm_api_base: str = ""
    litellm_api_key: str = ""

    # Dataset: "lite" or "full"
    dataset: str = "lite"

    # User simulator prompt version: "v1" (legacy) or "v2" (recommended)
    prompt_version: str = "v2"

    # Budget / turns
    patience: int = 3

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / f"bird-interact-{self.dataset}-hf-meta"

    @property
    def data_path(self) -> str:
        return str(self.data_dir / "bird_interact_data.jsonl")

    @property
    def db_data_path(self) -> str:
        return str(self.data_dir)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
