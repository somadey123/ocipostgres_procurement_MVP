import os
from pathlib import Path

import oci
import psycopg
from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv()


def embed_dim() -> int:
    return int(os.getenv("EMBED_DIM", "384"))


def oci_embed_model_id() -> str:
    return os.getenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-light-v3.0")


def oci_config():
    config_file = os.getenv("OCI_CONFIG_FILE")
    profile = os.getenv("OCI_PROFILE", "DEFAULT")
    cfg_path = Path(config_file).expanduser() if config_file else (Path.home() / ".oci" / "config")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"OCI config file not found at: {cfg_path}. "
            "Set OCI_CONFIG_FILE to your real path, e.g. /home/opc/.oci/config"
        )
    return oci.config.from_file(str(cfg_path), profile)


def oci_embed_client():
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=oci_config(),
        service_endpoint=os.environ["OCI_ENDPOINT"],
        retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
    )


def oci_object_storage_client():
    return oci.object_storage.ObjectStorageClient(oci_config())


def pg_conn():
    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )


def oci_auth_file_location() -> str:
    config_file = os.getenv("OCI_CONFIG_FILE")
    path = Path(config_file).expanduser() if config_file else (Path.home() / ".oci" / "config")
    return str(path)
