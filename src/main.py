"""Entry point for the ML Monitoring API server."""

import uvicorn

from src.utils.config import load_config
from src.utils.logger import setup_logging


def main() -> None:
    """Start the ML monitoring API server."""
    setup_logging()
    config = load_config()
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
