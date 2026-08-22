"""HTTP server and local profile dashboard for BareLLM."""

from barellm.web.app import create_app, run_server

__all__ = ["create_app", "run_server"]
