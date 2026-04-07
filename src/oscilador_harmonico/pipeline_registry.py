"""Pipeline registry."""

from kedro.pipeline import Pipeline
from oscilador_harmonico.pipeline import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    return {
        "__default__": create_pipeline(),
    }