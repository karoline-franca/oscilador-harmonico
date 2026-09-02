"""Pipeline registry."""

from kedro.pipeline import Pipeline
from oscilador_fitz_hugh_nagumo.pipelines.p00_data_generating.pipeline import create_pipeline as create_pipeline_data
from oscilador_fitz_hugh_nagumo.pipelines.p01_mlp.pipeline import create_pipeline as create_pipeline_mlp


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    return {
        "__default__": create_pipeline_data(),
        "p00_data_generating": create_pipeline_data(),
        "p01_mlp": create_pipeline_mlp(),
    }