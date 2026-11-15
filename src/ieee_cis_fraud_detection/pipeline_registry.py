"""Project pipelines."""

from kedro.pipeline import Pipeline

from ieee_cis_fraud_detection.pipelines import data_processing, data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_processing": data_processing.create_pipeline(),
        "data_science": data_science.create_pipeline(),
        "tree_model": data_science.pipeline.create_tree_model_pipeline(),
        "lightgbm_model": data_science.pipeline.create_lightgbm_model_pipeline(),
        "xgboost_model": data_science.pipeline.create_xgboost_model_pipeline(),
        "model_comparison": data_science.pipeline.create_model_comparison_pipeline()
    }

    from ieee_cis_fraud_detection.pipelines.data_processing.staged import create_pipeline as create_staged_pipeline
    from ieee_cis_fraud_detection.pipelines.data_science import docker_pipeline
    pipelines["docker_model_comparison"] = (
        create_staged_pipeline() + docker_pipeline.create_pipeline()
    )
    pipelines["docker_staged_data"] = create_staged_pipeline()
    pipelines["docker_split"] = data_science.pipeline.create_split_pipeline()
    pipelines["docker_tree_only"] = data_science.tree_pipeline.create_pipeline()
    pipelines["docker_lightgbm_only"] = docker_pipeline.create_lightgbm_branch()
    pipelines["docker_xgboost_only"] = docker_pipeline.create_xgboost_branch()
    pipelines["docker_saved_evaluation"] = docker_pipeline.create_saved_evaluation_pipeline()
    pipelines["docker_finalize"] = docker_pipeline.create_finalize_pipeline()

    pipelines["__default__"] = (
        pipelines["data_processing"] + pipelines["model_comparison"]
    )

    return pipelines
