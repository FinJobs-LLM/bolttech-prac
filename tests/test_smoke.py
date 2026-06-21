"""Lightweight smoke tests for CI.

These avoid network, the database, the OpenAI key, and the trained-model
artifact — they only verify that the package imports, the config contract holds,
the compatibility shims resolve, and the FastAPI apps construct.
"""
import sys
from pathlib import Path

# Mirror the runtime convention (`--app-dir src`): put src/ on the path.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def test_config_contract():
    import config
    assert config.TARGET == "status"
    assert config.POSITIVE_LABEL == "Declined" and config.NEGATIVE_LABEL == "Completed"
    assert config.EXCLUDE_COLS == ["other", "issueDesc"]
    assert config.RANDOM_STATE == 42
    assert set(config.ALL_MODELS) == {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}


def test_model_factory_shim_matches_ml():
    # The pickled artifact is `model_factory.ClaimModel`; the shim must resolve to
    # the same class object as ml.model_factory.ClaimModel.
    import model_factory
    import ml.model_factory as mlmf
    assert model_factory.ClaimModel is mlmf.ClaimModel


def test_run_pipeline_shim_matches_ml():
    import run_pipeline
    import ml.run_pipeline as mlrp
    assert run_pipeline.main is mlrp.main


def test_prompts_present():
    import prompts
    for name in ("SYSTEM_PROMPT", "ADJUSTER_SYSTEM_PROMPT", "CUSTOMER_SYSTEM_PROMPT"):
        assert getattr(prompts, name)


def test_fastapi_apps_construct():
    import dashboard_api
    import prediction_service_api
    assert dashboard_api.app is not None
    assert prediction_service_api.app is not None


def test_llm_config_versioned():
    # Model name + prompts come from the version-controlled YAML, not hardcoded.
    import prompts
    import llm_explain
    assert prompts.MODEL and prompts.VERSION
    g = prompts.get_generation("model_explanation")
    assert "temperature" in g and "max_tokens" in g
    assert llm_explain.MODEL_NAME == prompts.MODEL
