import utils


PUBLIC_NAMES = [
    "tune_pipelines",
    "tune_single_model",
    "tune_all_models",
    "compare_cv_metrics",
    "get_model_pipeline",
    "get_hyperparameters",
    "plot_dynamic_stripplot_swarmplot",
]


def test_documented_names_are_importable():
    for name in PUBLIC_NAMES:
        assert hasattr(utils, name), f"utils.{name} is missing"


def test_all_exports_resolve():
    for name in utils.__all__:
        assert getattr(utils, name) is not None
