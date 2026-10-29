def test_utils_imports():
    import utils

    assert callable(utils.check_data_information)
    assert callable(utils.eval_classification)
    assert callable(utils.eval_regression)
