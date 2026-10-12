from utils.config import data_path, load_config


def test_config_has_seed():
    assert load_config()["seed"] == 42


def test_data_folders_exist():
    for kind in ("raw", "interim", "processed"):
        assert data_path(kind).is_dir()
