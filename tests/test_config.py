import pytest
import torch

from barellm.config import load_settings


def test_settings_defaults_are_server_safe() -> None:
    settings = load_settings(environment={}, load_dotenv_file=False)

    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.model_id == "Qwen/Qwen3-0.6B"
    assert settings.enable_profile_api is True


def test_settings_environment_overrides_defaults() -> None:
    settings = load_settings(
        environment={
            "BARELLM_MODEL_ID": "example/model",
            "BARELLM_DEVICE": "cpu",
            "BARELLM_DTYPE": "float16",
            "BARELLM_HOST": "127.0.0.1",
            "BARELLM_PORT": "9000",
            "BARELLM_PROFILE_ROOT": "/tmp/barellm-profiles",
            "BARELLM_ENABLE_PROFILE_API": "off",
        },
        load_dotenv_file=False,
    )

    assert settings.model_id == "example/model"
    assert settings.device == "cpu"
    assert settings.dtype is torch.float16
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.profile_root.as_posix() == "/tmp/barellm-profiles"
    assert settings.enable_profile_api is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("BARELLM_PORT", "0"),
        ("BARELLM_PORT", "not-a-port"),
        ("BARELLM_ENABLE_PROFILE_API", "sometimes"),
    ],
)
def test_settings_reject_invalid_values(key: str, value: str) -> None:
    with pytest.raises(ValueError):
        load_settings(
            environment={"BARELLM_DEVICE": "cpu", key: value},
            load_dotenv_file=False,
        )
