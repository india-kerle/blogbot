import pytest
from unittest import mock
from blogbot.utils import get_model

import pytest
from unittest import mock
from blogbot.utils import get_model, VOLUME_CONFIG

@pytest.mark.asyncio
def test_get_model_real_logic_with_mocked_dependencies():
    model_name = "mock_model"

    with mock.patch("huggingface_hub.snapshot_download", side_effect=[FileNotFoundError(), "/mock/path/model"]) as mock_download, \
         mock.patch("blogbot.utils.logging") as mock_log, \
         mock.patch.object(VOLUME_CONFIG["/pretrained"], "commit") as mock_commit, \
         mock.patch.object(VOLUME_CONFIG["/pretrained"], "reload") as mock_reload, \
         mock.patch("time.time", side_effect=[10000.0, 12345.0]):

        path = get_model(model_name)

        assert path == "/mock/path/model"
        assert mock_download.call_count == 2
        mock_log.info.assert_any_call(
            f"Model {model_name} not found locally. Downloading from Hugging Face Hub."
        )
        mock_log.info.assert_any_call(
            f"Model downloaded in {12345.0 - 10000.0:.2f} seconds.",
        )

        mock_commit.assert_called_once()
        mock_reload.assert_called_once()