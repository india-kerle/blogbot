from blogbot.configs import ModelParamsConfig
from blogbot.configs import DataConfig
from blogbot.configs import ModelTrainingConfig
import pytest
from pydantic import ValidationError

# --- ModelParamsConfig Tests ---

def test_model_params_config_defaults():
    config = ModelParamsConfig()
    assert config.hf_model_id == "Qwen/Qwen2-7B-Instruct"
    assert config.max_seq_length == 9088
    assert config.dtype == "auto"
    assert config.tensor_parallel_size == 1
    assert config.gpu_memory_utilization == 0.9
    assert config.enable_chunked_prefill is True

def test_model_params_config_custom_values():
    config = ModelParamsConfig(hf_model_id="custom-model", tensor_parallel_size=2)
    assert config.hf_model_id == "custom-model"
    assert config.tensor_parallel_size == 2

def test_model_params_config_invalid_type():
    with pytest.raises(ValidationError):
        ModelParamsConfig(max_seq_length="not-an-int")


# --- DataConfig Tests ---

def test_data_config_defaults():
    config = DataConfig()
    assert config.groups == ['gender', 'age', 'topic']
    assert config.frac == 0.1
    assert config.random_state == 42
    assert config.output_name.endswith(".parquet")

def test_data_config_custom_values():
    config = DataConfig(groups=["region"], frac=0.5, sample_size=500)
    assert config.groups == ["region"]
    assert config.frac == 0.5
    assert config.sample_size == 500

def test_data_config_invalid_frac():
    with pytest.raises(ValidationError):
        DataConfig(frac="high")  # Should be a float


# --- ModelTrainingConfig Tests ---

def test_training_config_defaults():
    config = ModelTrainingConfig()
    assert config.test_size == 0.1
    assert config.num_labels == 2
    assert config.train_batch_size == 32
    assert config.model_name == "bert-base-uncased"

def test_training_config_custom_values():
    config = ModelTrainingConfig(model_name="roberta-base", learning_rate=5e-5)
    assert config.model_name == "roberta-base"
    assert config.learning_rate == 5e-5

def test_training_config_invalid_type():
    with pytest.raises(ValidationError):
        ModelTrainingConfig(num_train_epochs="three")