from pydantic import BaseModel
from pydantic import Field

class ModelParamsConfig(BaseModel):
    """Config for model parameters for LLM."""
    hf_model_id: str = Field(default="Qwen/Qwen2-7B-Instruct")
    max_seq_length: int = Field(default=9088)
    max_num_batched_tokens: int = Field(default=9088)
    dtype: str = Field(default="auto")
    tensor_parallel_size: int = Field(default=1)
    enforce_eager: bool = Field(default=False)
    quantization: str | None = Field(default=None)
    load_format: str | None = Field(default="auto")
    disable_log_stats: bool = Field(default=True)
    disable_log_requests: bool = Field(default=True)
    enable_chunked_prefill: bool = Field(default=True)
    enable_prefix_caching: bool = Field(default=True)
    gpu_memory_utilization: float = Field(default=0.9)
    max_num_seqs: int = Field(default=60)
    enable_lora: bool = Field(default=False)
    max_tokens: int = Field(default=9088 - 2048)

class DataConfig(BaseModel):
    """Config for synthetic data generator"""
    groups: list[str] = Field(default=['gender', 'age', 'topic'])
    frac: float = Field(default=0.005)
    random_state: int = Field(default=42)
    sample_size: int = Field(default=5)
    data_path: str = Field(default="data/raw")
    max_length: int = Field(default=1000)
    output_name: str = Field(default="synthetic_data.parquet")
    input_name: str = Field(default="blogtext.csv")

class ModelTrainingConfig(BaseModel):
    """Config for training smaller BERT model downstream with synthetic data."""
    test_size: float = Field(default=0.1)
    random_state: int = Field(default=42)
    model_name: str = Field(default="bert-base-uncased")
    max_length: int = Field(default=512)
    num_labels: int = Field(default=2)
    output_dir: str = Field(default="output")
    num_train_epochs: int = Field(default=3)
    learning_rate: float = Field(default=2e-5)
    train_batch_size: int = Field(default=32)
    eval_batch_size: int = Field(default=32)
    weight_decay: float = Field(default=0.01)