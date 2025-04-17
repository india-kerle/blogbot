from modal import Image
from modal import Volume
import logging 
from blogbot import PROJECT_DIR

LLM_APP_NAME = 'open-source-llm'
SYNTHETIC_DATA_GENERATOR_APP_NAME = 'blogbot-synthetic-data-generator'
DATASET_ID = "12345"

vllm_image = (
    Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "VLLM_USE_V1": "1"})  # faster model transfers
)

training_image = Image.debian_slim(python_version="3.12").pip_install(
    "transformers==4.38.0",
    "datasets==2.14.4",
    "torch==2.5.0")

data_generation_image = Image.debian_slim(python_version="3.12").pip_install(
    "pandas==2.2.3",
    "numpy==1.26.4",
    "pydantic==2.11.3").add_local_python_source(
        "blogbot",
        "utils",
        "pii_prompt"
    ).add_local_dir(
        PROJECT_DIR / "data",
        remote_path="/input-data",
    )

pretrained_llms_vol = Volume.from_name("pretrained-llms-vol", create_if_missing=True)
data_vol = Volume.from_name("blogbot-data-vol", create_if_missing=True)
fine_tuned_vol = Volume.from_name("fine-tuned-vol", create_if_missing=True)

VOLUME_CONFIG = {
    "/pretrained": pretrained_llms_vol,
    "/data": data_vol,
    "/fine-tuned": fine_tuned_vol,
}

def get_model(model_name: str) -> str:
    """Download the model weights into the image."""
    import time

    from huggingface_hub import snapshot_download

    start = time.time()

    try:
        path = snapshot_download(model_name, local_files_only=True)
        
    except FileNotFoundError:
        logging.info(
            f"Model {model_name} not found locally. Downloading from Hugging Face Hub."
        )
        path = snapshot_download(model_name)

        VOLUME_CONFIG["/pretrained"].commit()

    end = time.time()
    logging.info(
            f"Model downloaded in {end - start:.2f} seconds.",
        )

    VOLUME_CONFIG["/pretrained"].reload()

    return path

