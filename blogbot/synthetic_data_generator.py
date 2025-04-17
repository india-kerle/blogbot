from modal import Cls
from modal import App
import modal 

import pandas as pd  
from pii_prompt import inject_pii

from blogbot.configs import ModelParamsConfig
from blogbot.configs import DataConfig

from pathlib import Path

from utils import LLM_APP_NAME
from utils import SYNTHETIC_DATA_GENERATOR_APP_NAME
from utils import DATASET_ID
from utils import VOLUME_CONFIG
from utils import data_generation_image

model_params = ModelParamsConfig().model_dump()
synthetic_data_config = DataConfig().model_dump()

Llm = Cls.from_name(LLM_APP_NAME, "Llm")
llm = Llm(config=model_params)

app = App(
    name=SYNTHETIC_DATA_GENERATOR_APP_NAME,
    image=data_generation_image,
)


deploy_params={
        "timeout": 60 * 60 * 2,
    }

remote_path = "/input-data"

@app.function(timeout=deploy_params["timeout"])
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data from the raw data path."""
    #take a stratified sample of blogposts by different groups for text diversity
    data_sample = data.groupby(synthetic_data_config['groups']).sample(frac=synthetic_data_config['frac'], random_state=synthetic_data_config['random_seed'], replace=False)

    #sample further 
    clean_data = (data_sample
            .drop_duplicates(subset=['text'])
            .sample(synthetic_data_config['sample_size'], random_state=synthetic_data_config['random_seed'])
            .reset_index(drop=True)
            )
    
    clean_data = clean_data[['id', 'text']].rename(columns={"text": "value"})
    
    return clean_data

@app.function(timeout=deploy_params["timeout"])
async def generate_synthetic_data(data: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic data to inject PII.
        - inject PII into the text
    """
    import asyncio

    texts = data[['id', 'value']].to_dict(orient='records')

    fn_calls = []
    for text in texts:
        blog_snippet = f"{text['value'][:synthetic_data_config['max_length']]}..."
        compiled_prompt = inject_pii(document=blog_snippet)
        fn_calls.append(
            llm.generate._experimental_spawn.aio(
                id=text['id'],
                prompt_messages=compiled_prompt,
            )
        )

    fn_id = await asyncio.gather(*fn_calls)
    responses = modal.FunctionCall.gather(*fn_id)
    
    # Gather all the responses that did not return an error
    loaded_responses = [r for r in responses if not r.get("error")]

    return pd.DataFrame(loaded_responses)
    
@app.function(volumes=VOLUME_CONFIG, timeout=deploy_params["timeout"])
def save_data(data: pd.DataFrame) -> None:
    """Save the generated data to the volume."""
    output_dir = Path("/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_data_path = output_dir / synthetic_data_config['output_name'] # Construct full path

    data.to_parquet(full_data_path, index=False)
    
    VOLUME_CONFIG['/data'].commit()

@app.function(volumes=VOLUME_CONFIG, timeout=deploy_params["timeout"])
def run_pipeline() -> None:
    """Run the pipeline."""
    data_path = f"{remote_path}/raw/{synthetic_data_config['input_name']}"
    data = pd.read_csv(data_path)
    print(f"Raw data loaded with {len(data)} samples.")

    #1. clean data
    clean_data_sample = clean_data.remote(data)
    print(f"Clean data with {len(clean_data_sample)} samples.")

    #2. generate synthetic data for HALF of the data
    no_pii, pii = clean_data_sample[:len(clean_data_sample)//2], clean_data_sample[len(clean_data_sample)//2:]
    no_pii['flag'] = 0
    print(f"Data split into {len(no_pii)} samples without PII and {len(pii)} samples with PII.")
    
    print(f"Generating PII-injected data for {len(pii)} samples.")
    synthetic_data = generate_synthetic_data.remote(pii)
    print(f"Generated synthetic data for {len(synthetic_data)} samples.")

    all_data = pd.concat([no_pii, synthetic_data], ignore_index=True).sample(frac=1, random_state=synthetic_data_config['random_seed']).reset_index(drop=True)
    all_data['dataset_id'] = DATASET_ID 
    print(f"Data has {len(all_data)} samples.")
    print(all_data.head())

    #3. save data
    save_data.remote(all_data)
    print(f"Data saved with {len(all_data)} samples.")

    print(f"Pipeline completed. Data saved to {synthetic_data_config['output_name']}")

@app.local_entrypoint()
def main():
    """Main function to run the pipeline.
    
    modal run --detach blogbot/synthetic_data_generator.py
    """
    run_pipeline.remote()