from blogbot import PROJECT_DIR

import logging 

from modal import Cls
from modal import App
import modal 

import pandas as pd  
from pii_prompt import inject_pii

from blogbot.configs import ModelParamsConfig
from blogbot.configs import DataConfig

from tqdm import tqdm

from utils import LLM_APP_NAME
from utils import data_generation_image
from utils import SYNTHETIC_DATA_GENERATOR_APP_NAME
from utils import DATASET_ID
from utils import VOLUME_CONFIG

from schemas import BlogPost

raw_data_path = PROJECT_DIR / "data/raw"

model_params = ModelParamsConfig().model_dump()
synthetic_data_config = DataConfig().model_dump()

Llm = Cls.lookup(LLM_APP_NAME, "Llm")
llm = Llm(config=model_params)

app = App(
    name=SYNTHETIC_DATA_GENERATOR_APP_NAME,
    image=data_generation_image,
)


deploy_params={
        "timeout": 60 * 60 * 2,
    }


@app.function(timeout=deploy_params["timeout"])
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data from the raw data path."""
    #take a stratified sample of blogposts by different groups for text diversity
    logging.info(f"Stratified sampling of data with groups {synthetic_data_config['groups']} and frac {synthetic_data_config['frac']}")
    data_sample = data.groupby(synthetic_data_config['groups']).sample(frac=synthetic_data_config['frac'], random_state=synthetic_data_config['random_state'], replace=False)

    #sample further 
    logging.info(f"Further sampling of data with sample size {synthetic_data_config['sample_size']}")    
    clean_data = (data_sample
            .drop_duplicates(subset=['text'])
            .sample(synthetic_data_config['sample_size'], random_state=synthetic_data_config['random_state'])
            .reset_index(drop=True)
            )
    
    clean_data = data[['id', 'text']].rename(columns={"text": "value"})
    logging.info(f"Cleaned data shape: {clean_data.shape}")
    
    return clean_data

@app.function(volumes=VOLUME_CONFIG, timeout=deploy_params["timeout"])
async def generate_synthetic_data(data: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic data from the dataframe to inject PII.
        - inject PII into the text
    """
    logging.info(f"Generating synthetic data with {len(data)} samples")

    texts = clean_data[['id', 'value']].to_dict(orient='records')
    target_schema = BlogPost.model_json_schema()

    fn_calls = []
    for text in tqdm(texts):
        compiled_prompt = inject_pii(document=text)
        fn_calls.append(
            llm._spawn.aio(
                id=text['id'],
                prompt_messages=compiled_prompt,
                target_schema=target_schema
            )
        )
    # Gather all the responses
    res = await modal.functions.gather(*fn_calls)

    loaded_responses = [r for r in res if not r.get("error")]

    responses_df = pd.DataFrame(loaded_responses).rename(columns={"text": "value"})

    logging.info(f"Generated synthetic data shape: {responses_df.shape}")
    
    return responses_df 

@app.function(volumes=VOLUME_CONFIG, timeout=deploy_params["timeout"])
def save_data(data: pd.DataFrame) -> None:
    """Save the generated data to the volume."""
    data_path = 'data' / synthetic_data_config['output_name']
    data.to_parquet(data_path, index=False)
    
    VOLUME_CONFIG['/data'].commit()

    logging.info(f"Data saved to {data_path}")

@app.function(volumes=VOLUME_CONFIG, timeout=deploy_params["timeout"])
def run_pipeline() -> None:
    """Run the pipeline."""
    data_path = raw_data_path / synthetic_data_config['input_name']
    data = pd.read_csv(data_path)

    #1. clean data
    clean_data_sample = clean_data.remote(data)

    #2. generate synthetic data for HALF of the data
    no_pii, pii = clean_data_sample[:len(clean_data_sample)//2], clean_data_sample[len(clean_data_sample)//2:]
    no_pii['flag'] = 0

    synthetic_data = generate_synthetic_data.remote(pii)

    all_data = pd.concat([no_pii, synthetic_data], ignore_index=True).sample(frac=1, random_seed=synthetic_data_config['random_seed']).reset_index(drop=True)
    all_data['dataset_id'] = DATASET_ID 

    #3. save data
    save_data.remote(all_data)
    logging.info(f"Pipeline completed. Data saved as {synthetic_data_config['output_name']}")


@app.local_entrypoint()
def main():
    """Main function to run the pipeline.
    
    modal run --detach blogbot/synthetic_data_generator.py
    """
    run_pipeline.remote()

