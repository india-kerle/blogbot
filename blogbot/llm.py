from modal import App
from modal import Secret
from utils import vllm_image # Assuming this defines the image correctly
from utils import get_model
from utils import LLM_APP_NAME
from utils import VOLUME_CONFIG
from pydantic_core import from_json # Keep this
from typing import List, Dict # Add typing imports
import modal

app = App(LLM_APP_NAME,
          secrets=[Secret.from_name("huggingface-token")])

deploy_params={
        "gpu": ["A100-40GB"],
        "timeout": 60 * 60 * 2,
        "scaledown_window": 60 * 3,
        "allow_concurrent_inputs": 180,
        "max_containers": 5,
    }

@app.cls(
    gpu=deploy_params["gpu"],
    timeout=deploy_params["timeout"],
    scaledown_window=deploy_params["scaledown_window"],
    allow_concurrent_inputs=deploy_params["allow_concurrent_inputs"],
    max_containers=deploy_params["max_containers"],
    image=vllm_image,
    volumes=VOLUME_CONFIG,  # type: ignore
)
class Llm:
    """A class to generate structured data using an open-source LLM."""
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    @modal.enter()
    def load_model(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from transformers import AutoTokenizer

        model = self.config["hf_model_id"]
        path = get_model(model)

        engine_args = AsyncEngineArgs(
                model=path,
                max_model_len=self.config["max_seq_length"],
                dtype=self.config["dtype"],
                tensor_parallel_size=self.config["tensor_parallel_size"],
                gpu_memory_utilization=self.config["gpu_memory_utilization"],
                enforce_eager=self.config["enforce_eager"],
                quantization=self.config["quantization"],
                load_format=self.config.get("load_format", "auto"),
                disable_log_stats=self.config["disable_log_stats"],
                disable_log_requests=self.config["disable_log_requests"],
                enable_chunked_prefill=self.config["enable_chunked_prefill"],
                enable_prefix_caching=self.config["enable_prefix_caching"],
                max_seq_len_to_capture=self.config["max_seq_length"],
                max_num_seqs=self.config["max_num_seqs"],
                enable_lora=self.config["enable_lora"],
                max_num_batched_tokens=self.config["max_num_batched_tokens"],
            )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["hf_model_id"])

    @modal.method()
    async def generate(
        self,
        id: str,
        prompt_messages: List[Dict[str, str]],
        target_schema: dict,
        ) -> dict:
        """Generate structured data using an LLM.

        Args:
        ---
        id (str): The ID of the request.
        prompt_messages (List[Dict[str, str]]): The structured messages for the chat template.
        target_schema (dict): The JSON schema for guided decoding.

        Returns:
        ---
        dict: The generated data.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from vllm.utils import random_uuid
        
        request_id = random_uuid()
        
        compiled_prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        

        guided_decoding_params = GuidedDecodingParams(
            json=target_schema, 
            backend="outlines", 
            whitespace_pattern=None)
        
        sampling_params = SamplingParams(
                guided_decoding=guided_decoding_params,
                max_tokens=self.config['max_tokens'],
                temperature=0.0,
                top_p=0.9,
            )

        result_generator = self.engine.generate(
            compiled_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        output = {} # Initialize output
        final_output = None # Initialize final_output
        try:
            async for request_output in result_generator:
                final_output = request_output

            if final_output and final_output.outputs:
                vllm_output_text = final_output.outputs[0].text
                try:
                    output = from_json(vllm_output_text, allow_partial=False)
                except Exception as e:
                    print(f"Error parsing JSON output: {e}")
                    output = {"error": "Failed to parse generated output"}

        except Exception as e:
            print(f"Error processing generation result: {e}")
            output = {"error": "Failed to parse generated output"}

        output.update({"id": id})
        output.update({"label": 1})
        
        return output