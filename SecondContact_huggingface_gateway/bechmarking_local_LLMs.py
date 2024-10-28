import time
import sys
from google_flan_T5_base import run_flant5_gpu
from google_flan_T5_Large import run_flant5_large_gpu
from gpt2_112M_Local import run_gpt2_gpu
from gpt_Neo_13B_Local import run_gptneo_gpu
import logging

# Suppress warnings from the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


def dhamma_printing(text, delay=0.01):
    """
    Simulates human-like typing by printing text character by character with a delay.

    Args:
        text (str): The text to be printed.
        delay (float): Delay in seconds between each character.
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # For newline after the text is printed


def benchmark_model(model_name, model_function, input_text, model_params):
    """
    Benchmarks a given model by running its generation function and measuring execution time.

    Args:
        model_name (str): The name of the model.
        model_function (callable): The function to run the model.
        input_text (str): The input prompt for the model.
        model_params (dict):
            - A dictionary containing the generation parameters that control the behavior of the text generation.
            - **Parameters:**
                - `max_new_tokens` (int):
                    - **Usage:**
                        - Higher values allow for longer responses but may increase computation time.
                        - Lower values restrict the length of the output.

                - `temperature` (float):
                    - **Description:** Controls the randomness of the generated text.
                    - **Usage:**
                        - Lower values (e.g., 0.4) make the output more deterministic and focused.
                        - Higher values (e.g., 1.0) increase randomness, making the output more diverse but potentially less coherent.
                    - **Recommendation:** Start with a value around `0.7` and adjust based on desired creativity vs. coherence.

                - `top_k` (int):
                    - **Description:** Limits the sampling pool to the top K most probable next tokens.
                    - **Usage:**
                        - Lower values (e.g., 30) restrict the model to a narrower set of options, enhancing coherence.
                        - Higher values (e.g., 50) allow more diversity but may introduce irrelevance.
                    - **Recommendation:** A balance between `30` and `50` is generally effective.
                    - **Example:** `50`

                - `repetition_penalty` (float):
                    - **Description:** Penalizes the model for repeating the same tokens or phrases.
                    - **Usage:**
                        - Values greater than `1.0` discourage repetition, promoting more varied outputs.
                        - Extremely high values might make the output unnatural.
                    - **Recommendation:** Values between `1.2` and `1.5` are typically effective.

    """
    print(f"\nBenchmarking {model_name}...")

    start_time = time.time()
    try:
        # Call the model function with the specified parameters
        generated_output = model_function(
            input_text=input_text,
            max_new_tokens=model_params.get("max_new_tokens", 150),
            temperature=model_params.get("temperature", 0.7),
            top_k=model_params.get("top_k", 50),
            repetition_penalty=model_params.get("repetition_penalty", 1.2)
        )
    except Exception as e:
        print(f"Error during benchmarking {model_name}: {e}")
        return
    end_time = time.time()

    execution_time = end_time - start_time

    print(f"\n{model_name} Output:")
    dhamma_printing(generated_output)

    print(f"\n{model_name} Execution Time: {execution_time:.2f} seconds\n{'-' * 60}")


def start_benchmarking_engine(input_prompt, models):
    """
    Starts the benchmarking process for all specified models with given parameters.

    Args:
        input_prompt (str): The input prompt for the models.
        models (list): A list of dictionaries containing model details and parameters.
    """
    for model in models:
        benchmark_model(
            model_name=model["name"],
            model_function=model["function"],
            input_text=input_prompt,
            model_params=model["params"]
        )

    print("\nBenchmarking Completed Successfully!")


def main():
    # Define the input prompt
    input_prompt = "To discern the reality of nature beyond the Mind's creation is"
    print(f"Benchmarked Prompt: {input_prompt}")
    # Define individual model parameters
    models = [
        {
            "name": "Google Flan-T5 Base",
            "function": run_flant5_gpu,
            "params": {
                "max_new_tokens": 100,  # Maximum number of tokens to generate
                "temperature": 0.7,  # Sampling temperature
                "top_k": 50,  # Top-K sampling
                "repetition_penalty": 1.2  # Repetition penalty
            }
        },
        {
            "name": "Google Flan-T5 Large",
            "function": run_flant5_large_gpu,
            "params": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_k": 50,
                "repetition_penalty": 1.2
            }
        },
        {
            "name": "GPT-2 112M",
            "function": run_gpt2_gpu,
            "params": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_k": 50,
                "repetition_penalty": 1.2
            }
        },
        {
            "name": "GPT-Neo 1.3B",
            "function": run_gptneo_gpu,
            "params": {
                "max_new_tokens": 150,
                "temperature": 0.9,  # Lower temperature to reduce hallucinations
                "top_k": 150,  # Slightly lower top_k
                "repetition_penalty": 1.3
            }
        }
    ]

    print("Starting Benchmark for All Models...\n" + "=" * 60)

    start_benchmarking_engine(input_prompt, models)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmarking Interrupted by User.")
        sys.exit(0)
