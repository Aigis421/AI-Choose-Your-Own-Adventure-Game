from transformers import pipeline


class LLMUtil:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, max_tokens: int = 250):
        response = self.generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        return response[0]["generated_text"]