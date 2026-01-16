import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        self.base_model = "microsoft/phi-2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(
            base,
            "./",
        )

        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Enter a learning prompt")
    ) -> str:

        text = (
            "<instruction>\n"
            f"{prompt}\n"
            "</instruction>\n"
            "<response>\n"
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "<response>" in decoded:
            decoded = decoded.split("<response>")[-1]

        if "</response>" in decoded:
            decoded = decoded.split("</response>")[0]

        return decoded.strip()
