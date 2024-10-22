import openai
model = "D:\\hf_models\\Hermes 2 Theta\\Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q6_K.gguf"

models = [
    "D:\\hf_models\\Hermes 2 Theta\\Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q6_K.gguf",
    "D:\\hf_models\\Hermes 2 Theta\\Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q5_K_M.gguf",
    "D:\\hf_models\\Hermes 2 Theta\\Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q6_K.gguf",
    "D:\\hf_models\\Llama_3_2_1B_Instruct\\model.safetensors",
    "D:\\hf_models\\Llama_3_2_1B_Instruct\\model.safetensors",
    "D:\\hf_models\\Hermes_3_1\\Hermes-3-Llama-3.1-8B.Q6_K.gguf",
    "D:\\hf_models\\Hermes_3_1\\Hermes-3-Llama-3.1-8B.Q8_0.gguf"
]

client = openai.OpenAI(
    base_url="http://192.168.1.4:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)