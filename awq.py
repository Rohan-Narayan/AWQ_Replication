from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
from find_salients import find_s_and_salient_weights
from quant import quantize
from perplexity import compute_perplexity
from scale import apply_awq_scaling


models = ["facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b"]
if __name__ == "__main__":
    for model_path in models:
        print("Working on " + model_path + "...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        enc = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False, trust_remote_code=True
                    )
        
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
                        model_path, config=config, trust_remote_code=True, **kwargs
                    )
        model.eval()
        q_config = {
            "zero_point": True,
            "q_group_size": 128, 
        }

        awq_results = find_s_and_salient_weights(model,
                        enc,
                        q_config=q_config)
        torch.save(awq_results, model_path + "_awq.pt")

    testset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    perplexities = []
    for model_path in models:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        enc = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False, trust_remote_code=True
                    )
        
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
                        model_path, config=config, trust_remote_code=True, **kwargs
                    )
        apply_awq_scaling(model, awq_results)
        quantize(model, w_bit=4, q_config=q_config)

        testenc = enc("\n\n".join(testset["text"]), return_tensors="pt")

        model.to('cuda')
        perplexity = compute_perplexity(model, testenc, 'cuda')
        perplexities.append(perplexity)
        print()
        print(perplexity.item())