from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
from find_salients import find_s_and_salient_weights
from quant import quantize
from perplexity import compute_perplexity
from scale import apply_awq_scaling
import gc


models = ["opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"]
q_config = {
            "zero_point": True,
            "q_group_size": 128, 
        }
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
if __name__ == "__main__":
    for model_name in models:
        model_path = "facebook/" + model_name
        print("Working on " + model_name + "...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        enc = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False, trust_remote_code=True
                    )
        
        model = AutoModelForCausalLM.from_pretrained(
                        model_path, config=config, trust_remote_code=True, **kwargs
                    )
        model.eval()

        awq_results = find_s_and_salient_weights(model,
                        enc,
                        q_config=q_config)
        
        torch.save(awq_results, model_name + "_awq.pt")

    testset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    perplexities = {}
    for model_name in models:
        gc.collect()
        torch.cuda.empty_cache()
        results_path = model_name + "_awq.pt"
        awq_results = torch.load(results_path, map_location="cpu")
        model_path = "facebook/" + model_name
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
        perplexities[model_name] = perplexity.item()
        # print()
        # print(perplexity.item())

    print("Full AWQ Perplexities")
    for k,v in perplexities:
        print("Perplexity for k:", v)
