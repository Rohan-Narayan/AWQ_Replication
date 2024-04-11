# AWQ Replication

## Goals
The goal of this repo is to replicate the results of the paper, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration". 
Specifically, this repo replicates the following table from the paper:\
<img width="555" alt="Screen Shot 2024-03-18 at 3 57 25 PM" src="https://github.com/Rohan-Narayan/AWQ_Replication/assets/59516165/2a639b35-a206-46ee-8d47-b68c4169f792"> \
The main focus was on the bottom two rows (AWQ and s=2) as those were the main contributions of the paper.

## Repo Structure
Running awq.py with flags --test (to run on a small model to test end to end workflow) and --s_val (to fix a scale value) will trigger the AWQ process. The code in this repo was written
to mimic the described AWQ workflow in the AWQ paper. Heavy inspiration was drawn from the official AWQ repo: https://github.com/mit-han-lab/llm-awq. I used this repo as a reference to stay
on the right track and debug at times (for example, I initially did not implement weight clipping based on MSE). Furthermore, I used the INT3 kernel implementations for the OPT models provided in their repo. 

## Results
|          | opt-1.3b| opt-2.7b| opt-6.7b| opt-13b |
| -------- | ------- | ------- | ------- | ------- |
| s=2      |  17.72  |  15.33  |  11.85  |  11.17  |
| AWQ      |  16.31  |  13.56  |  11.37  |  10.56  |


The above table reports the observed perplexities after applying AWQ on the specified models (either with a fixed scale s=2 or grid searching for the scales).\
As observed above, the results up to 6.7b parameters were replicated effectively. Curiously, certain configurations outperformed the paper's presented results, such as s=2 for opt-1.3b.
However, other observed results were marginally worse than the presented results. For the most part, the AWQ results were very similar, with slight deviations to the s=2 results. The goal was to replicate opt-30b as well, but I ran into GPU memory requirements. For opt-13b, a A100 GPU 
is required. The results for opt-1.3b to opt-6.7b were run on an L4 GPU on Google Colab. The link to the Google Colab notebook is below. \
https://colab.research.google.com/drive/12Gwv-0alJc_rCNdIOETjZApCakYSkWOk?usp=sharing \
The results can be seen under the header "Run Files".

## Next Steps
In the original paper, the authors do claim that opt-30b can "AWQ-ized" on a single A100 GPU. However, I was unable to replicate this. It's possible they used an 80GB RAM A100, while the Colab A100 only has 40GB. Regardless, it would be interesting to explore memory savings to achieve an AWQ form of opt-30b.

