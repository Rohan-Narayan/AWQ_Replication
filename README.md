# AWQ Replication

## Goals
The goal of this repo is to replicate the results of the paper, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration". 
Specifically, this repo replicates the following table from the paper:\
![Screen Shot 2024-03-18 at 3 57 25 PM](https://github.com/Rohan-Narayan/AWQ_Replication/assets/59516165/e9b1e525-5585-467d-8bee-f812b4aa4987)
The main focus was on the bottom two rows (AWQ and s=2) as those were the main contributions of the paper.

## Repo Structure
Running awq.py with flags --test (to run on a small model to test end to end workflow) and --s_val (to fix a scale value) will trigger the AWQ process. The code in this repo was written
to mimic the described AWQ workflow in the AWQ paper. Heavy inspiration was drawn from the official AWQ repo: https://github.com/mit-han-lab/llm-awq. I used this repo as a reference to stay
on the right track and debug at times (for example, I initially did not implement weight clipping based on MSE). Furthermore, I used the INT3 kernel implementations for the OPT models provided
in their repo. 

## Results
|          | opt-1.3B | opt-2.7b | opt-6.7b | opt-13b
| -------- | ------- | ------- | ------- | ------- |
| s=2      |  17.72  |  15.33  |  11.85  |  TBD  |
| AWQ      |  16.31  |  13.56  |  11.37  |  TBD  |

As observed above, the results up to 6.7b parameters were replicated effectively. Curiously, certain configurations outperformed the paper's presented results, such as s=2 for opt-1.3b.
However, other observed results were marginally worse than the presented results. The goal was to replicate opt-30b as well, but I ran into GPU memory requirements. For opt-13b, a A100 GPU 
is required. However, the access is limited on Google Colab, so those results will be filled in once the device is available.\
The results for opt-1.3b to opt-6.7b were run on an L4 GPU on Google Colab. 


