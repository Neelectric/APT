# Let's evaluate an Arithmetic Pretrained Transformer

# System imports
import time
import os
import pickle

# External imports
import torch
from tqdm import tqdm

# Local imports
from src.arithmetic_pretrained_transformer import APT, APTConfig, DataLoaderLite, DataLoaderPyTorch
from src.arithmetic_tokenizer import ArithmeticTokenizer
from src.async_realtime_plots import plot_async

# Environment prep
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)
torch.set_printoptions(sci_mode=False)
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()




def eval_naive(
    model: APT,
    eval_prompts: list,
    eval_ground_truths: list,
    print_incorrect=False, 
    max_length_eval_prompt=8, 
    max_length=12
    ):
    num_correct = 0
    for prompt, ground_truth in tqdm(zip(eval_prompts, eval_ground_truths), dynamic_ncols=True, disable=False, total=len(eval_prompts)):
        prediction = model.answer(prompt, max_length_eval_prompt=max_length_eval_prompt, max_length=max_length)
        if prediction == ground_truth:
            num_correct += 1
        elif print_incorrect:
            print(ground_truth, prediction)
    EM_score = num_correct/len(eval_prompts)
    if print_incorrect:
        print(f"Out of {len(eval_prompts)} questions, APT got {num_correct} correct.")
    return EM_score


def eval_parallel(
    model: APT,
    eval_prompts: list,
    eval_ground_truths: list,
    print_incorrect=False, 
    max_length_eval_prompt=8, 
    max_length=12
    ):
    num_correct = 0
    predictions = model.answer(eval_prompts, max_length_eval_prompt=max_length_eval_prompt, max_length=max_length)
    for prediction, ground_truth in tqdm(zip(predictions, eval_ground_truths), dynamic_ncols=True, disable=True):
        if prediction == ground_truth:
            num_correct += 1
        elif print_incorrect:
            print(ground_truth, prediction)
    EM_score = num_correct/len(eval_prompts)
    if print_incorrect:
        print(f"Out of {len(eval_prompts)} questions, APT got {num_correct} correct.")
    return EM_score

@torch.inference_mode()
def eval_parallel_claude(
    model: APT,
    eval_prompts: list,
    eval_ground_truths: list,
    print_incorrect=False, 
    max_length_eval_prompt=8, 
    max_length=12
    ):
    model.eval()
    tokens = model.tokenizer(
        eval_prompts, 
        return_tensors="pt", 
        padding='max_length', 
        max_length=max_length_eval_prompt, 
        padding_side="left"
        )["input_ids"].to(model.device)
    
    for _ in range(max_length - max_length_eval_prompt):
        logits, _ = model(tokens)
        tokens = torch.cat([tokens, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    
    # Score by digit count
    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}
    num_correct = 0
    
    for gt, ids in zip(eval_ground_truths, tokens):
        answer = gt.split('=')[1]
        n_digits = len(answer)
        total[n_digits] += 1
        
        pred = "".join(model.tokenizer.batch_decode(ids[:-1].tolist(), skip_special_tokens=True))
        if pred == gt:
            num_correct += 1
            correct[n_digits] += 1
        elif print_incorrect:
            print(gt, pred)
    
    if print_incorrect:
        print(f"Out of {len(eval_prompts)} questions, APT got {num_correct} correct.")
    
    acc_by_digits = {k: 100 * correct[k] / total[k] if total[k] > 0 else 0 for k in [1, 2, 3]}
    return num_correct / len(eval_prompts), acc_by_digits