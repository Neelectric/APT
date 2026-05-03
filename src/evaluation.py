# Let's evaluate an Arithmetic Pretrained Transformer

# System imports
import time
import os
import pickle
from collections import Counter
import json, random
from collections import defaultdict


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

def make_balanced_eval_lists(data_location, n_per_bucket=25, seed=42):
    """
    Returns (train_raw, eval_prompts, eval_ground_truths).
    Stratifies by answer digit count so every bucket has equal eval samples.
    Caps each bucket at half its size so we never starve training of a class.
    """
    with open(data_location, 'r') as f:
        all_data = json.load(f)

    buckets = defaultdict(list)
    for elt in all_data:
        buckets[len(elt.split('=')[1])].append(elt)

    rng = random.Random(seed)
    eval_raw, train_raw = [], []
    for n_digits in sorted(buckets):
        items = buckets[n_digits]
        rng.shuffle(items)
        take = min(n_per_bucket, len(items) // 2)
        eval_raw.extend(items[:take])
        train_raw.extend(items[take:])
        print(f"  {n_digits}-digit: {take} eval / {len(items) - take} train (of {len(items)} total)")

    rng.shuffle(train_raw)
    eval_prompts       = [e.split("=")[0] + "=" for e in eval_raw]
    eval_ground_truths = list(eval_raw)
    return train_raw, eval_prompts, eval_ground_truths



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
    # print(Counter(len(gt.split('=')[1]) for gt in eval_ground_truths))

    
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
    
    acc_by_digits = {k: 100 * correct[k] / total[k] if total[k] > 0 else None for k in [1, 2, 3]}
    return num_correct / len(eval_prompts), acc_by_digits

@torch.inference_mode()
def eval_matched_padding(
    model,
    eval_prompts,
    eval_ground_truths,
    max_answer_len=3,
    print_incorrect=False,
):
    """
    Drop-in replacement for eval_parallel_claude that matches training-time
    positional alignment by grouping prompts by length instead of left-padding.
    """
    model.eval()

    # Tokenize per-prompt (no padding) and bucket by length
    groups = defaultdict(list)  # prompt_len -> [(orig_idx, ids_list), ...]
    for i, prompt in enumerate(eval_prompts):
        ids = model.tokenizer(prompt)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids[0], list):  # tokenizer wrapped a single example in a batch
            ids = ids[0]
        groups[len(ids)].append((i, ids))

    predictions = [None] * len(eval_prompts)

    for prompt_len, items in groups.items():
        idxs   = [it[0] for it in items]
        tokens = torch.tensor([it[1] for it in items], dtype=torch.long, device=model.device)

        # Greedy-generate max_answer_len tokens; we only consume what each example needs
        for _ in range(max_answer_len):
            logits, _ = model(tokens)
            next_tok  = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens    = torch.cat([tokens, next_tok], dim=1)

        for row, idx in enumerate(idxs):
            gt        = eval_ground_truths[idx]
            n_answer  = len(gt.split('=')[1])
            pred_ids  = tokens[row, : prompt_len + n_answer].tolist()
            pred      = "".join(model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            predictions[idx] = pred

    correct = {1: 0, 2: 0, 3: 0}
    total   = {1: 0, 2: 0, 3: 0}
    num_correct = 0
    for gt, pred in zip(eval_ground_truths, predictions):
        n = len(gt.split('=')[1])
        total[n] += 1
        if pred == gt:
            num_correct += 1
            correct[n]  += 1
        elif print_incorrect:
            print(gt, pred)

    acc_by_digits = {k: 100 * correct[k] / total[k] if total[k] > 0 else None for k in [1, 2, 3]}
    return num_correct / len(eval_prompts), acc_by_digits