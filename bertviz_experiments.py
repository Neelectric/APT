"""This script is to help me understand how the heck bertviz works and what adaptations to APT are necessary to visualise attention with it"""
# System imports
import time
import os
import pickle

# External imports
import torch
from tqdm import tqdm
from bertviz import model_view
from transformers import utils, AutoModelForCausalLM, AutoTokenizer

# Local imports
from arithmetic_pretrained_transformer import APT, APTConfig, DataLoaderLite
from arithmetic_tokenizer import ArithmeticTokenizer

# Environment prep
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)
torch.set_printoptions(
    sci_mode=False, 
    threshold=10_000,
    edgeitems=3,
    )
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
print(f"using device {device}")
utils.logging.set_verbosity_error()  # Suppress standard warnings

# ------------------------------------------------------------------------------------------- #

# questions to pass to the model 
questions = [
    "<bos> 3 1 + 1 3 =", # 4 4 <eos>
    "<bos> 3 1 + 1 4 =", # 4 5 <eos>
    "<bos> 3 1 + 1 5 =", # 4 6 <eos>
    "<bos> 3 1 + 1 6 =", # 4 7 <eos>
]
# question = "13+4"
# prompt = f"Question: What is {question}? Answer: {question}="

apt = True
if apt:
    filename = 'apt_checkpoints/base/finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    vocab_path = 'tokenizer/sum_0-9_vocab.json'
    tokenizer = ArithmeticTokenizer(vocab_path)
    config = APTConfig(
        vocab_size=len(tokenizer._id_tokens),
        n_layer=1,
        n_head=3,
        n_embd=6,
        bias=True,
        pos_embd='learned',
        output_attentions = True,

        print_setup = True,
        print_ln_1 = True,
        print_attn = True,
        print_closed_res_streams = True,
        print_ln_2 = True,
        print_mlp = True,
        print_final = True
        )
    model.config = config
    for layer in model.transformer.h:
        layer.config = config
        layer.attn.config.output_attentions = True
else:
    # model_id = "allenai/OLMo-1B-0724-hf"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_id = "EleutherAI/pythia-1.4b-deduped"
    if "llama" not in model_id: 
        cache_dir = "./models/" + model_id
    else:
        cache_dir = None
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        cache_dir=cache_dir,
        device_map=device,
        output_attentions=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
        cache_dir=cache_dir,
        )

prompt = questions[3]
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # Tokenize input text
outputs = model(input_ids)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
# we copy attention and append it to 
print(len(attention))
print(attention[-1].shape)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # Convert input ids to token strings
model_view(
    attention, 
    tokens, 
    # include_layers=list(range(28,32))
    )  # Display model view