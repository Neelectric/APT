"""Implements a function-complete transformer architecture for addition"""
# System imports
import math
import json
from dataclasses import dataclass
import random
import weakref
import pickle

# External imports
import torch
import torch.backends
import torch.nn as nn
from torch.nn import functional as F

# Local imports

# Environment prep
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)
torch.set_printoptions(
    sci_mode=False, 
    threshold=10_000,
    edgeitems=3,
    )
random.seed(10)
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
print(f"using device {device}")

# -------------------------------------------- #

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        # key, query and value projections for all heads, but batched!
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #technically a mask not a bias but follows HF/OAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate q, k and v for all heads in batch and move head forward to be the batch
        # nh is num_heads, hs is head_size, C (number of channels) is nh * ns
        # so for GPT-2 (124M), n_head=12, hs=64, nh*hs=C=768 channels in transformer
        # pickle.dump(x, open("random_pickles/x.sav", 'wb'))
        qkv = self.c_attn(x)
        queries, keys, values = qkv.split(self.n_embd, dim=2)
        queries = queries.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        keys = keys.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        values = values.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all queries and keys)
        attn_weights = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ values # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        attn_output = attn_output.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        attn_output = self.c_proj(attn_output)

        # at the end of the day, what we want this output to be is:
        # a list of num_layers tensors, with shape (batch_size(must be 1), num_heads, sequence_length, sequence_length)
        # when visualising we may want to output attentions too
        if not self.config.output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, config.mlp_expansion_factor * config.n_embd, bias=config.bias) # by default up-projection by 4x expansion factor
        self.gelu = nn.GELU(approximate='tanh') #gpt2 used tanh approximation, dan hendrycks suggested in github comment, nowadays irrelevant
        self.c_proj = nn.Linear(config.mlp_expansion_factor * config.n_embd, config.n_embd, bias=config.bias) # by default down-projection from 4x expansion factor

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config, apt):
        super().__init__()
        self.config = config
        self.apt = weakref.ref(apt)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        # self.ln_1 = nn.RMS
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, hidden_states):
        # open residual stream and do input layernorm
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        if self.config.print_ln_1:
            print(f"hidden_states after ln_1: \n{self.apt.apt_print(hidden_states)}\n")

        # self-attn and close residual stream
        attn_output, attn_weights = self.attn(hidden_states)
        if self.config.print_attn:
            print(f"attn_output: \n{self.apt.apt_print(attn_output)}\n")
        hidden_states = residual + attn_output
        if self.config.print_closed_res_streams:
            print(f"hidden_states closing res stream: \n{self.apt.apt_print(hidden_states)}\n")

        # reopen residual stream and do second layernorm
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        if self.config.print_ln_2:
            print(f"hidden_states after ln_2: \n{self.apt.apt_print(hidden_states)}\n")

        # fully connected and close residual stream
        hidden_states = self.mlp(hidden_states)
        if self.config.print_mlp:
            print(f"hidden_states after mlp: \n{self.apt.apt_print(hidden_states)}\n")
        hidden_states = residual + hidden_states
        if self.config.print_closed_res_streams:
            print(f"hidden_states closing res streamv2: \n{self.apt.apt_print(hidden_states)}\n")

        if not self.config.output_attentions:
            attn_weights = None
        return hidden_states, attn_weights
        
@dataclass
class APTConfig:
    block_size: int = 10
    vocab_size: int = 17
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 4 #512 seems to work well. 256 can also generalize w bsz=< 1024. 128 or 64 works with 256 bsz and lr 8e-4. for 32 we need lr 8e-3. for 8 or 16 we need lr 1e-2. actually for 8, 12e-3 seems ideal. for 4, 20e-3 or even 208e-4 could work
    mlp_expansion_factor: int = 4,
    bias: bool = True
    pos_embd: str = 'learned'
    output_attentions: bool = False

    print_setup: bool = False
    print_ln_1: bool = False
    print_attn: bool = False
    print_closed_res_streams: bool = False
    print_ln_2: bool = False
    print_mlp: bool = False
    print_final: bool = False
    

class APT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        print("Add different options for learned vs rotational vs alibi positional encodings!!!")
        print("Swap layernorm to RMSNorm!")

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd), # weight token embeddings
            # if config.pos_embd == 'learned':
            wpe = nn.Embedding(config.block_size, config.n_embd), # weight positional embeddings
            h = nn.ModuleList([Block(config, self) for _ in range(config.n_layer)]), # layers
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias), # final layernorm, introduced by GPT2 paper
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias) # final classification head
        self.tokenizer = None # keeping this empty for now but in apt_print we need it for tokenizing and detokenizing
        self.current_input_ids = None # same as above!
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)

    def forward(self, input_ids, targets=None):
        self.current_input_ids = input_ids
        # input_ids is of shape (B, T), but sometimes just a single sequence, in which case we unsqueeze to make it a batch of size 1:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        all_self_attns = () if self.config.output_attentions else None

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (B, T, n_embd)
        hidden_states = tok_emb + pos_emb
        
        if self.config.print_setup:
            torch.set_printoptions(
                precision=3,
                sci_mode=False, 
                threshold=10_000,
                edgeitems=3,
                linewidth=200,
            )
            print(f"positional embeddings: \n{self.apt_print(pos_emb)}\n")
            print(f"token embeddings: \n{self.apt_print(tok_emb)}\n")
            print(f"pos+tok: \n{self.apt_print(hidden_states)}\n")

        # forward the blocks of the transformer
        for block in self.transformer.h:
            hidden_states, attn_weights = block(hidden_states)
            if self.config.output_attentions:
                all_self_attns += (attn_weights,)

        # forward the final layernorm and the classifier
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states) # (B, T, vocab_size)
        if self.config.print_final:
            print(f"hidden_states after ln_f: \n{hidden_states[:,-1,:]}\n")
            print(f"logits: \n{logits[:,-1,:]}\n")

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # function does not like multi-dim tensors, so we flatten them to be BxT for all inputs and all targets
        if self.config.output_attentions:
            return logits, loss, all_self_attns
        
        return logits, loss

    def generate(self, input_ids, max_length=10):
        eos_token_id = self.tokenizer.eos_token_id
        while True:
            if (input_ids.shape[1] >= max_length) or (input_ids[0][-1] == eos_token_id):
                input_ids = input_ids.tolist()
                return input_ids
            logits, loss = self(input_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(50, self.config.vocab_size), dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            input_ids = torch.cat((input_ids, xcol), dim=1)
    
    def answer(self, prompt, max_length_eval_prompt, max_length=10):
        # tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        tokens = self.tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=max_length_eval_prompt, padding_side="left")["input_ids"]
        input_ids = tokens.to(self.device)
        output_ids = self.generate(input_ids, max_length=max_length)[0]
        output_ids_no_eos = output_ids[0:-1]
        decoded = self.tokenizer.batch_decode(output_ids_no_eos, skip_special_tokens=True)
        output = "".join(decoded)
        return output
    
    def apt_print(self, hidden_state):
        current_sequence = []
        for token in self.current_input_ids[0]:
            current_sequence.append(self.tokenizer.decode(token))
        output_string = ''
        for token, vector in zip(current_sequence, hidden_state.squeeze()):
            output_string += f"'{token}' - {vector.detach()}\n"
        return output_string
    
    def convert_weakrefs_to_strongrefs(self):
        for layer in self.transformer['h']:
            layer.apt = self

class DataLoaderLite:
    def __init__(self, B, T, data_location, tokenizer, shuffle=True, eval_percentage=0.1):
        self.B = B
        self.T = T
        self.max_length = tokenizer.max_length
        # vocab_path = 'tokenizer/vocab.json'
        # tokenizer = APTTokenizer(vocab_path)
        with open(data_location, 'r') as f:
            text = json.load(f)
        if shuffle:
            random.shuffle(text)
        num_eval = int(eval_percentage * len(text))
        eval_raw, train_raw = text[0:num_eval], text[num_eval+1:]
        self.trainset_size = len(train_raw)
        print(f"we have self.trainset_size {self.trainset_size}, and num_eval {num_eval}")
        # train = " ".join(train_raw)
        # eval = " ".join(eval_raw)
        self.tokens_train = tokenizer(train_raw, return_tensors="pt", padding='max_length', max_length=self.max_length, padding_side="left")["input_ids"]
        self.eval_raw = eval_raw
        self.tokens_eval = tokenizer(eval_raw, return_tensors="pt", padding='max_length', max_length=self.max_length, padding_side="left")["input_ids"]
        print(f"loaded {self.tokens_train.shape[0] * self.tokens_train.shape[1]} tokens")
        print(f"1 epoch = {len(self.tokens_train) // (B * T)} batches")
        self.current_position_train = 0
        self.current_position_eval = 0
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def next_batch_train(self):
        B, T = self.B, self.T
        batch = self.tokens_train[self.current_position_train : self.current_position_train + B]
        self.current_position_train += B
        x = batch
        eos_tok = self.eos_token_id
        y = torch.cat([
            batch[:, 1:], # we remove first token of each prompt
            torch.full((B, 1), eos_tok, dtype=batch.dtype, device=batch.device)
        ], dim=1)
        # if loading next batch would be out of bounds, reset
        if self.current_position_train + (B + 1) > len(self.tokens_train):
            self.current_position_train = 0
        y[x == self.pad_token_id] = -100 # pytorch cross_entropy convention is to ignore all instances with label -100, we use this for pad tokens so they don't muddle gradients
        return x,y
    
    def next_batch_eval(self):
        B, T = self.B, self.T
        batch = self.tokens_eval[self.current_position_eval : self.current_position_eval + B]
        self.current_position_eval += B
        x = batch
        eos_tok = self.eos_token_id
        y = torch.cat([
            batch[:, 1:], # we remove first token of each prompt
            torch.full((B, 1), eos_tok, dtype=batch.dtype, device=batch.device)
        ], dim=1)
        # if loading next batch would be out of bounds, reset
        if self.current_position_eval + (B + 1) > len(self.tokens_eval):
            self.current_position_eval = 0
        return x,y
    
    # def next_batch_eval(self):
    #     B, T = self.B, self.T
    #     B = math.floor(250/T)
    #     buf = self.tokens_eval[self.current_position_eval : self.current_position_eval + B*T + 1]
    #     x = (buf[:-1]).view(B, T) # inputs
    #     y = (buf[1:]).view(B, T) # targets
    #     # advance the current position in tensor
    #     self.current_position_eval += B * T
    #     # if loading next batch would be out of bounds, reset
    #     if self.current_position_eval + (B * T + 1) > len(self.tokens_eval):
    #         self.current_position_eval = 0
    #     return x,y
    



### Lets test!
# from arithmetic_tokenizer import ArithmeticTokenizer
# with_bos = False
# vocab_path = 'tokenizer/sum_0-9+special_vocab.json'
# num_tokens_per_sample = 11
# data_location = 'datasets/no_bos_no_eos/499by499.json'

# # MODEL SETUP
# tokenizer = ArithmeticTokenizer(vocab_path, max_length=num_tokens_per_sample, padding="max_length")

# # HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
# batch_size = 2 #1024 works?
# train_loader = DataLoaderLite(
#     B=batch_size, 
#     T=num_tokens_per_sample, 
#     data_location=data_location, 
#     tokenizer=tokenizer
#     )

# x, y = train_loader.next_batch_train()
# print(x)
# print(y)