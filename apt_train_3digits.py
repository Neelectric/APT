# Let's train a Sum Pretrained Transformer
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


# ------------------------------------------TRAINING-----------------------------------------------------------
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
print(f"using device {device}")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  # or 'medium'


with_bos = False
vocab_path = 'tokenizer_variations/sum_0-9+special_vocab.json'
num_tokens_per_sample = 11
data_location = 'datasets/no_bos_no_eos/499by499.json'

# MODEL SETUP
tokenizer = ArithmeticTokenizer(vocab_path, max_length=num_tokens_per_sample, padding="max_length")
config = APTConfig(vocab_size=len(tokenizer._id_tokens),
                   block_size=num_tokens_per_sample,
                   n_layer=6,
                   n_head=1,
                   n_embd=3,
                   mlp_expansion_factor=32,
                   bias=True,
                   pos_embd='learned',
                   )
print(f"VOCAB SIZE IS {config.vocab_size}")
model = APT(config)
# model = torch.load("apt_checkpoints/base/6_1_3_32_600efinalized_model_bos_is_False.sav", 
                #    map_location=device, weights_only=False)

model.to(device)
model.device = device
model.tokenizer = tokenizer
# 6,1,4,16 goes 1.2343 loss on 100 epochs.
#8,3,3,32 goes 1.4243 loss on 50 epochs
#8,1,3,32 goes 1.4112 loss on 50 epochs
#6,1,3,32 goes to 1.2769 on 600 epochs

# HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
batch_size = 4096 #131072 #65536 #32768 #16384 #8192 #4096 #2048 #1024 works?

# train_loader = DataLoaderLite(
#     B=batch_size, 
#     T=num_tokens_per_sample, 
#     data_location=data_location, 
#     tokenizer=tokenizer,
#     eval_percentage=0.01
#     )
train_loader = DataLoaderPyTorch(
    B=batch_size, 
    T=num_tokens_per_sample, 
    data_location=data_location, 
    tokenizer=tokenizer,
    eval_percentage=0.01,
    num_workers=0,
)

peak_learning_rate = 0.025 #0.04
min_learning_rate = 0.005
weight_decay = 0.02
max_grad_norm = 0.75
epochs = int(1000 * 1)
trainset_size = train_loader.trainset_size
max_steps = epochs * (trainset_size) // batch_size

train_hparam_dict = {
    "peak_learning_rate": peak_learning_rate,
    # "min_learning_rate": min_learning_rate,
    "weight_decay": weight_decay,
    "max_grad_norm": max_grad_norm,
    "batch_size": batch_size,
    "trainset_size": trainset_size,
    "epochs": epochs,
    "max_steps": max_steps,
    
}

eval_intervals = max_steps // 25
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learning_rate, weight_decay=weight_decay) # easy gains: decrease weights for different language tokens!
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_learning_rate) # claude considers this important


pytorch_total_params = sum(p.numel() for p in model.parameters(recurse=True))
print(f"Total number of parameters in model: {pytorch_total_params:,}")
print(f"max_steps: {max_steps}, eval_intervals: {eval_intervals}, learning_rate: {peak_learning_rate}")


eval_prompts = []
eval_ground_truths = []
for elt in train_loader.eval_raw:
    eval_prompts.append(elt.split("=")[0] + "=")
    eval_ground_truths.append(elt)

def eval_naive(print_incorrect=False, max_length_eval_prompt=8, max_length=12):
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

def eval_parallel(print_incorrect=False, max_length_eval_prompt=8, max_length=12):
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
def eval_parallel_claude(print_incorrect=False, max_length_eval_prompt=8, max_length=12):
    model.eval()
    tokens = model.tokenizer(eval_prompts, return_tensors="pt", padding='max_length', 
                                max_length=max_length_eval_prompt, padding_side="left")["input_ids"].to(device)
    
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

# TRAINING BEGINS
losses_train = []
losses_eval = []
accuracies = []
norms = []
eval_step_numbers = []
acc_1d_list, acc_2d_list, acc_3d_list = [], [], []


for step in tqdm(range(max_steps), dynamic_ncols=True):
    model.train()
    x, y = train_loader.next_batch_train()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    # y[:,0:5] = -100
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    # writer.add_scalar("Loss/train", loss, step)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step() # this actually updates the params
    scheduler.step()
    if (step !=0) & (step % eval_intervals == 0):
    # if False:
        with torch.no_grad():
            model.eval()
            # tqdm.write(f"step {step} | loss_train: {loss.item():.4f}")
            
            # em_score_reading = eval_naive() * 100
            em_score_reading_parallel = 99999
            x_eval, y_eval = train_loader.next_batch_eval()
            x_eval, y_eval = x_eval.to(device), y_eval.to(device)
            logits_eval, loss_eval = model(x_eval, y_eval)
            # writer.add_scalar("Loss/eval", loss_eval.item(), step)
            em_score, acc_by_digits = eval_parallel_claude()
            em_score_reading_parallel = em_score * 100

            acc_1d_list.append(acc_by_digits[1])
            acc_2d_list.append(acc_by_digits[2])
            acc_3d_list.append(acc_by_digits[3])

            # em_score_reading_parallel = eval_naive() * 100
            tqdm.write(f"step {step} | loss_train: {loss.item():.4f} | loss_eval: {loss_eval.item():.4f} | norm: {norm:.3f}| EM (parallel): {em_score_reading_parallel:.2f}%") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu
            
            #now send plotting code
            eval_step_numbers.append(step)
            losses_train.append(loss.item())
            losses_eval.append(loss_eval.item())
            norms.append(norm.item())
            accuracies.append(em_score_reading_parallel)
            
            plot_async(
                steps=eval_step_numbers, 
                losses_train=losses_train, 
                losses_eval=losses_eval, 
                norms=norms, 
                accuracies=accuracies, 
                acc_1d=acc_1d_list, 
                acc_2d=acc_2d_list, 
                acc_3d=acc_3d_list, 
                config=config,
                train_hparam_dict=train_hparam_dict,
                )
            
            # writer.add_scalar("EM Score", em_score_reading_parallel, step)
# graph_inputs = [x,y]
# writer.add_graph(model, graph_inputs)
# writer.flush()
# writer.close()
    
# final_em_score_reading = eval_naive(print_incorrect=True) * 100
# print(f"step {step}, train loss: {loss.item():.4f}, eval accuracy (EM): {final_em_score_reading:.2f}%") 

save = True
if save:
    filename = 'apt_checkpoints/base/finalized_model_bos_is_' + str(with_bos) + '.pt'
    # before pickle dump we need to undo the weakref lol, but for torch.save this messes things up!
    # model.convert_weakrefs_to_strongrefs()
    torch.save(model.state_dict(), filename)  # save only weights
    print(f"Saved APT file as pickle dump under {filename}")
else:
    print("Save is false! We have not saved this model!")