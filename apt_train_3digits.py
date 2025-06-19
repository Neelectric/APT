# Let's train a Sum Pretrained Transformer
# System imports
import time
import os
import pickle

# External imports
import torch
from tqdm import tqdm

# Local imports
from arithmetic_pretrained_transformer import APT, APTConfig, DataLoaderLite
from arithmetic_tokenizer import ArithmeticTokenizer

# Environment prep
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)
torch.set_printoptions(sci_mode=False)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# ------------------------------------------TRAINING-----------------------------------------------------------
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
print(f"using device {device}")

with_bos = False
vocab_path = 'tokenizer/sum_0-9+special_vocab.json'
num_tokens_per_sample = 11
data_location = 'datasets/no_bos_no_eos/499by499.json'

# MODEL SETUP
tokenizer = ArithmeticTokenizer(vocab_path, max_length=num_tokens_per_sample, padding="max_length")
config = APTConfig(vocab_size=len(tokenizer._id_tokens),
                   block_size=num_tokens_per_sample,
                   n_layer=1,
                   n_head=3,
                   n_embd=6,
                   bias=True,
                   pos_embd='learned',
                   )
print(f"VOCAB SIZE IS {config.vocab_size}")
model = APT(config)
model.to(device)
model.device = device
model.tokenizer = tokenizer


# HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
batch_size = 2048 #1024 works?
train_loader = DataLoaderLite(
    B=batch_size, 
    T=num_tokens_per_sample, 
    data_location=data_location, 
    tokenizer=tokenizer,
    eval_percentage=0.01
    )
learning_rate = 0.042
trainset_size = train_loader.trainset_size
epochs = int(150 * 1)
max_steps = epochs * (trainset_size) // batch_size
eval_intervals = max_steps // 6
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # easy gains: decrease weights for different language tokens!

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in model: {pytorch_total_params:,}")
print(f"max_steps: {max_steps}, eval_intervals: {eval_intervals}, learning_rate: {learning_rate}")


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

# TRAINING BEGINS
losses_train = []
losses_eval = []
accuracies = []
accuracy_steps = []

for step in tqdm(range(max_steps), dynamic_ncols=True):
    model.train()
    x, y = train_loader.next_batch_train()
    x, y = x.to(device), y.to(device)
    # y[:,0:5] = -100
    optimizer.zero_grad() # always need to start with 0 gradient
    logits, loss = model(x, y)
    writer.add_scalar("Loss/train", loss, step)
    loss.backward() # this adds to gradients! which is why we need to zero_grad
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    # norm = 1
    optimizer.step() # this actually updates the params
    if step % eval_intervals == 0:
    # if False:
        with torch.no_grad():
            model.eval()
            # tqdm.write(f"step {step} | loss_train: {loss.item():.4f}")
            
            # em_score_reading = eval_naive() * 100
            print("overwritting em score parallel")
            em_score_reading_parallel = 99999
            x_eval, y_eval = train_loader.next_batch_eval()
            x_eval, y_eval = x_eval.to(device), y_eval.to(device)
            logits_eval, loss_eval = model(x_eval, y_eval)
            writer.add_scalar("Loss/eval", loss_eval, step)
            # em_score_reading_parallel = eval_parallel() * 100
            em_score_reading_parallel = eval_naive() * 100
            tqdm.write(f"step {step} | loss_train: {loss.item():.4f} | loss_eval: {loss_eval.item():.4f} | norm: {norm:.3f}| EM (parallel): {em_score_reading_parallel:.2f}%") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu
            # accuracies.append(em_score_reading_parallel)
            # accuracy_steps.append(step)
            # losses_train.append(loss.item())
            # losses_eval.append(loss_eval.item())
            # writer.add_scalar("EM Score", em_score_reading_parallel, step)
graph_inputs = [x,y]
writer.add_graph(model, graph_inputs)
writer.flush()
writer.close()
    
# final_em_score_reading = eval_naive(print_incorrect=True) * 100
# print(f"step {step}, train loss: {loss.item():.4f}, eval accuracy (EM): {final_em_score_reading:.2f}%") 

save = True
if save:
    filename = 'apt_checkpoints/base/finalized_model_bos_is_' + str(with_bos) + '.sav'
    # before pickle dump we need to undo the weakref lol
    model.convert_weakrefs_to_strongrefs()
    pickle.dump(model, open(filename, 'wb'))
    print(f"Saved APT file as pickle dump under {filename}")
else:
    print("Save is false! We have not saved this model!")