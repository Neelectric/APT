# Let's train a Sum Pretrained Transformer
# System imports

# External imports
import torch
from tqdm import tqdm

# Local imports
from arithmetic_pretrained_transformer import APT, APTConfig, DataLoaderLite
from apt_tokenizer import APTTokenizer

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


# MODEL SETUP
model = APT(APTConfig())
model.to(device)
model.device = device
vocab_path = 'tokenizer/vocab.json'
tokenizer = APTTokenizer(vocab_path)
model.tokenizer = tokenizer

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in model: {pytorch_total_params:,}")



# HYPERPARAMETERS AND UTILITIES FOR TRAINING, EVAL DATASET PREP
batch_size = 2048 #1024 works?
num_tokens_per_sample = 10
data_location = 'datasets/sum_dataset.json'
train_loader = DataLoaderLite(B=batch_size, T=num_tokens_per_sample, data_location='datasets/sum_dataset.json')
learning_rate = 208e-4
trainset_size = train_loader.trainset_size
epochs = 3000
max_steps = epochs * (trainset_size) // batch_size
eval_intervals = max_steps // 16
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # easy gains: decrease weights for different language tokens!
print(f"max_steps: {max_steps}, eval_intervals: {eval_intervals}, learning_rate: {learning_rate}")



eval_prompts = []
eval_ground_truths = []
for elt in train_loader.eval_raw:
    eval_prompts.append(elt.split("=")[0] + "=")
    eval_ground_truths.append(elt)

def eval_naive(print_incorrect=False):
    # model.eval()
    num_correct = 0
    for prompt, ground_truth in tqdm(zip(eval_prompts, eval_ground_truths), dynamic_ncols=True, disable=True):
        prediction = model.answer(prompt)
        if prediction == ground_truth:
            num_correct += 1
        elif print_incorrect:
            print(ground_truth, prediction)
    EM_score = num_correct/len(eval_prompts)
    if print_incorrect:
        print(f"Out of {len(eval_prompts)} questions, APT got {num_correct} correct.")
    return EM_score


# TRAINING BEGINS
losses = []
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
    optimizer.step() # this actually updates the params
    if step % eval_intervals == 0:
        em_score_reading = eval_naive() * 100
        # print(x[0])
        # print(y[0])
        # print(logits[0][6:8])
        tqdm.write(f"step {step}, train loss: {loss.item():.4f}, eval accuracy (EM): {em_score_reading:.2f}%") #we use .item() because this is a tensor with a single element that lives on .device. .item() sends it to cpu
        accuracies.append(em_score_reading)
        accuracy_steps.append(step)
        losses.append(loss.item())
        writer.add_scalar("EM Score", em_score_reading, step)
graph_inputs = [x,y]
writer.add_graph(model, graph_inputs)
writer.flush()
writer.close()

    
final_em_score_reading = eval_naive(print_incorrect=True) * 100
print(f"step {step}, train loss: {loss.item():.4f}, eval accuracy (EM): {final_em_score_reading:.2f}%") 


# # PLOT
# fig, ax1 = plt.subplots(figsize=(10, 5))
    
# # Plot losses on the left y-axis
# ax1.plot(accuracy_steps, losses, label='Losses', color='blue')
# ax1.set_xlabel('Steps')
# ax1.set_ylabel('Losses', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Create a secondary y-axis for accuracies
# ax2 = ax1.twinx()
# ax2.plot(accuracy_steps, accuracies, label='Accuracies', color='green', linestyle='-', marker='o')
# ax2.set_ylabel('Accuracies (%)', color='green')
# ax2.tick_params(axis='y', labelcolor='green')

# # Set y-ticks frequency for the right y-axis
# ax2.yaxis.set_major_locator(MultipleLocator(5))

# # Add grid
# # ax1.grid(True)
# ax2.grid(True)

# # Combine legends
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper left')


# # Add title
# plt.title('Training Losses and Accuracies')

# # Display the plot
# plt.show()