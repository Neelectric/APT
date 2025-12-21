# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sns.set_theme(style="darkgrid", palette="muted")
_executor = ThreadPoolExecutor(max_workers=2)

def _plot(steps, losses_train, losses_eval, norms, accuracies, acc_1d, acc_2d, acc_3d, config, train_hparam_dict):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('APT Training Progress', fontsize=16, fontweight='bold', y=0.98)
    # Add config info to top right
    if config is not None:
        hparams = ["".join(key + str(val)) for key, val in train_hparam_dict.items()]
        hparam_string = ", ".join(hparams)
        # print(hparam_string)
        config_text = (f"n_layer={config.n_layer}  n_head={config.n_head} n_embd={config.n_embd}  mlp_exp={config.mlp_expansion_factor}\n"
                       f"{hparam_string}")
        fig.text(0.98, 0.98, config_text, transform=fig.transFigure,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    
    ax1.plot(steps, losses_train, label='Train', linewidth=2)
    ax1.plot(steps, losses_eval, label='Eval', linewidth=2, linestyle='--')
    ax1.legend(frameon=True)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    
    ax2.plot(steps, norms, color='coral', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norms')
    
    ax3.plot(steps, accuracies, color='seagreen', linewidth=2)
    ax3.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='Target')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('EM %')
    ax3.set_title('Exact Match Accuracy')
    ax3.set_ylim(bottom=0)
    ax3.set_yticks(np.arange(0, 110, step=10))
    ax3.legend(frameon=True)
    
    ax4.plot(steps, acc_1d, label='1-digit', linewidth=2)
    ax4.plot(steps, acc_2d, label='2-digit', linewidth=2)
    ax4.plot(steps, acc_3d, label='3-digit', linewidth=2)
    ax4.axhline(y=100, color='gold', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('EM %')
    ax4.set_title('Accuracy by Answer Length')
    ax4.set_ylim(bottom=0)
    ax4.legend(frameon=True)

    
    plt.tight_layout()
    plt.savefig('plots/current_run.png', dpi=120, facecolor='white', edgecolor='none')
    plt.close()

def plot_async(steps, losses_train, losses_eval, norms, accuracies, acc_1d, acc_2d, acc_3d, config, train_hparam_dict):
    _executor.submit(_plot, list(steps), list(losses_train), list(losses_eval), list(norms), list(accuracies), list(acc_1d), list(acc_2d), list(acc_3d), config, train_hparam_dict)