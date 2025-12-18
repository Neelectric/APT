# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

sns.set_theme(style="darkgrid")
_executor = ProcessPoolExecutor(max_workers=1)

def _plot(steps, losses, norms, ems):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    
    ax1.plot(steps, losses, label='loss')
    ax1.plot(steps, norms, label='norm', alpha=0.7)
    ax1.legend()
    ax1.set_xlabel('step')
    
    ax2.plot(steps, ems, color='green')
    ax2.set_xlabel('step')
    ax2.set_ylabel('EM %')
    
    plt.tight_layout()
    plt.savefig('plots/current_run.png', dpi=100)
    plt.close()

def plot_async(steps, losses, norms, ems):
    _executor.submit(_plot, list(steps), list(losses), list(norms), list(ems))