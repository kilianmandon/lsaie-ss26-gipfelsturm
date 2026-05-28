from data_extraction_from_logs import read_data
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def job_name(model_size):
    if model_size == '125m':
        return 'gipfel-throughput-125m-4096seq-16mbs-256gbs-1n'
    elif model_size == '760m':
        return 'gipfel-throughput-760m-4096seq-4mbs-256gbs-1n'
    elif model_size == '3B':
        return 'gipfel-throughput-3b-4096seq-4mbs-256gbs-1n'
    elif model_size == '8B':
        return 'gipfel-throughput-8b-4096seq-2mbs-256gbs-1n'
    else:
        raise ValueError(f'No such size {model_size}')

def get_path(exp_type, size):
    candidates = list(Path(f'kilian_runs/throughput_logs/logs_{exp_type}').glob(f'{job_name(size)}*.log'))
    assert len(candidates) == 1
    return candidates[0]


def bar_plot():
    model_sizes = ("125m", "760m", "3B", "8B")

    data = {
        exp_type: {
            size: read_data(get_path(exp_type, size)) for size in model_sizes
        } 
        for exp_type in ['base', 'fp8', 'activation_offloading']
    }

    # mbs_4 = np.median(read_data('logs/log-gipfel-throughput-8b-auto-causalfull-4096seq-4mbs-256gbs-1n.txt')['tokens_per_s']) / 1000
    # print(f'8B Offloaded MBS 4: {mbs_4:.1f}')


    throughputs = {}
    for exp_type, sub in data.items():
        throughputs[exp_type] = [0, 0, 0, 0]
        for size, values in sub.items():
            throughputs[exp_type][model_sizes.index(size)] = np.median(values['tokens_per_s'])/1000


    x = np.arange(len(model_sizes))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in throughputs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt=lambda x: f'{x:.1f}')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Throughput (thousand tokens per second)')
    ax.set_xlabel('Model Size')
    # ax.set_title('Throughput on Different Model Sizes and Methods')
    ax.set_xticks(x + width, model_sizes)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 100)

    plt.show()
    plt.savefig('kilian_runs/plot_bar.png', dpi=200)

def loss_plot():
    data = {
        exp_type: read_data(get_path(exp_type, '8B')) 
        for exp_type in ['base', 'fp8', 'activation_offloading']
    }

    fig, ax = plt.subplots(layout='constrained')

    for exp_type in ['base', 'fp8', 'activation_offloading']:
        ax.plot(data[exp_type]['iteration'], data[exp_type]['loss'], label=exp_type)

    ax.set_ylabel('Training Loss')
    ax.set_xlabel('Iteration')
    # ax.set_title('Training Loss (Base / Activation Offloading / FP8 Training)')
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 25)

    plt.show()
    plt.savefig('kilian_runs/plot_loss.png', dpi=200)

def grad_check():
    fp8_data = read_data(get_path('fp8', '8B'))['grad_norm']
    base_data = read_data(get_path('base', '8B'))['grad_norm']

    plt.plot(np.arange(len(fp8_data)), fp8_data, label='FP8')
    plt.plot(np.arange(len(fp8_data)), base_data, label='base')
    plt.legend()
    plt.show()
    plt.savefig('kilian_runs/grad_norm_debug.png', dpi=200)


if __name__=='__main__':
    # Median in 8B 4MBS: 10503
    grad_check()
    bar_plot()
    loss_plot()