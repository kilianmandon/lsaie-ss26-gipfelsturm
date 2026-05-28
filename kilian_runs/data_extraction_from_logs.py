import re
from matplotlib import pyplot as plt
import numpy as np


def read_data(log_file):
    with open(log_file, 'r') as f:
        data_lines = f.readlines()
        # Source - https://stackoverflow.com/a/53870514
        # Posted by Warren Weckesser, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-05-27, License - CC BY-SA 4.0

        number_pattern = r"[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?"

        # regex = rf'.*iteration\s*({number_pattern}).*throughput per GPU (TFLOP/s/GPU):\s*({number_pattern}).*tokens/sec/GPU:\s*(\d*\.?\d*).*lm loss:\s*{number_pattern}.* grad norm:\s*{number_pattern}[^\n]\n'
        regex = rf'.*iteration\s*({number_pattern}).*throughput per GPU \(TFLOP/s/GPU\):\s*({number_pattern}).*tokens/sec/GPU:\s*({number_pattern}).*lm loss:\s*({number_pattern}).*grad norm:\s*({number_pattern})'


        iterations = []
        tflop_per_s_list = []
        tokens_per_s_list = []
        loss_list = []
        grad_norm_list = []
        pattern = re.compile(regex)
        for data_line in data_lines:
            match = pattern.match(data_line)
            if not match:
                continue

            iterations.append(int(match.group(1)))
            tflop_per_s_list.append(float(match.group(2)))
            tokens_per_s_list.append(float(match.group(3)))
            loss_list.append(float(match.group(4)))
            grad_norm_list.append(float(match.group(5)))

        iterations = np.array(iterations)
        tflop_per_s = np.array(tflop_per_s_list)
        tokens_per_s = np.array(tokens_per_s_list)
        loss = np.array(loss_list)
        grad_norm = np.array(grad_norm_list)

        return {
            'iteration': iterations,
            'tflop_per_s': tflop_per_s,
            'tokens_per_s': tokens_per_s,
            'loss': loss,
            'grad_norm': grad_norm
        }

if __name__=='__main__':
    d1 = read_data('logs/log-gipfel-throughput-3b-auto-causalfull-4096seq-4mbs-256gbs-1n.txt')
    d2 = read_data('kilian_runs/throughput_logs/logs_activation_offloading/log-gipfel-throughput-3b-auto-causalfull-4096seq-4mbs-256gbs-1n.txt')
    print(d1['iteration'])

    plt.plot(d1['iteration'], d1['loss'], label='FP8 Training')
    plt.plot(d2['iteration'], d2['loss'], label='Offloading')
    plt.legend()
    plt.show()
    plt.savefig('base_fig.png')