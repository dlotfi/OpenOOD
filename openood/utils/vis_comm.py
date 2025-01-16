from matplotlib import pyplot as plt

plt.style.use(['seaborn-v0_8-white'])


def save_fig_and_close(output_path):
    if not output_path.endswith('.svg'):
        raise ValueError('output_path must have an svg extension')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(output_path, bbox_inches='tight', format='svg')
    plt.close()
