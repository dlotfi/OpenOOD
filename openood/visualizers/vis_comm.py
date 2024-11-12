import os

import scienceplots
import matplotlib.pyplot as plt

__all__ = ('scienceplots', 'safe_plt_str')

current_dir = os.path.dirname(__file__)
color_style_path = os.path.join(current_dir, 'material-colors.mplstyle')
plt.style.use(['science', 'no-latex', color_style_path])


def safe_plt_str(s: str) -> str:
    # cmr10 font which is used by SciencePlots no-latex style,
    # has no '_' character!
    return s.replace('_', '-')
