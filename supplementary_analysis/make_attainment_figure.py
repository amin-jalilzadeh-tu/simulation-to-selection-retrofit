"""Generate manuscript Figures 2 and 3 through the canonical figure generator.

Data paths and FIGDIR follow make_main_figures.py. This entry point is retained
for the existing build and extracted-S1 workflows.
"""
from make_main_figures import fig2


if __name__ == "__main__":
    fig2()
