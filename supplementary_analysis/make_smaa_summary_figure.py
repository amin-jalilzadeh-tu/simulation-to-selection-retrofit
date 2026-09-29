"""Generate manuscript Figure 5 through the canonical main-figure generator.

Data paths and FIGDIR follow make_main_figures.py. This entry point is retained
for the existing build and extracted-S1 workflows.
"""
from make_main_figures import fig5


if __name__ == "__main__":
    fig5()
