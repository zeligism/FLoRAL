import sys
from floral.utils.plotting import generate_plots
    

if __name__ == "__main__":
    experiments = None if len(sys.argv) <= 1 else sys.argv[1:]
    generate_plots(experiments=experiments)
