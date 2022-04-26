from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from examples.plotting_definition import plotting_definition_template,get_comparison_decision_makers, my_comparison_decision_makers


def run_evaluation():
    #TODO: change model names
    models =  c.My_TENSORFLOW_MODELS + c.MY_BRAIN_MODELS + c.ADV_ROBUST_MODELS + c.MY_TORCHVISION_MODELS
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 32, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    #TODO change model names
    plotting_def = my_comparison_decision_makers
    figure_dirname = "analysis-figures\\"
    Plot(plot_types=plot_types, plotting_definition=plotting_def,
         figure_directory_name=figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
