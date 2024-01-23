import matplotlib.pyplot as plt
import numpy as np

from src.constantes import LIST_OF_IOS
from src.network import Network, Weight

MIN_WEIGHT = -5
MAX_WEIGHT = 5
WEIGHT_STEP = 11


def get_error_matrix():
    """Retrun a 2x2 matrix of all global errors."""

    linspace_of_weights = np.linspace(MIN_WEIGHT, MAX_WEIGHT, WEIGHT_STEP)
    number_of_weights = len(linspace_of_weights)
    error_matrix = np.zeros((number_of_weights, number_of_weights))

    for i in range(0, number_of_weights):
        for j in range(0, number_of_weights):
            error_matrix[i][j] = Network(
                weights=Weight(
                    linspace_of_weights[i],
                    linspace_of_weights[j],
                )
            ).global_error_to_objective(LIST_OF_IOS)
    return error_matrix


def plot_errors():
    error_matrix = get_error_matrix()

    fig, ax = plt.subplots()
    im = ax.imshow(error_matrix, extent=[-5, 5, -5, 5])

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(linspace_of_weights)), labels=linspace_of_weights)
    # ax.set_yticks(np.arange(len(linspace_of_weights)), labels=linspace_of_weights)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(linspace_of_weights)):
    #     for j in range(len(linspace_of_weights)):
    #         text = ax.text(j, i, error_matrix[i, j],
    #                     ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.colorbar(im, ax=ax, label="Interactive colorbar")
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_errors()
