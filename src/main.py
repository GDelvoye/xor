import matplotlib.pyplot as plt
import numpy as np

from src.constantes import LIST_OF_IOS, MAX_WEIGHT, MIN_WEIGHT, WEIGHT_NUMBER
from src.network import Network, Weight


def get_error_matrix():
    """Return a 2D matrix of all global errors."""
    error_matrix = np.zeros((WEIGHT_NUMBER, WEIGHT_NUMBER))

    all_testable_weights = np.linspace(MIN_WEIGHT, MAX_WEIGHT, WEIGHT_NUMBER)
    # le linspace c'est un array de tous les poids a tester entre -5 et 5.

    for i in range(0, WEIGHT_NUMBER):  # lignes de la matrice
        for j in range(0, WEIGHT_NUMBER):  # colonnes de la matrice
            network = Network(
                weights=Weight(
                    all_testable_weights[i],
                    all_testable_weights[j],
                )
            )
            error_matrix[i][j] = network.global_error_to_objective(LIST_OF_IOS)
    return error_matrix


def plot_errors():
    error_matrix = get_error_matrix()

    fig, ax = plt.subplots()
    im = ax.imshow(error_matrix, extent=[-5, 5, -5, 5], cmap="seismic")
    fig.colorbar(im, ax=ax, label="Interactive colorbar")
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_errors()
