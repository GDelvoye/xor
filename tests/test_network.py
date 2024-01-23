import numpy as np

from src.constantes import LIST_OF_IOS
from src.network import IO, Input, Network, Weight


def test_get_result():
    # Given
    network = Network(Weight(0.7, 0.3))
    # When
    result = network.get_result(Input(1, 1))
    # Then
    assert result == 1


def test_single_error_to_objective():
    # Given
    network = Network(Weight(0.7, 0.3))
    inputs = Input(1, 1)
    expected_output = 0.5
    io = IO(inputs, expected_output)
    # When
    result = network.single_error_to_objective(io)
    # Then
    assert result == 0.5


def test_global_error_to_objective():
    # Given
    network = Network(Weight(0.7, 0.3))
    list_of_ios = LIST_OF_IOS
    # When
    result = network.global_error_to_objective(list_of_ios)
    # Then
    assert np.isclose(result, 0.94)
