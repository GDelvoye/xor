from dataclasses import dataclass

import numpy as np


@dataclass
class Weight:
    weight_1: float
    weight_2: float


@dataclass
class Input:
    input_1: float
    input_2: float


@dataclass
class IO:
    inputs: Input
    expected_output: float


@dataclass
class Network:
    weights: Weight

    def get_result(self, inputs: Input) -> float:
        return (
            inputs.input_1 * self.weights.weight_1
            + inputs.input_2 * self.weights.weight_2
        )

    def single_error_to_objective(self, io: IO) -> float:
        return np.abs(self.get_result(io.inputs) - io.expected_output)

    def global_error_to_objective(self, list_of_ios: list[IO]) -> float:
        error = 0
        for io in list_of_ios:
            error += (self.single_error_to_objective(io)) ** 2
        return error
