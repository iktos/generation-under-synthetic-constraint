from typing import List, Optional

from guacamol.guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.guacamol.scoring_function import ScoringFunction

from .generator import RandomSmilesSampler


class RandomSamplingOptimizer(GoalDirectedGenerator):
    """
    Mock optimizer that will return molecules drawn from a random sampler
    """

    def __init__(self, sampler: RandomSmilesSampler) -> None:
        self.sampler = sampler

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None, scoring_function_start=None, benchmark_name="random_sampler") -> List[str]:
        return self.sampler.generate(number_samples=number_molecules)
