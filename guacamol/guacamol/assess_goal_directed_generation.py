import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Any, Dict
from guacamol.guacamol.goal_directed_benchmark import GoalDirectedBenchmark, GoalDirectedBenchmarkResult
from guacamol.guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.guacamol.utils.data import get_time_string

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def assess_goal_directed_generation(goal_directed_molecule_generator: GoalDirectedGenerator,
                                    json_output_file='output_goal_directed.json',
                                    benchmark_version='v1', synth_score = None) -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    logger.info(f'Benchmarking goal-directed molecule generation, version {benchmark_version}')
    benchmarks = goal_directed_benchmark_suite(version_name=benchmark_version, synth_score=synth_score)

    results = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmarks=benchmarks, json_output_file=json_output_file)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_goal_directed_benchmarks(goal_directed_molecule_generator: GoalDirectedGenerator,
                                       benchmarks: List[GoalDirectedBenchmark], json_output_file
                                       ) -> List[GoalDirectedBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        goal_directed_molecule_generator: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    logger.info(f'Number of benchmarks: {len(benchmarks)}')
    benchmark_results: Dict[str, Any] = OrderedDict()

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f'Running benchmark {i}/{len(benchmarks)}: {benchmark.name}')
        result = benchmark.assess_model(goal_directed_molecule_generator)
        logger.info(f'Save results to file {json_output_file}')
        logger.info(f'Results for the benchmark "{result.benchmark_name}":')
        logger.info(f'  Score: {result.score:.6f}')
        logger.info(f'  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}')
        logger.info(f'  Metadata: {result.metadata}')
        results.append(result)
        benchmark_results['results'] = [vars(result) for result in results]
        with open(json_output_file, 'wt') as f:
            f.write(json.dumps(benchmark_results, indent=4))


    logger.info('Finished execution of the benchmarks')

    return results
