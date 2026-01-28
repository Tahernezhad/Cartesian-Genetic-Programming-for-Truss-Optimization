import os
import cgp
import pytest
import numpy as np

from src.organism import Organism, MaterialProperties
from src.controller import CGPController
from src.evolutionary_algorithm import CGPGeneticAlgorithm
from main import define_seedling, define_environment


@pytest.fixture
def cgp_setup():
    """Sets up a standard truss environment and seedling for testing."""
    seedling = define_seedling()
    environment = define_environment()

    genome_params_edge = {
        "n_inputs": 2, "n_outputs": 1, "n_columns": 8, "n_rows": 2,
        "levels_back": 3, "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat)
    }
    genome_params_node = {
        "n_inputs": 2, "n_outputs": 2, "n_columns": 8, "n_rows": 2,
        "levels_back": 3, "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat)
    }

    return seedling, environment, genome_params_edge, genome_params_node


# Physics & Environment
def test_truss_physics_equilibrium(cgp_setup):
    """Verifies that the equilibrium matrix is correctly shaped for the 9-node truss."""
    seedling, env, _, _ = cgp_setup

    # 9 nodes * 2 dimensions (x, y) = 18
    eq_mat = env._get_equilibrium_mat(seedling['nodes'], seedling['edges'], env.reactions)
    assert eq_mat.shape == (18, 18)

    # Verify member lengths are strictly positive
    lengths = env.cal_mem_lens(seedling['nodes'], seedling['edges'])
    assert np.all(lengths > 0)


def test_organism_initialization(cgp_setup):
    """Ensures organism correctly inherits seedling properties and generates adjacency matrices."""
    seedling, _, _, _ = cgp_setup
    org = Organism(gen_id=0, pop_id=0, run_dir=None, seedling=seedling)

    assert org.nodes.shape == seedling['nodes'].shape
    assert org.adj_mat_node_edge.shape == (9, 15)  # 9 nodes, 15 edges


# CGP Controller & Mutation
def test_cgp_mutation_logic(cgp_setup):
    """Verifies that the mutation variance actually changes the genome."""
    seedling, _, edge_params, node_params = cgp_setup
    rng = np.random.RandomState(42)

    parent = CGPController(seedling, genome_params_edge=edge_params, genome_params_node=node_params, rng=rng)

    # Create an evolutionary algorithm instance to access mutation
    ga = CGPGeneticAlgorithm(seedling, None, node_params, edge_params, population_size=4)

    # Mutate a pair of genomes
    child_edge, child_node = ga._mutate_genome_pair((parent.genome_edge, parent.genome_node), mut_var=1.0)

    # Assert deep copies were made and differences exist
    assert parent.genome_edge is not child_edge
    assert str(parent.genome_edge) != str(child_edge)


def test_controller_evaluation_output(cgp_setup):
    """Checks if the controller's evaluate function returns a valid total reward."""
    seedling, env, edge_params, node_params = cgp_setup
    ctrl = CGPController(seedling, genome_params_edge=edge_params, genome_params_node=node_params)

    # Evaluate for a small number of steps
    reward = ctrl.evaluate(env, max_devo_step=2, grn_type="node-edge-etg")
    assert isinstance(reward, float)


# Evolution Pipeline
def test_ga_fit_pipeline(tmpdir, cgp_setup):
    """Runs a minimal genetic algorithm loop. """
    seedling, env, edge_params, node_params = cgp_setup
    run_dir = str(tmpdir.mkdir("cgp_test_run"))

    # Minimal settings for speed
    ga = CGPGeneticAlgorithm(
        seedling=seedling,
        environment=env,
        genome_params_node=node_params,
        genome_params_edge=edge_params,
        generations=2,
        population_size=4,
        top_k=2,
        run_dir=run_dir,
        num_threads=1
    )

    ga.fit()

    # Verify outputs
    assert os.path.exists(os.path.join(run_dir, "reward_plot.csv"))
    assert os.path.exists(os.path.join(run_dir, "best_controller.pkl"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))