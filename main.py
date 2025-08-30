import cgp
import numpy as np
from time import time
from src.utils import make_run_dir
from src.environment import Environment
from src.organism import MaterialProperties
from src.evolutionary_algorithm import CGPGeneticAlgorithm


def define_seedling():
    """Produces the seedling parameters.

    The seedling is the initial starting point of development for each organism.

    Returns:
        A dictionary containing parameters of the seedling.
    """
    nodes = np.array([[0.0, 0.0],
                      [12.5, 21.650635],
                      [25.0, 0.0],
                      [37.5, 21.650635],
                      [50.0, 0.0],
                      [62.5, 21.650635],
                      [75.0, 0.0],
                      [87.5, 21.650635],
                      [100.0, 0.0]])

    edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 6],
                      [5, 7], [6, 7], [6, 8], [7, 8]])

    cs_areas = np.full((edges.shape[0],), 1.0)
    node_constraints = np.array([0, 2, 4, 6, 8])

    materials = MaterialProperties()
    materials.names = ["Steel"] * edges.shape[0]
    materials.young_mods = np.full((edges.shape[0],), 7e10)
    materials.densities = np.full((edges.shape[0],), 7872)
    materials.poisson_ratios = np.full((edges.shape[0],), 0.3)

    seedling = {"nodes": nodes, "edges": edges, "cs_areas": cs_areas, "materials": materials,
                "node_constraints": node_constraints}

    return seedling


def define_environment():
    reactions = np.array([[1, 1],  # Node 0
                          [0, 0],  # Node 1
                          [0, 0],  # Node 2
                          [0, 0],  # Node 3
                          [0, 0],  # Node 4
                          [0, 0],  # Node 5
                          [0, 0],  # Node 6
                          [0, 0],  # Node 7
                          [0, 1]])  # Node 8

    loads = np.array([[0, 0],  # Node 0
                      [0, 0],  # Node 1
                      [0, 0],  # Node 2
                      [0, 0],  # Node 3
                      [0, -17000],  # Node 4
                      [0, 0],  # Node 5
                      [0, 0],  # Node 6
                      [0, 0],  # Node 7
                      [0, 0]])  # Node 8

    environment = Environment(reactions=reactions, loads=loads)

    return environment

if __name__ == "__main__":
    for i in range(1, 2):

        grn_type = "node-edge-etg"
        seed=i
        np.random.seed(seed)
        print(f"Random seed: {seed}")

        run_dir = make_run_dir(grn_type)
        seedling = define_seedling()
        environment = define_environment()

        if grn_type == "node-edge-etg":
            genome_params_edge = {
                "n_inputs": 2,
                "n_outputs": 1,
                "n_columns": 8,
                "n_rows": 2,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
            genome_params_node = {
                "n_inputs": 2,
                "n_outputs": 2,
                "n_columns": 8,
                "n_rows": 2,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
        elif grn_type == "node-edge-etg-advanced-agg":
            genome_params_edge = {
                "n_inputs": 10,
                "n_outputs": 1,
                "n_columns": 10,
                "n_rows": 3,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
            genome_params_node = {
                "n_inputs": 10,
                "n_outputs": 2,
                "n_columns": 10,
                "n_rows": 3,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
        elif grn_type == "node-edge-etg-with-neighbors":
            genome_params_edge = {
                "n_inputs": 4,
                "n_outputs": 1,
                "n_columns": 10,
                "n_rows": 5,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
            genome_params_node = {
                "n_inputs": 4,
                "n_outputs": 2,
                "n_columns": 10,
                "n_rows": 5,
                "levels_back": 3,
                "primitives": (
                    cgp.Add,
                    cgp.Sub,
                    cgp.Mul,
                    cgp.ConstantFloat,
                ),
            }
        else:
            print('new methods should be implemented')

        ga = CGPGeneticAlgorithm(
            seedling=seedling,
            environment=environment,
            genome_params_node=genome_params_node,
            genome_params_edge=genome_params_edge,
            generations=150,
            population_size=200,
            population_decay=1.0,
            min_population_size=128,
            run_dir=run_dir,
            initial_epsilon=1.0,
            epsilon_taper=0.4,
            crossover_rate= 0.4,
            num_devo_steps=10,
            top_k=32,
            grn_type=grn_type,
            verbose=True,
            num_threads=1,
            seed=seed
        )

        start_time = time()
        ga.fit()
        print(f"Finished in {round((time()-start_time)/60, 3)} minutes")

        print("The end of program")
