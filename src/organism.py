import os
import copy
import numpy as np
import csv

from dataclasses import dataclass
from typing import Any, List

from src.utils import Normalizer


EPSILON = 1e-8


@dataclass
class MaterialProperties:
    name: str = "Steel"
    young_mods: np.ndarray = np.full((15,), 7e10)
    densities: np.ndarray = np.full((15,), 7872)
    poisson_ratios: np.ndarray = np.full((15,), 0.3)


@dataclass
class PhysicalState:
    forces: Any = None
    stresses: Any = None
    volumes: Any = None
    strains: Any = None
    strain_energies: Any = None
    node_disps: Any = None
    buckling: Any = None


class Organism:
    def __init__(self, gen_id, pop_id, run_dir, seedling):
        self.generation_id = gen_id
        self.population_id = pop_id
        self.devo_step = 0
        self._org_id = None
        self.run_dir = run_dir
        self.init_fitness = None
        self.init_sum_norm_fit = None

        self.nodes = copy.deepcopy(seedling["nodes"])
        self.edges = copy.deepcopy(seedling["edges"])
        self.cs_areas = copy.deepcopy(seedling["cs_areas"])
        self.node_constraints = copy.deepcopy(seedling["node_constraints"])
   
        self.adj_mat_node = self._get_adj_mat_node()
        self.adj_mat_edge = self._get_adj_mat_edge()
        self.adj_mat_node_edge = self._adj_mat_node_edge()

        self.physical_state = PhysicalState()
        self.material = copy.deepcopy(seedling["materials"])

        self.normalizer_node = Normalizer(2)
        self.normalizer_edge = Normalizer(2)

    def __str__(self):
        return f"Organism {self.organism_id} in population {self.population_id} on devo_step {self.devo_step}"
    
    @property
    def org_id(self):
        self._org_id = f"{self.generation_id}-{self.population_id}-{self.devo_step}"
        return self._org_id
    
    def _get_adj_mat_node(self):
        adj_mat = np.zeros((self.nodes.shape[0], self.nodes.shape[0]))

        for edge in self.edges:
            adj_mat[edge[0], edge[1]] = 1.0
            adj_mat[edge[1], edge[0]] = 1.0

        return adj_mat
    
    def _get_adj_mat_edge(self):
        adj_mat = np.zeros((self.edges.shape[0], self.edges.shape[0]))

        for i, edge in enumerate(self.edges):
            for j, adj_edge in enumerate(self.edges):
                if i != j:
                    for node in edge:
                        if node in adj_edge:
                            adj_mat[i, j] = 1.0

        return adj_mat
    
    def _adj_mat_node_edge(self):
        adj_mat = np.zeros((self.nodes.shape[0], self.edges.shape[0]))

        for i, edge in enumerate(self.edges):
            adj_mat[edge[0], i] = 1.0
            adj_mat[edge[1], i] = 1.0

        return adj_mat

    def sense_environment(self, environment):
        """Sense environment to update physical state."""
        self.physical_state = environment.update_physical_state(self.org_id, self.nodes, self.edges, self.cs_areas, self.physical_state, self.material)
    
    def get_cell_inputs(self, devo_step):
        N = self.nodes
        E = np.zeros((self.edges.shape[0], 2))
        E[:, :1] = self.physical_state.strain_energies.reshape(-1, 1)
        E[:, 1:] = self.physical_state.volumes.reshape(-1, 1)

        A_N = self.adj_mat_node
        A_E = self.adj_mat_edge
        A_NE = self.adj_mat_node_edge

        self.normalizer_node.observe(np.mean(N, axis=0), np.var(N, axis=0))
        N_norm = self.normalizer_node.normalize(N)

        self.normalizer_edge.observe(np.mean(E, axis=0), np.var(E, axis=0))
        E_norm = self.normalizer_edge.normalize(E)

        return N_norm, E_norm, A_N, A_E, A_NE
    
    def update_with_cell_outputs(self, E_out, N_out, devo_step):
        """Using cell as face outputs from NEAT, update artefact."""
        self._update_node_coords(N_out)
        self._update_cs_areas(E_out)
        
    def update_with_cell_outputs_edge(self, E_out, devo_step):
        """Using cell as face outputs from NEAT, update artefact."""
        self._update_cs_areas(E_out)
        
    def update_with_cell_outputs_node(self, N_out, devo_step):
        """Using cell as face outputs from NEAT, update artefact."""
        self._update_node_coords(N_out)

    def _update_node_coords(self, coord_deltas):
        """Update node coordinates in artefact using cell as edge NEAT outputs."""
        coord_deltas = np.array(coord_deltas).reshape([-1, 2])
        
        
        for constraint in self.node_constraints:
            coord_deltas[constraint, :] = 0.0

        self.nodes += coord_deltas

    def _update_cs_areas(self, area_deltas):
        """Update node coordinates in artefact using cell as edge NEAT outputs."""
        area_deltas = np.array(area_deltas).reshape((-1,))
        self.cs_areas += area_deltas
        self.cs_areas = np.where(self.cs_areas < 0.01, 0.01, self.cs_areas)

    def get_fitness(self, previous_fitness=None, init_fitness=None):
        """Get fitness function."""
        total_strain_energy = np.sum(self.physical_state.strain_energies)
        total_volume = np.sum(self.physical_state.volumes)

        #disp_magnitudes = np.linalg.norm(self.physical_state.node_disps, axis=1)
        #total_displacement = np.sum(disp_magnitudes)
        
        if init_fitness is None:
            #init_fitness = [total_strain_energy, total_volume, total_displacement]
            init_fitness = [total_strain_energy, total_volume]

        se_norm = total_strain_energy / init_fitness[0]
        vol_norm = total_volume / init_fitness[1]
        #disp_norm = total_displacement / init_fitness[2]

        #fitness = se_norm + vol_norm + disp_norm
        #fit = [total_strain_energy, total_volume, total_displacement]
        fitness = se_norm + vol_norm
        fit = [total_strain_energy, total_volume]

        if previous_fitness is None:
            reward = fitness
        else:
            reward = fitness - previous_fitness

        fitness_path = os.path.join(self.run_dir, "fitness.csv")
        self._write_csv(fitness_path, np.array(fit))

        self.devo_step += 1

        return reward, fitness, fit

    def save_organism(self):
        nodes_path = os.path.join(self.run_dir, "nodes.csv")
        edges_path = os.path.join(self.run_dir, "edges.csv")
        cs_areas_path = os.path.join(self.run_dir, "cs_areas.csv")
        strain_energy_path = os.path.join(self.run_dir, "strain_energy.csv")

        self._write_csv(nodes_path, self.nodes)
        self._write_csv(edges_path, self.edges)
        self._write_csv(cs_areas_path, self.cs_areas)
        self._write_csv(strain_energy_path, self.physical_state.strain_energies)

    def _write_csv(self, file_path, data):
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # [gen_id, pop_id, devo_step] in first three columns
            a1 = np.array([self.generation_id, self.population_id, self.devo_step])
            # Flatten the “data” array (e.g. node coords, cross-sectional areas, strain energies).
            a2 = data.flatten()
            a3 = np.concatenate((a1, a2))

            writer.writerow(iter(a3))

    def norm_z(self, input):
        return (input - np.mean(input, axis=0)) / (np.std(input, axis=0) + EPSILON)


