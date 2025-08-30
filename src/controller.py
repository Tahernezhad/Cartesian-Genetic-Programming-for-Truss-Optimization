import cgp  # hal-cgp
import copy
import numpy as np
from src.organism import Organism


class CGPController:
    def __init__(self, seedling, genome_node=None, genome_edge=None, genome_params_node=None, genome_params_edge=None,
                 gen_id=0, pop_id=0, run_dir=None, rng=None):
        """
        CGP-based genotype.
        """
        self.seedling = seedling
        self.gen_id = gen_id
        self.pop_id = pop_id
        self.run_dir = run_dir
        #rng = np.random.RandomState(seed)
        self._rng = rng

        if genome_edge is None:
            g_e = cgp.Genome(**genome_params_edge)
            g_e.randomize(self._rng)
            self.genome_edge = g_e
        else:
            self.genome_edge = copy.deepcopy(genome_edge)

        if genome_node is None:
            g_n = cgp.Genome(**genome_params_node)
            g_n.randomize(self._rng)
            self.genome_node = g_n
        else:
            self.genome_node = copy.deepcopy(genome_node)

        self._edge_func = None
        self._node_func = None

    def _compile_funcs(self):
        if self._edge_func is None:
            self._edge_func = cgp.CartesianGraph(self.genome_edge).to_func()
        if self._node_func is None:
            self._node_func = cgp.CartesianGraph(self.genome_node).to_func()
        return self._edge_func, self._node_func

    def evaluate(self, environment, max_devo_step, grn_type="node-edge-etg"):
        """
        1) Create an Organism
        2) For devo steps, use CGP outputs to update nodes/edges
        3) Compute cumulative reward
        """
        org = Organism(self.gen_id, self.pop_id, self.run_dir, self.seedling)
        org.sense_environment(environment)
        org.save_organism()
        fitness, _, init_fitness = org.get_fitness()

        total_reward = 0.0
        edge_func, node_func = self._compile_funcs()

        for devo_step in range(1, max_devo_step + 1):
            # Gather node & edge data:
            N, E, A_N, A_E, A_NE = org.get_cell_inputs(devo_step)
            edges = org.edges

            if grn_type == "node-edge-etg":
                E_out = self._act_edge_cgp(E, edge_func)
                org.update_with_cell_outputs_edge(E_out, devo_step)

                # re-sense environment so nodes see updated physics
                org.sense_environment(environment)

                # Get fresh inputs *after* the edge change
                N2, E2, A_N, A_E, _ = org.get_cell_inputs(devo_step)

                N_out = self._act_node_cgp(N2, node_func)
                org.update_with_cell_outputs_node(N_out, devo_step)

            elif grn_type == "node-edge-etg-advanced-agg":
                E_out = self._act_edge_cgp_advanced_aggregators(E, A_E, edge_func)
                org.update_with_cell_outputs_edge(E_out, devo_step)

                # re-sense environment so nodes see updated physics
                org.sense_environment(environment)

                # Get fresh inputs *after* the edge change
                N2, E2, A_N2, A_E2, _ = org.get_cell_inputs(devo_step)

                N_out = self._act_node_cgp_advanced_aggregators(N2, A_N2, node_func)
                org.update_with_cell_outputs_node(N_out, devo_step)

            elif grn_type == "node-edge-etg-with-neighbors":
                E_out = self._act_edge_cgp_with_neighbors(E, A_E, edge_func)
                org.update_with_cell_outputs_edge(E_out, devo_step)

                # re-sense environment so nodes see updated physics
                org.sense_environment(environment)

                # Get fresh inputs *after* the edge change
                N2, E2, A_N2, A_E2, _ = org.get_cell_inputs(devo_step)

                N_out = self._act_node_cgp_with_neighbors(N2, A_N2, node_func)
                org.update_with_cell_outputs_node(N_out, devo_step)
            else:
                print("No method!")

            # sense environment again
            org.sense_environment(environment)
            org.save_organism()

            reward, fitness, _ = org.get_fitness(fitness, init_fitness)
            total_reward += reward

        return total_reward

    def _act_node_cgp(self, N, cgp_func):
        """
        Call CGP once per node,
        passing node_x, node_y, returning dx, dy.
        """
        n_nodes, dim = N.shape
        N_out = np.zeros_like(N)
        for i in range(n_nodes):
            x_i, y_i = N[i]
            dx, dy = cgp_func(x_i, y_i)
            N_out[i, 0] = dx
            N_out[i, 1] = dy
        return np.tanh(N_out)

    def _act_node_cgp_with_neighbors(self, N, A_N, cgp_func):
        """
        A node-centric update that includes neighboring node coordinates.

        N: shape (num_nodes, 2) => Each node's (x, y)
        A_N: shape (num_nodes, num_nodes) => Node-to-node adjacency
        cgp_func: A CGP function with 4 inputs => 2 outputs, e.g.:
                  dx, dy = cgp_func(x_i, y_i, avg_x, avg_y)
        Returns:
            N_out: shape (num_nodes, 2)
                   The updated (dx, dy) for each node after activation.
        """

        n_nodes, dim = N.shape
        N_out = np.zeros_like(N)

        for i in range(n_nodes):
            x_i, y_i = N[i]


            neighbor_indices = np.where(A_N[i] == 1)[0]
            if len(neighbor_indices) > 0:
                avg_x = np.mean(N[neighbor_indices, 0])
                avg_y = np.mean(N[neighbor_indices, 1])
            else:
                avg_x, avg_y = 0.0, 0.0

            dx, dy = cgp_func(x_i, y_i, avg_x, avg_y)

            N_out[i, 0] = dx
            N_out[i, 1] = dy

        return np.tanh(N_out)

    def _act_node_cgp_advanced_aggregators(self, N, A_N, cgp_func):
        """
        Gather multiple aggregator stats from neighbor nodes:
          - (avg_x, avg_y), (min_x, max_x), (std_x, std_y)
        Then pass 1 node's own (x_i,y_i) + these 6 aggregator features (8 inputs) => 2 outputs
        """
        n_nodes = N.shape[0]
        N_out = np.zeros_like(N)
        for i in range(n_nodes):
            x_i, y_i = N[i]
            neighbors = np.where(A_N[i] == 1)[0]

            if len(neighbors) > 0:
                x_vals = N[neighbors, 0]
                y_vals = N[neighbors, 1]
                avg_x, avg_y = np.mean(x_vals), np.mean(y_vals)
                min_x, max_x = np.min(x_vals), np.max(x_vals)
                std_x = np.std(x_vals)
                min_y, max_y = np.min(y_vals), np.max(y_vals)
                std_y = np.std(y_vals)
            else:
                # default aggregator if no neighbors
                avg_x, avg_y, min_x, max_x, std_x = 0, 0, 0, 0, 0
                min_y, max_y, std_y = 0, 0, 0

            # Suppose cgp_func => 10 inputs => 2 outputs
            dx, dy = cgp_func(
                x_i, y_i,
                avg_x, avg_y,
                min_x, max_x,
                std_x, std_y,
                min_y, max_y
            )
            N_out[i, 0] = dx
            N_out[i, 1] = dy

        return np.tanh(N_out)

    def _act_edge_cgp(self, E, cgp_func):
        """
        Call CGP once per edge,
        passing e.g. (strain_energy, volume), return area delta
        """
        num_edges, _ = E.shape
        E_out = np.zeros(num_edges)
        for i in range(num_edges):
            se, vol = E[i]
            delta_area = cgp_func(se, vol)  # CGP has 2 inputs, 1 output
            E_out[i] = delta_area
        return np.tanh(E_out)

    def _act_edge_cgp_with_neighbors(self, E, A_E, cgp_func):
        """
        Aggregates each edge’s neighboring edges (via A_E) and passes:
          (strain, volume, avg_strain, avg_vol) => delta_area

        E: shape (num_edges, 2), e.g. [strain, volume] for each edge
        A_E: shape (num_edges, num_edges) => edge-to-edge adjacency
        cgp_func: a CGP function with 4 inputs => 1 output
                  e.g., delta_area = cgp_func(strain, volume, avg_strain, avg_vol)
        Returns:
            E_out: shape (num_edges,) of updated cross-section deltas
                   (or whatever single dimension is returned by CGP).
        """
        num_edges, _ = E.shape
        E_out = np.zeros(num_edges)

        for e_idx in range(num_edges):
            strain, vol = E[e_idx]

            neighbor_indices = np.where(A_E[e_idx] == 1)[0]
            if len(neighbor_indices) > 0:
                avg_strain = np.mean(E[neighbor_indices, 0])
                avg_vol = np.mean(E[neighbor_indices, 1])
            else:
                avg_strain, avg_vol = 0.0, 0.0

            delta_area = cgp_func(strain, vol, avg_strain, avg_vol)
            E_out[e_idx] = delta_area

        return np.tanh(E_out)

    def _act_edge_cgp_advanced_aggregators(self, E, A_E, cgp_func):
        """
        Gathers multiple aggregator stats (avg, min, max, std) from neighboring edges
        for each edge, then calls CGP with 10 inputs => 1 output:

          (strain, vol, avg_strain, avg_vol, min_strain, max_strain,
           std_strain, min_vol, max_vol, std_vol) => delta_area

        E: shape (num_edges, 2) => [strain, volume]
        A_E: shape (num_edges, num_edges) => edge-to-edge adjacency
        cgp_func: a CGP function with 10 inputs => 1 output
        Returns:
            E_out: shape (num_edges,) containing the updated cross-section deltas,
                   or anything else you choose to interpret from the single output.
        """

        num_edges, _ = E.shape
        E_out = np.zeros(num_edges)

        for e_idx in range(num_edges):
            strain, vol = E[e_idx]

            neighbor_indices = np.where(A_E[e_idx] == 1)[0]
            if len(neighbor_indices) > 0:
                strain_vals = E[neighbor_indices, 0]
                vol_vals = E[neighbor_indices, 1]

                avg_strain = np.mean(strain_vals)
                avg_vol = np.mean(vol_vals)
                min_strain, max_strain = np.min(strain_vals), np.max(strain_vals)
                std_strain = np.std(strain_vals)
                min_vol, max_vol = np.min(vol_vals), np.max(vol_vals)
                std_vol = np.std(vol_vals)
            else:
                avg_strain, avg_vol, min_strain, max_strain, std_strain = 0, 0, 0, 0, 0
                min_vol, max_vol, std_vol = 0, 0, 0

            delta_area = cgp_func(
                strain, vol,
                avg_strain, avg_vol,
                min_strain, max_strain,
                std_strain, std_vol,
                min_vol, max_vol
            )

            E_out[e_idx] = delta_area

        return np.tanh(E_out)






