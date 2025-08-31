import os
import csv
import copy
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from src.controller import CGPController



class CGPGeneticAlgorithm:
    def __init__(self, seedling, environment, genome_params_node,genome_params_edge,
                 generations=30, population_size=64,
                 population_decay=1.0, min_population_size=64, run_dir=None,
                 initial_epsilon=1.0, epsilon_taper=0.3, crossover_rate = 0.3,
                 verbose=False, print_every=1, num_devo_steps=10,
                 top_k=4, grn_type="node-edge-etg", num_threads=1, seed=2):

        self.generations = generations
        self.population_size = population_size
        self.min_population_size = min_population_size
        self.population_decay = population_decay
        self.epsilon_taper = epsilon_taper
        self.crossover_rate = crossover_rate
        self.initial_epsilon = initial_epsilon
        self.top_k = top_k
        self.verbose = verbose
        self.print_every = print_every
        self.num_devo_steps = num_devo_steps
        self.run_dir = run_dir
        self.seedling = seedling
        self.environment = environment
        self.grn_type = grn_type

        self.num_threads = num_threads
        self.max_threads = cpu_count()
        self.num_processes = min(self.num_threads, self.max_threads)
        self._rng = np.random.RandomState(seed)

        # Build initial population
        self.controllers = []
        for pop_id in range(population_size):
            ctrl = CGPController(
                seedling=self.seedling,
                genome_params_node=genome_params_node,
                genome_params_edge=genome_params_edge,
                gen_id=0,
                pop_id=pop_id,
                run_dir=self.run_dir,
                rng = self._rng
            )
            self.controllers.append(ctrl)

        self.best_controller = None

    def evaluate_controller(self, controller):
        """Evaluate a single CGP controller and return its raw reward"""
        return controller.evaluate(self.environment, self.num_devo_steps, self.grn_type)

    def fit(self):
        gens = []
        highest_rewards = []
        avg_rewards = []

        top_controllers = []
        top_rewards = np.array([])

        for gen in range(self.generations):
            with Pool(processes=self.num_processes) as pool:
                new_rewards = np.array(pool.map(self.evaluate_controller, self.controllers))

            new_rewards = new_rewards * -1

            rewards = np.zeros((len(self.controllers) + len(top_controllers),))
            rewards[:len(new_rewards)] = new_rewards

            if gen != 0 and len(top_rewards) > 0:
                rewards[len(new_rewards):] = top_rewards

                self.controllers = self.controllers + top_controllers

            best_idx = np.argmax(rewards)
            self.best_controller = self.controllers[best_idx]

            top_controllers, top_rewards = self._select_top_k(gen, rewards)

            current_best = rewards.max()
            current_avg = rewards.mean()
            if self.verbose and ((gen % self.print_every == 0) or (gen == 0)):
                print(f"Generation: {gen} | Highest Reward: {current_best:.6f} | Average Reward: {current_avg:.6f}")

            gens.append(gen)
            highest_rewards.append(round(current_best, 6))
            avg_rewards.append(round(current_avg, 6))

        self._write_reward_csv(gens, highest_rewards, avg_rewards)
        self.save_best_controller(self.best_controller)
        print(f"Best controller: Gen={self.best_controller.gen_id}, Pop={self.best_controller.pop_id}")

    def _select_top_k(self, gen, rewards):
        """
        Keep the best self.top_k controllers, then fill the new population
        with mutated copies of those elites.
        """

        idxs = np.argpartition(rewards, -self.top_k)[-self.top_k:]
        idxs = idxs[np.argsort(-rewards[idxs])]  # sort high-to-low
        elites = [self.controllers[i] for i in idxs]
        elite_rewards = rewards[idxs]

        self._next_population_size()
        need = self.population_size - self.top_k
        #mut_var = self._get_mutation_variance(gen)
        mut_var = self._get_mutation_variance_linear(gen)

        rng = self._rng
        new_population = []
        next_pop_id = self.top_k

        while len(new_population) < need:
            parent = rng.choice(elites)

            child_edge, child_node = self._mutate_genome_pair(
                (parent.genome_edge, parent.genome_node), mut_var)

            child = CGPController(
                seedling=self.seedling,
                genome_edge=child_edge,
                genome_node=child_node,
                gen_id=gen + 1,
                pop_id=next_pop_id,
                run_dir=self.run_dir,
                rng = self._rng)

            new_population.append(child)
            next_pop_id += 1

        self.controllers = new_population
        return elites, elite_rewards

    def _mutate_genome_pair(self, genome_pair, mut_var):
        g_edge, g_node = genome_pair
        g_edge = copy.deepcopy(g_edge);
        g_node = copy.deepcopy(g_node)
        g_edge.mutate(mutation_rate=mut_var, rng=self._rng)
        g_node.mutate(mutation_rate=mut_var, rng=self._rng)
        return g_edge, g_node

    def _get_mutation_variance(self, generation):
        """Decreasing epsilon over generations."""
        return self.initial_epsilon / (1 + self.epsilon_taper * generation)

    def _get_mutation_variance_linear(self, generation):
        e0 = self.initial_epsilon  # start
        ef = getattr(self, "final_epsilon", 0.01)  # add a field, or hard-code
        T = self.generations
        var = e0 - (e0 - ef) * (generation / T)
        return max(ef, var)

    def _next_population_size(self):
        """Reduce population size over generations."""
        self.population_size = int(max(self.population_size * self.population_decay, self.min_population_size))

    def _write_reward_csv(self, gens, bests, avgs):
        if not self.run_dir:
            return
        file_path = os.path.join(self.run_dir, "reward_plot.csv")
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Generation","BestReward","AvgReward"])
            for i in range(len(gens)):
                writer.writerow([gens[i], bests[i], avgs[i]])

    def save_best_controller(self, best_ctrl):
        """Save the champion to disk."""
        if not self.run_dir:
            return
        file_path = os.path.join(self.run_dir, "best_controller.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(best_ctrl, f, protocol=pickle.HIGHEST_PROTOCOL)
