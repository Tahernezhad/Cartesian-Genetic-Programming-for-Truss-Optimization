import cgp
import numpy as np
import os, csv, pickle
from numpy import tanh
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from src.environment import Environment
from src.organism import Organism, MaterialProperties
from src.controller import CGPController


def define_seedling():
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


def plot_truss(ax, nodes, edges, cs_areas, title=""):
    for i, (n1, n2) in enumerate(edges):
        ax.plot([nodes[n1, 0], nodes[n2, 0]],
                [nodes[n1, 1], nodes[n2, 1]],
                lw=max(cs_areas[i] * 2, 0.1), color="blue")
    ax.scatter(nodes[:, 0], nodes[:, 1], color="red", zorder=5)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.grid(True)


def _get_plot_range(data, default_min=-1.0, default_max=1.0, buffer_frac=0.1, n_points=50):
    """Gets a linspace range for plotting, handling empty or single-point data."""
    if not data:
        return np.linspace(default_min, default_max, n_points)

    d_min = min(data)
    d_max = max(data)

    if d_min == d_max:
        d_min -= 0.5
        d_max += 0.5

    span = d_max - d_min
    buffer = span * buffer_frac
    if buffer == 0:  # Handle very small spans
        buffer = 0.1

    return np.linspace(d_min - buffer, d_max + buffer, n_points)


def replay_best(run_dir: str, grn_type: str, devo_steps: int):
    best_pkl = os.path.join(run_dir, "best_controller.pkl")
    if not os.path.exists(best_pkl):
        print(f"[ERROR] {best_pkl} not found");
        return
    best_ctrl: CGPController = pickle.load(open(best_pkl, "rb"))

    results_dir = os.path.join(run_dir, "results")
    frames_dir = os.path.join(results_dir, "best_devo_frames")
    os.makedirs(frames_dir, exist_ok=True)

    edge_expr = cgp.CartesianGraph(best_ctrl.genome_edge).to_sympy()
    node_expr = cgp.CartesianGraph(best_ctrl.genome_node).to_sympy()
    with open(os.path.join(results_dir, "controller_expressions.txt"), "w") as f:
        f.write("Edge-CGP  : {}\nNode-CGP  : {}\n".format(edge_expr, node_expr))

    env = define_environment()
    seed = define_seedling()
    org = Organism(best_ctrl.gen_id, best_ctrl.pop_id, run_dir, seed)
    org.sense_environment(env)
    fitness, _, init = org.get_fitness()

    n_nodes = org.nodes.shape[0]
    n_edges = org.edges.shape[0]
    node_hist = np.zeros((devo_steps + 1, n_nodes, 2))
    area_hist = np.zeros((devo_steps + 1, n_edges))

    node_hist[0, :, :] = org.nodes
    area_hist[0, :] = org.cs_areas

    frame_paths = []

    def save_frame(step: int, caption: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_truss(ax, org.nodes, org.edges, org.cs_areas, caption)
        fig.text(0.5, 0.02, f"Method: {grn_type}", ha="center", fontsize=9)
        p = os.path.join(frames_dir, f"frame_{step}.jpg")
        fig.savefig(p);
        plt.close(fig);
        frame_paths.append(p)

    save_frame(0, "Step 0")

    edge_f, node_f = best_ctrl._compile_funcs()
    E_hist = [];
    V_hist = [];
    C_hist = []

    all_se_inputs = []
    all_v_inputs = []
    all_dA_outputs = []
    all_x_inputs = []
    all_y_inputs = []
    all_dx_outputs = []
    all_dy_outputs = []


    for step in range(1, devo_steps + 1):
        N, E, A_N, A_E, A_NE = org.get_cell_inputs(step)

        all_se_inputs.extend(E[:, 0].tolist())
        all_v_inputs.extend(E[:, 1].tolist())


        if grn_type == "node-edge-etg":
            E_out = best_ctrl._act_edge_cgp(E, edge_f)
            org.update_with_cell_outputs_edge(E_out, step)
            org.sense_environment(env)
            N2, E2, A_N2, A_E2, _ = org.get_cell_inputs(step)
            N_out = best_ctrl._act_node_cgp(N2, node_f)
            org.update_with_cell_outputs_node(N_out, step)

        elif grn_type == "node-edge-etg-advanced-agg":
            E_out = best_ctrl._act_edge_cgp_advanced_aggregators(E, A_E, edge_f)
            org.update_with_cell_outputs_edge(E_out, step)
            org.sense_environment(env)
            N2, E2, A_N2, A_E2, _ = org.get_cell_inputs(step)
            N_out = best_ctrl._act_node_cgp_advanced_aggregators(N2, A_N2, node_f)
            org.update_with_cell_outputs_node(N_out, step)

        elif grn_type == "node-edge-etg-with-neighbors":
            E_out = best_ctrl._act_edge_cgp_with_neighbors(E, A_E, edge_f)
            org.update_with_cell_outputs_edge(E_out, step)
            org.sense_environment(env)
            N2, E2, A_N2, A_E2, _ = org.get_cell_inputs(step)
            N_out = best_ctrl._act_node_cgp_with_neighbors(N2, A_N2, node_f)
            org.update_with_cell_outputs_node(N_out, step)
        else:
            print(f"[ERROR] Unknown grn_type in replay: {grn_type}. Aborting input/output capture.")
            E_out = np.zeros(org.edges.shape[0])
            N_out = np.zeros(org.nodes.shape)
            N2 = N

        all_dA_outputs.extend(E_out.tolist())

        all_x_inputs.extend(N2[:, 0].tolist())
        all_y_inputs.extend(N2[:, 1].tolist())

        all_dx_outputs.extend(N_out[:, 0].tolist())
        all_dy_outputs.extend(N_out[:, 1].tolist())

        org.sense_environment(env)
        _, fitness, sv = org.get_fitness(fitness, init)
        E_hist.append(sv[0]);
        V_hist.append(sv[1]);
        C_hist.append(fitness)

        node_hist[step, :, :] = org.nodes
        area_hist[step, :] = org.cs_areas

        save_frame(step, f"Step {step} | E={sv[0]:.3f} V={sv[1]:.3f} Cost={fitness:.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f'CGP Input vs. Output Plots (Method: {grn_type})\n across {devo_steps} steps for all nodes/edges',
        fontsize=16)

    # Strain Energy vs. Area Delta
    axs[0, 0].scatter(all_se_inputs, all_dA_outputs, alpha=0.3, s=10, c='blue')
    axs[0, 0].set_title('Edge CGP: Strain Energy vs. Area Delta')
    axs[0, 0].set_xlabel('Input: Normalized Strain Energy (se)')
    axs[0, 0].set_ylabel('Output: Area Delta (dA)')
    axs[0, 0].grid(True)

    # Volume vs. Area Delta
    axs[0, 1].scatter(all_v_inputs, all_dA_outputs, alpha=0.3, s=10, c='green')
    axs[0, 1].set_title('Edge CGP: Volume vs. Area Delta')
    axs[0, 1].set_xlabel('Input: Normalized Volume (v)')
    axs[0, 1].set_ylabel('Output: Area Delta (dA)')
    axs[0, 1].grid(True)

    # X-coord vs. X-delta
    axs[1, 0].scatter(all_x_inputs, all_dx_outputs, alpha=0.3, s=10, c='red')
    axs[1, 0].set_title('Node CGP: X-coord vs. X-delta')
    axs[1, 0].set_xlabel('Input: Normalized X-coordinate (x)')
    axs[1, 0].set_ylabel('Output: X-delta (dx)')
    axs[1, 0].grid(True)

    # Y-coord vs. Y-delta
    axs[1, 1].scatter(all_y_inputs, all_dy_outputs, alpha=0.3, s=10, c='purple')
    axs[1, 1].set_title('Node CGP: Y-coord vs. Y-delta')
    axs[1, 1].set_xlabel('Input: Normalized Y-coordinate (y)')
    axs[1, 1].set_ylabel('Output: Y-delta (dy)')
    axs[1, 1].grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_save_path = os.path.join(results_dir, "cgp_input_output_scatter.jpg")
    fig.savefig(plot_save_path)
    plt.close(fig)
    print(f"Saved CGP I/O scatter plot -> {plot_save_path}")

    N_POINTS = 50  # Resolution of heatmap
    x_range = _get_plot_range(all_x_inputs, n_points=N_POINTS)
    y_range = _get_plot_range(all_y_inputs, n_points=N_POINTS)
    se_range = _get_plot_range(all_se_inputs, n_points=N_POINTS)
    v_range = _get_plot_range(all_v_inputs, n_points=N_POINTS)

    xx, yy = np.meshgrid(x_range, y_range)
    sese, vv = np.meshgrid(se_range, v_range)

    dx_grid = np.zeros_like(xx)
    dy_grid = np.zeros_like(xx)
    dA_grid = np.zeros_like(sese)

    if grn_type == "node-edge-etg":
        n_inputs_node = 2
        n_inputs_edge = 2
    elif grn_type == "node-edge-etg-advanced-agg":
        n_inputs_node = 10
        n_inputs_edge = 10
    elif grn_type == "node-edge-etg-with-neighbors":
        n_inputs_node = 4
        n_inputs_edge = 4
    else:
        print(f"[ERROR] Unknown grn_type '{grn_type}' in heatmap generation. Assuming 2 inputs.")
        n_inputs_node = 2
        n_inputs_edge = 2

    node_inputs = np.zeros(n_inputs_node)
    edge_inputs = np.zeros(n_inputs_edge)

    for i in range(N_POINTS):
        for j in range(N_POINTS):
            node_inputs[0] = xx[i, j]  # x
            node_inputs[1] = yy[i, j]  # y

            raw_outputs = node_f(*node_inputs)
            raw_dx, raw_dy = raw_outputs[0], raw_outputs[1]

            dx_grid[i, j] = tanh(raw_dx)
            dy_grid[i, j] = tanh(raw_dy)

    for i in range(N_POINTS):
        for j in range(N_POINTS):
            edge_inputs[0] = sese[i, j]  # se
            edge_inputs[1] = vv[i, j]  # v

            raw_dA = edge_f(*edge_inputs)

            dA_grid[i, j] = tanh(raw_dA)

    # Plot the heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'CGP Controller Desision Heatmap (Method: {grn_type})', fontsize=16)

    # Plot dx = f(x, y)
    c0 = axs[0].pcolormesh(xx, yy, dx_grid, cmap='RdBu_r', shading='gouraud')
    axs[0].set_title('Node Output: $dx = f(x, y)$')
    axs[0].set_xlabel(' X-coordinate')
    axs[0].set_ylabel(' Y-coordinate')
    fig.colorbar(c0, ax=axs[0], label='Output $dx$')

    # Plot dy = f(x, y)
    c1 = axs[1].pcolormesh(xx, yy, dy_grid, cmap='RdBu_r', shading='gouraud')
    axs[1].set_title('Node Output: $dy = f(x, y)$')
    axs[1].set_xlabel(' X-coordinate')
    axs[1].set_ylabel(' Y-coordinate')
    fig.colorbar(c1, ax=axs[1], label='Output $dy$')

    # Plot dA = f(se, v)
    c2 = axs[2].pcolormesh(sese, vv, dA_grid, cmap='RdBu_r', shading='gouraud')
    axs[2].set_title('Edge Output: $dA = f(se, v)$')
    axs[2].set_xlabel(' Strain Energy')
    axs[2].set_ylabel(' Volume')
    fig.colorbar(c2, ax=axs[2], label='Output $dA$')

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])  # Adjust for suptitle
    heatmap_save_path = os.path.join(results_dir, "cgp_decision_surface_heatmaps.jpg")
    fig.savefig(heatmap_save_path)
    plt.close(fig)
    print(f"Saved CGP heatmap plot -> {heatmap_save_path}")

    if frame_paths:
        gif_path = os.path.join(results_dir, "best_devo.gif")
        imageio.mimsave(gif_path, [imageio.imread(p) for p in frame_paths], fps=1, loop=0)

    header = ["step"] + [f"x{idx}" for idx in range(n_nodes)] + [f"y{idx}" for idx in range(n_nodes)]
    with open(os.path.join(results_dir, "node_positions.csv"), "w", newline="") as f:
        w = csv.writer(f);
        w.writerow(header)
        for s in range(devo_steps + 1):
            row = [s] + node_hist[s, :, 0].tolist() + node_hist[s, :, 1].tolist()
            w.writerow(row)

    with open(os.path.join(results_dir, "edge_areas.csv"), "w", newline="") as f:
        w = csv.writer(f);
        w.writerow(["step"] + [f"edge{e}" for e in range(n_edges)])
        for s in range(devo_steps + 1):
            w.writerow([s] + area_hist[s, :].tolist())

    t_axis = np.arange(devo_steps + 1)

    steps = np.arange(1, devo_steps + 1)
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    for ax, data, l, c in zip(
            axs, [E_hist, V_hist, C_hist],
            ["Energy", "Volume", "Cost"], ["blue", "green", "red"]):
        ax.plot(steps, data, 'o-', color=c)
        ax.set_title(l);
        ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "Node_Edge_CGP_devo_plot.jpg"))
    plt.close(fig)

    edge_cmap = plt.cm.get_cmap('tab20', n_edges)
    edge_cols = edge_cmap(np.arange(n_edges))

    fig, ax = plt.subplots(figsize=(6, 4))
    changed = False
    for e in range(n_edges):
        if np.any(np.diff(area_hist[:, e])):
            ax.plot(t_axis, area_hist[:, e],
                    '-o', lw=1, ms=3,
                    color=edge_cols[e],
                    label=f'Edge {e}')
            changed = True

    if changed:
        ax.set_xlabel('Development step')
        ax.set_ylabel('Cross-sectional area')
        ax.set_title('Evolution of *changing* member areas')
        ax.grid(True)
        ax.legend(fontsize=6, ncol=5, framealpha=.9)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, 'edge_area_evolution.jpg'))
    plt.close(fig)

    node_cmap = plt.cm.get_cmap('gist_ncar', n_nodes)
    node_cols = node_cmap(np.arange(n_nodes))

    fig, ax = plt.subplots(figsize=(6, 4))
    for n in range(n_nodes):
        ax.plot(node_hist[:, n, 0], node_hist[:, n, 1],
                '-o', lw=1, ms=3,
                color=node_cols[n],
                label=f'Node {n}')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Node trajectories over development')
    ax.grid(True)
    ax.legend(fontsize=6, ncol=5, framealpha=.9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'node_trajectories.jpg'))
    plt.close(fig)


def plot_generation_rewards(run_dir: str):
    csv_path = os.path.join(run_dir, "reward_plot.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} missing; skip reward plot.");
        return
    gens, best, avg = [], [], []
    with open(csv_path) as f:
        r = csv.reader(f);
        next(r)
        for g, b, a in r:
            gens.append(int(g));
            best.append(float(b));
            avg.append(float(a))
    plt.figure(figsize=(6, 4))
    plt.plot(gens, best, 'o-', label="Best")
    plt.plot(gens, avg, 'x-', label="Avg")
    plt.xlabel("Generation");
    plt.ylabel("Reward")
    plt.title("Reward trajectory");
    plt.grid(True);
    plt.legend()
    out = os.path.join(run_dir, "results", "Node_Edge_CGP_evo_plot.jpg")
    plt.savefig(out);
    plt.close()
    print("Saved reward plot →", out)


# ──────────────────────────── main ──────────────────────────────────
def main():
    run_dir = "data/31-08-2025-00-32-23-node-edge-etg"
    grn_type = "node-edge-etg"
    devo_steps = 10

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    replay_best(run_dir, grn_type, devo_steps)
    plot_generation_rewards(run_dir)


if __name__ == "__main__":
    main()