import os, csv, pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cgp

from src.environment import Environment
from src.organism    import Organism, MaterialProperties
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
                      [10000, 0],  # Node 1
                      [0, 0],  # Node 2
                      [0, 0],  # Node 3
                      [0, -17000],  # Node 4
                      [0, 0],  # Node 5
                      [0, 0],  # Node 6
                      [0, -17000],  # Node 7
                      [0, 0]])  # Node 8

    environment = Environment(reactions=reactions, loads=loads)
    return environment

def plot_truss(ax, nodes, edges, cs_areas, title=""):
    for i, (n1, n2) in enumerate(edges):
        ax.plot([nodes[n1,0], nodes[n2,0]],
                [nodes[n1,1], nodes[n2,1]],
                lw=max(cs_areas[i]*2, 0.1), color="blue")
    ax.scatter(nodes[:,0], nodes[:,1], color="red", zorder=5)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.grid(True)


def replay_best(run_dir: str, grn_type: str, devo_steps: int):
    best_pkl = os.path.join(run_dir, "best_controller.pkl")
    if not os.path.exists(best_pkl):
        print(f"[ERROR] {best_pkl} not found"); return
    best_ctrl: CGPController = pickle.load(open(best_pkl, "rb"))


    results_dir = os.path.join(run_dir, "results")
    frames_dir  = os.path.join(results_dir, "best_devo_frames")
    os.makedirs(frames_dir, exist_ok=True)


    edge_expr = cgp.CartesianGraph(best_ctrl.genome_edge).to_sympy()
    node_expr = cgp.CartesianGraph(best_ctrl.genome_node).to_sympy()
    with open(os.path.join(results_dir,"controller_expressions.txt"),"w") as f:
        f.write("Edge-CGP  : {}\nNode-CGP  : {}\n".format(edge_expr, node_expr))

    env  = define_environment()
    seed = define_seedling()
    org  = Organism(best_ctrl.gen_id, best_ctrl.pop_id, run_dir, seed)
    org.sense_environment(env)
    fitness, _, init = org.get_fitness()

    n_nodes  = org.nodes.shape[0]
    n_edges  = org.edges.shape[0]
    node_hist  = np.zeros((devo_steps+1, n_nodes, 2))
    area_hist  = np.zeros((devo_steps+1, n_edges))

    node_hist[0,:,:] = org.nodes
    area_hist[0,:]   = org.cs_areas

    frame_paths=[]
    def save_frame(step:int, caption:str):
        fig,ax=plt.subplots(figsize=(6,4))
        plot_truss(ax, org.nodes, org.edges, org.cs_areas, caption)
        fig.text(0.5,0.02,f"Method: {grn_type}",ha="center",fontsize=9)
        p=os.path.join(frames_dir,f"frame_{step}.eps")
        fig.savefig(p); plt.close(fig); frame_paths.append(p)

    save_frame(0,"Step 0")

    edge_f,node_f = best_ctrl._compile_funcs()
    E_hist=[]; V_hist=[]; C_hist=[]


    for step in range(1, devo_steps+1):
        N,E,A_N,A_E,A_NE = org.get_cell_inputs(step)

        E_out = best_ctrl._act_edge_cgp(E, edge_f)
        org.update_with_cell_outputs_edge(E_out, step)

        org.sense_environment(env)
        N2, *_ = org.get_cell_inputs(step)
        N_out = best_ctrl._act_node_cgp(N2, node_f)
        org.update_with_cell_outputs_node(N_out, step)

        org.sense_environment(env)
        _, fitness, sv = org.get_fitness(fitness, init)
        E_hist.append(sv[0]); V_hist.append(sv[1]); C_hist.append(fitness)

        node_hist[step,:,:] = org.nodes
        area_hist[step,:]   = org.cs_areas

        save_frame(step,f"Step {step} | E={sv[0]:.3f} V={sv[1]:.3f} Cost={fitness:.3f}")

    if frame_paths:
        gif_path=os.path.join(results_dir,"best_devo.gif")
        imageio.mimsave(gif_path,[imageio.imread(p) for p in frame_paths],fps=1, loop=0)


    header = ["step"]+[f"x{idx}" for idx in range(n_nodes)]+[f"y{idx}" for idx in range(n_nodes)]
    with open(os.path.join(results_dir,"node_positions.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for s in range(devo_steps+1):
            row=[s]+node_hist[s,:,0].tolist()+node_hist[s,:,1].tolist()
            w.writerow(row)

    with open(os.path.join(results_dir,"edge_areas.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["step"]+[f"edge{e}" for e in range(n_edges)])
        for s in range(devo_steps+1):
            w.writerow([s]+area_hist[s,:].tolist())

    t_axis = np.arange(devo_steps+1)

    steps=np.arange(1,devo_steps+1)
    fig,axs=plt.subplots(3,1,figsize=(6,8))
    for ax,data,l,c in zip(
            axs,[E_hist,V_hist,C_hist],
            ["Energy","Volume","Cost"],["blue","green","red"]):
        ax.plot(steps,data,'o-',color=c)
        ax.set_title(l); ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir,"Node_Edge_CGP_devo_plot.eps"))
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
        fig.savefig(os.path.join(results_dir, 'edge_area_evolution.eps'))
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
    fig.savefig(os.path.join(results_dir, 'node_trajectories.eps'))
    plt.close(fig)


def plot_generation_rewards(run_dir: str):
    csv_path = os.path.join(run_dir, "reward_plot.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} missing; skip reward plot."); return
    gens, best, avg = [], [], []
    with open(csv_path) as f:
        r = csv.reader(f); next(r)
        for g, b, a in r:
            gens.append(int(g)); best.append(float(b)); avg.append(float(a))
    plt.figure(figsize=(6,4))
    plt.plot(gens, best, 'o-', label="Best")
    plt.plot(gens, avg,  'x-', label="Avg")
    plt.xlabel("Generation"); plt.ylabel("Reward")
    plt.title("Reward trajectory"); plt.grid(True); plt.legend()
    out = os.path.join(run_dir, "results", "Node_Edge_CGP_evo_plot.eps")
    plt.savefig(out); plt.close()
    print("Saved reward plot →", out)


# ──────────────────────────── main ──────────────────────────────────
def main():
    # ----- USER SETTINGS -----
    run_dir   = "data/29-05-2025-17-24-44-node-edge-etg"
    grn_type  = "node-edge-etg"
    devo_steps = 10
    # -------------------------

    replay_best(run_dir, grn_type, devo_steps)
    plot_generation_rewards(run_dir)

if __name__ == "__main__":
    main()
