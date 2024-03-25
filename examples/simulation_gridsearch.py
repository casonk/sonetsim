import multiprocessing as mp
import pandas as pd
import sonetsim  # v8
import os

num_logical_processors = mp.cpu_count()

def set_num_logical_processors(n):
    global num_logical_processors
    num_logical_processors = n

homophilies = [x / 4 for x in range(1, 5)]
isolations = [x / 4 for x in range(1, 5)]
insulations = [x / 4 for x in range(1, 5)]
affinities = [x / 4 for x in range(1, 5)]
num_nodes = [50 * (x + 1) ** 2 for x in range(0, 5)]
edge_multipliers = [2 * x for x in range(1, 6)]
num_communities = [2 * x**2 for x in range(1, 6)]
# resloutions      = [x/4 for x in range(0,5)]
# resloutions      = [1]


def create_simulation_and_evaluation_grid_search(seed):
    grid_search = []
    for hom in homophilies:
        for iso in isolations:
            for ins in insulations:
                for aff in affinities:
                    for nns in num_nodes:
                        for em in edge_multipliers:
                            nes = int(nns * em)
                            for ncs in num_communities:
                                grid_search.append(
                                    [hom, iso, ins, aff, nns, nes, ncs, seed]
                                )
                                # for res in resloutions:
                                #     grid_search.append([
                                #         hom, iso, ins, aff,
                                #         nns, nes, ncs, res,
                                #         seed
                                #     ])
    return grid_search

index_cols = [
    "algorithm",
    "community",
    "target_homophily",
    "target_isolation",
    "target_insulation",
    "target_affinity",
    "num_nodes",
    "num_edges",
    "edge_multiplier",
    "num_communities",
    "resolution",
    "seed",
]
num_batches = 111
# eval_dir = "/workspace/research/Funded/Ethical_Reccomendations/Paper_Poster/ACM/EchoChambers/FigTeX/EX-Networks/Method/JUPYT/Simulator/soscsim/checkpoints_v2/mega_eval_df/"
eval_dir = "/mnt/r/Funded/Ethical_Reccomendations/Paper_Poster/ACM/EchoChambers/FigTeX/EX-Networks/Method/JUPYT/Simulator/soscsim/checkpoints_v2/mega_eval_df/"

for seed in range(0, 5, 1):
    print(f"working on seed {seed}")
    grid_search = create_simulation_and_evaluation_grid_search(seed)
    _grid_search = create_simulation_and_evaluation_grid_search(seed)
    chunk_size = len(grid_search) / num_batches

    for n in range(num_batches):
        if os.path.isfile(eval_dir + f"seed_{seed}_batch_{n}.parquet"):
            print(f"skipping seed {seed} batch {n}, already processed")
            print("------------------------------------\n")
            continue
        print(f"working on seed {seed} batch {n}")
        batch = grid_search[int(chunk_size * n) : int(chunk_size * (n + 1))].copy()

        print(f"working on seed {seed} batch {n} graph simulations")
        with mp.Pool(num_logical_processors) as pool:
            graphs = pool.starmap(func=sonetsim.simulation, iterable=batch)

        _batch = _grid_search[int(chunk_size * n) : int(chunk_size * (n + 1))]
        params = []
        for param_set, graph in zip(_batch, graphs):
            params.append(param_set + list(graph))

        print(f"working on seed {seed} batch {n} parallel algorithms")
        with mp.Pool(num_logical_processors) as pool:
            evals = pool.starmap(
                func=sonetsim.evaluate_communities_parallel, iterable=params
            )

        print(f"working on seed {seed} batch {n} serial algorithms")
        for param_set in params:
            evals.append(sonetsim.evaluate_communities_serial(*param_set))

        print(f"working on seed {seed} batch {n} saving")
        mega_eval_df = pd.concat(evals)
        mega_eval_df.to_parquet(eval_dir + f"seed_{seed}_batch_{n}.parquet")
        print("------------------------------------\n")
    print("----------------------------------------------\n")