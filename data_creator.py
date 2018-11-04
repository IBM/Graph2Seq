"""this python file is used to construct the fake data for the model"""
import random
import json
import numpy as np

from networkx import *
import networkx.algorithms as nxalg

def create_random_graph(type, filePath, numberOfCase, graph_scale):
    """

    :param type: the graph type
    :param filePath: the output file path
    :param numberOfCase: the number of examples
    :return:
    """
    with open(filePath, "w+") as f:
        degree = 0.0
        for _ in range(numberOfCase):
            info = {}
            graph_node_size = graph_scale
            edge_prob = 0.3

            while True:
                edge_count = 0.0
                if type == "random":
                    graph = nx.gnp_random_graph(graph_node_size, edge_prob, directed=True)
                    for id in graph.edge:
                        edge_count += len(graph.edge[id])
                    start = random.randint(0, graph_node_size - 1)
                    adj = nx.shortest_path(graph, start)

                    max_len = 0
                    path = []
                    paths = []
                    for neighbor in adj:
                        if len(adj[neighbor]) > max_len and neighbor != start:
                            paths = []
                            max_len = len(adj[neighbor])
                            path = adj[neighbor]
                            end = neighbor
                            for p in nx.all_shortest_paths(graph, start, end):
                                paths.append(p)

                    if len(path) > 0 and path[0] == start and len(path) == 3 and len(paths) == 1:
                        degree += edge_count / graph_node_size
                        break

                elif type == "no-cycle":
                    graph = nx.DiGraph()
                    for i in range(graph_node_size):
                        nodes = graph.nodes()
                        if len(nodes) == 0:
                            graph.add_node(i)
                        else:
                            size = random.randint(1, min(i, 2));
                            fathers = random.sample(range(0, i), size)
                            for father in fathers:
                                graph.add_edge(father, i)
                    for id in graph.edge:
                        edge_count += len(graph.edge[id])
                    start = 0
                    end = graph_node_size-1
                    path = nx.shortest_path(graph, 0, graph_node_size-1)
                    paths = [p for p in nx.all_shortest_paths(graph, 0, graph_node_size-1)]
                    if len(path) >= 4 and len(paths) == 1:
                        degree += edge_count / graph_node_size
                        break

                elif type == "baseline":
                    num_nodes = graph_node_size
                    graph = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, 3, edge_prob)
                    for id in graph.edge:
                        edge_count += len(graph.edge[id])
                    start, end = np.random.randint(num_nodes, size=2)

                    if start == end:
                        continue  # reject trivial paths

                    paths = list(nxalg.all_shortest_paths(graph, source=start, target=end))

                    if len(paths) > 1:
                        continue  # reject when more than one shortest path

                    path = paths[0]

                    if len(path) != 4:
                        continue
                    degree += edge_count / graph_node_size
                    break

            adj_list = graph.adjacency_list()


            g_ids = {}
            g_ids_features = {}
            g_adj = {}
            for i in range(graph_node_size):
                g_ids[i] = i
                if i == start:
                    g_ids_features[i] = "START"
                elif i == end:
                    g_ids_features[i] = "END"
                else:
                    # g_ids_features[i] = str(i+10)
                    g_ids_features[i] = str(random.randint(1, 15))
                g_adj[i] = adj_list[i]

            # print start, end, path
            text = ""
            for id in path:
                text += g_ids_features[id] + " "

            info["seq"] = text.strip()
            info["g_ids"] = g_ids
            info['g_ids_features'] = g_ids_features
            info['g_adj'] = g_adj
            f.write(json.dumps(info)+"\n")

        print("average degree in the graph is :{}".format(degree/numberOfCase))

if __name__ == "__main__":
    create_random_graph("no-cycle", "data/no_cycle/train.data", 1000, graph_scale=100)
    create_random_graph("no-cycle", "data/no_cycle/dev.data", 1000, graph_scale=100)
    create_random_graph("no-cycle", "data/no_cycle/test.data", 1000, graph_scale=100)


