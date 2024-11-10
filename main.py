import osmnx
import networkx
import solver
import data_management


osmnx.settings.log_console = True
osmnx.settings.use_cache = True

graph = osmnx.graph.graph_from_place("New York", network_type="drive")
graph = osmnx.project_graph(graph)
graph = osmnx.routing.add_edge_speeds(graph)
graph = osmnx.routing.add_edge_travel_times(graph)
graph = networkx.convert_node_labels_to_integers(graph)

data_management.download_datasets("/home/onyxia/work/data")
data_management.clean_datasets("/home/onyxia/work/data/FOIL2013")

solver.preprocess_datasets(graph, "/home/onyxia/work/data/FOIL2013")
