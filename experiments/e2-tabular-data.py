from numpy.core.fromnumeric import nonzero
from ucimlrepo import fetch_ucirepo

def graph_from_dataframe(data, type="fk"):
    graph, node_id_map = None, None
    if type=="knn":
        pass

    else:
        pass
        # create node-per-row
    
    return graph, node_id_map


if __name__ == "__main__":
    mushroom = fetch_ucirepo(id=73) 

    df_mushroom = mushroom.data.features 
    target = mushroom.data.targets 
    num_classes = 2

    # create graph




