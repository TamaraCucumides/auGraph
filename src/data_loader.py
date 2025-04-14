import os
import json
import pandas as pd
import networkx as nx

class RelationalDatabase:
    def __init__(self, tables: dict, foreign_keys: list, primary_keys: dict):
        """
        Args:
            tables (dict): table_name -> DataFrame
            foreign_keys (list): (src_table, src_col, dst_table, dst_col)
            primary_keys (dict): table_name -> primary key column
        """
        self.tables = tables
        self.foreign_keys = foreign_keys
        self.primary_keys = primary_keys

    def get_table(self, name):
        return self.tables[name]

    def get_foreign_keys(self):
        return self.foreign_keys

    def get_all_attributes(self, exclude_keys=True, label_table=None, max_depth=None):
        """
        Return a list of (table, attribute) pairs that are promotable.

        If label_table and max_depth are specified, restrict to attributes reachable via FK paths of length ≤ max_depth.

        Args:
            exclude_keys (bool): Exclude PK and FK columns
            label_table (str or None): If specified, only return attributes reachable from this table
            max_depth (int or None): FK path limit from label_table

        Returns:
            List of (table, attribute) pairs
        """
        # Build schema graph if needed
        if not hasattr(self, "schema_graph"):
            self.schema_graph = nx.Graph()
            for src, _, dst, _ in self.foreign_keys:
                self.schema_graph.add_edge(src, dst)

        # Build set of key columns to exclude
        key_columns = set()
        if exclude_keys:
            for table, pk in self.primary_keys.items():
                key_columns.add((table, pk))
            for src, src_col, _, _ in self.foreign_keys:
                key_columns.add((src, src_col))

        # Determine which tables are reachable from label_table
        if label_table and max_depth is not None:
            # BFS traversal
            reachable_tables = set()
            queue = [(label_table, 0)]
            visited = set()

            while queue:
                current, depth = queue.pop(0)
                if current in visited or depth > max_depth:
                    continue
                visited.add(current)
                reachable_tables.add(current)
                for neighbor in self.schema_graph.neighbors(current):
                    queue.append((neighbor, depth + 1))
        else:
            reachable_tables = set(self.tables.keys())

        # Collect promotable attributes
        promotable = []
        for table in reachable_tables:
            for col in self.tables[table].columns:
                if (table, col) not in key_columns:
                    promotable.append((table, col))

        return promotable

    def get_fk_path(self, source: str, target: str, max_depth: int = 2):
        """
        Find a FK path between `source` and `target` tables in the schema graph,
        constrained by `max_depth`. Returns a list of table names or None.
        """
        if not hasattr(self, "schema_graph"):
            # Build if not yet initialized
            import networkx as nx
            self.schema_graph = nx.Graph()
            for src, _, dst, _ in self.foreign_keys:
                self.schema_graph.add_edge(src, dst)

        if source not in self.schema_graph or target not in self.schema_graph:
            return None

        try:
            path = nx.shortest_path(self.schema_graph, source=source, target=target)
            if len(path) - 1 <= max_depth:
                return path
            return None
        except nx.NetworkXNoPath:
            return None

    def print_schema(self):
        print("Tables:", list(self.tables.keys()))
        print("Foreign Keys:")
        for fk in self.foreign_keys:
            print(f"  {fk[0]}.{fk[1]} → {fk[2]}.{fk[3]}")


def load_relational_data(data_dir: str) -> RelationalDatabase:
    schema_path = os.path.join(data_dir, "schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    tables = {
        table_name: pd.read_csv(os.path.join(data_dir, f"{table_name}.csv"))
        for table_name in schema["tables"]
    }

    return RelationalDatabase(
        tables=tables,
        foreign_keys=schema["foreign_keys"],
        primary_keys=schema["primary_keys"]
    )


if __name__ == "__main__":
    print("Data loader test")
    db = load_relational_data("data/toy")
    db.print_schema()

    print("Available attributes for promotion:")
    print(db.get_all_attributes())

    print("Max depth = 2")
    print(db.get_all_attributes(label_table="customers", max_depth=2))
