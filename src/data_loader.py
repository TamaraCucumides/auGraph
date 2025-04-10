import os
import json
import pandas as pd

class RelationalDatabase:
    def __init__(self, tables: dict, foreign_keys: list):
        """
        Args:
            tables (dict): table_name -> pandas DataFrame
            foreign_keys (list): list of (src_table, src_col, dst_table, dst_col)
        """
        self.tables = tables
        self.foreign_keys = foreign_keys

    def get_table(self, name):
        return self.tables[name]

    def get_foreign_keys(self):
        return self.foreign_keys

    def get_all_attributes(self, exclude_keys=True):
        """
        Return all (table, attribute) pairs excluding FK and PK columns if requested.
        """
        all_attrs = []
        fk_keys = set((src_table, src_col) for src_table, src_col, _, _ in self.foreign_keys)
        pk_keys = set((t, f"{t[:-1]}_id") for t in self.tables)  # crude pk heuristic

        for table_name, df in self.tables.items():
            for col in df.columns:
                if exclude_keys and ((table_name, col) in fk_keys or (table_name, col) in pk_keys):
                    continue
                all_attrs.append((table_name, col))
        return all_attrs

    def print_schema(self):
        print("Tables:", list(self.tables.keys()))
        print("Foreign Keys:")
        for fk in self.foreign_keys:
            print(f"  {fk[0]}.{fk[1]} → {fk[2]}.{fk[3]}")


def load_relational_data(data_dir: str) -> RelationalDatabase:
    """
    Load relational database from CSVs and a schema.json file.

    Args:
        data_dir (str): Path to folder containing schema.json and tables.

    Returns:
        RelationalDatabase object
    """
    schema_path = os.path.join(data_dir, "schema.json")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    tables = {}
    for table_name in schema["tables"]:
        path = os.path.join(data_dir, f"{table_name}.csv")
        tables[table_name] = pd.read_csv(path)

    foreign_keys = schema["foreign_keys"]

    return RelationalDatabase(tables=tables, foreign_keys=foreign_keys)


if __name__ == "__main__":
    db = load_relational_data("data/toy")
    db.print_schema()

    print("Available attributes for promotion:")
    print(db.get_all_attributes())