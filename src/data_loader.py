import os
import json
import pandas as pd

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

    def get_all_attributes(self, exclude_keys=True):
        """
        Return all (table, attribute) pairs for possible promotion.
        
        f exclude_keys=True, primary and foreign keys are excluded.
        """
        all_attrs = []

        # Set of (table, column) pairs to exclude
        key_columns = set()

        if exclude_keys:
            # Add all PKs
            for table, pk_col in self.primary_keys.items():
                key_columns.add((table, pk_col))

            # Add all FKs
            for src_table, src_col, _, _ in self.foreign_keys:
                key_columns.add((src_table, src_col))

        for table_name, df in self.tables.items():
            for col in df.columns:
                if (table_name, col) in key_columns:
                    continue
                all_attrs.append((table_name, col))

        return all_attrs

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