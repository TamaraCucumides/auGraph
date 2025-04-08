import pandas as pd

class RelationalDatabase:
    def __init__(self, tables: dict, foreign_keys: list):
        """
        Args:
            tables (dict): Maps table names to pandas DataFrames.
            foreign_keys (list): List of FK tuples: (src_table, src_col, dst_table, dst_col)
        """
        self.tables = tables
        self.foreign_keys = foreign_keys
        self.schema_graph = self._build_schema_graph()

    def _build_schema_graph(self):
        """Create a graph representation of the schema."""
        pass

    def get_table(self, name):
        return self.tables[name]

    def get_foreign_keys(self):
        return self.foreign_keys

    def get_all_attributes(self, exclude_keys=True):
        """
        Return a list of non-key attributes for possible promotion.
        """
        pass

    def entities(self):
        """Return all row-level entity IDs, globally or per table."""
        pass

    def print_schema(self):
        """Visual debugging of table connections and structure."""
        pass

if __name__ == "__main__":
    pass