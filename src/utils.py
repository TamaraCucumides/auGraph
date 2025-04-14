import pandas as pd

def join_path_tables(db, path) -> pd.DataFrame:
    """
    Perform FK-based joins along a given path of table names.

    Args:
        db (RelationalDatabase): schema + data
        path (list[str]): ordered list of table names (e.g., ['customers', 'orders', 'products'])

    Returns:
        pd.DataFrame: a flattened DataFrame with joined content along the path
    """
    assert len(path) >= 2, "Join path must include at least two tables"

    joined = db.get_table(path[0]).copy()
    for i in range(len(path) - 1):
        t1 = path[i]
        t2 = path[i + 1]

        # Find the FK between t1 and t2 (in either direction)
        fk = next(
            (fk for fk in db.foreign_keys if
             (fk[0] == t1 and fk[2] == t2) or (fk[0] == t2 and fk[2] == t1)),
            None
        )
        if fk is None:
            raise ValueError(f"No FK path between {t1} and {t2}")

        # Extract FK info
        src_table, src_col, dst_table, dst_col = fk
        left_df = joined if t1 == src_table else db.get_table(t2).copy()
        right_df = db.get_table(dst_table if t1 == src_table else src_table).copy()

        left_key = src_col if t1 == src_table else dst_col
        right_key = dst_col if t1 == src_table else src_col

        right_df = right_df.add_prefix(f"{right_df.columns.name or t2}__")
        joined = left_df.merge(right_df, left_on=left_key, right_on=f"{t2}__{right_key}", how="left")

    return joined