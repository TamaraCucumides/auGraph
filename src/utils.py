import pandas as pd

def join_path_tables(db, path):
    """
    Join tables along the given FK path in order.

    - The first table is used as the base and is NOT prefixed.
    - All other tables are prefixed, except for their join key column.
    - All FK-relevant columns are preserved across joins.

    Args:
        db (RelationalDatabase): schema and data
        path (list[str]): ordered list of table names (e.g. ['products', 'orders', 'customers'])

    Returns:
        pd.DataFrame: joined DataFrame with prefixed columns
    """
    assert len(path) >= 2, "Join path must contain at least two tables."

    joined = db.get_table(path[0]).copy()

    # Collect all FK columns that might be needed
    all_fk_keys = set()
    for src, src_col, dst, dst_col in db.foreign_keys:
        all_fk_keys.add(src_col)
        all_fk_keys.add(dst_col)

    for i in range(len(path) - 1):
        t1 = path[i]
        t2 = path[i + 1]

        # Find FK between t1 and t2
        fk = next(
            (fk for fk in db.foreign_keys if
             (fk[0] == t1 and fk[2] == t2) or
             (fk[0] == t2 and fk[2] == t1)),
            None
        )
        if fk is None:
            raise ValueError(f"No FK found between {t1} and {t2}")

        src_table, src_col, dst_table, dst_col = fk

        if t1 == src_table:
            left_key = src_col
            right_key = dst_col
            right_table = dst_table
        else:
            left_key = dst_col
            right_key = src_col
            right_table = src_table

        right_df = db.get_table(right_table).copy()

        # Rename non-key columns in right table
        right_prefix = f"{right_table}__"
        cols_to_prefix = [col for col in right_df.columns if col != right_key]
        right_renamed = right_df.rename(columns={col: right_prefix + col for col in cols_to_prefix})

        # Ensure join key is preserved
        if right_key not in right_renamed.columns:
            right_renamed[right_key] = right_df[right_key]

        # Merge
        joined = joined.merge(
            right_renamed,
            left_on=left_key,
            right_on=right_key,
            how="left"
        )

        # ✅ Preserve all FK-relevant columns for future joins
        for fk_col in all_fk_keys:
            if fk_col in right_df.columns and fk_col not in joined.columns:
                joined[fk_col] = right_df[fk_col]

    return joined