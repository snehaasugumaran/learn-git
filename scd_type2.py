import pandas as pd


def scd_type2_merge(historical: pd.DataFrame,
                    incoming: pd.DataFrame,
                    keys: list,
                    tracked_cols: list,
                    effective_from_col: str = "effective_from",
                    effective_to_col: str = "effective_to",
                    is_current_col: str = "is_current",
                    now=None) -> pd.DataFrame:
    """Perform an SCD Type 2 merge of `incoming` into `historical`.

    - `historical` must contain the SCD metadata columns: `effective_from`,
      `effective_to`, and `is_current` (booleans).
    - `incoming` should contain the business keys and the tracked columns.
    - Returns an updated historical DataFrame with expired rows adjusted
      and new rows inserted for changes and new keys.
    """
    if now is None:
        now = pd.Timestamp.now()

    # Ensure date columns are datetime and is_current is boolean
    if effective_from_col in historical.columns:
        historical[effective_from_col] = pd.to_datetime(historical[effective_from_col])
    if effective_to_col in historical.columns:
        historical[effective_to_col] = pd.to_datetime(historical[effective_to_col])
    if is_current_col in historical.columns:
        historical[is_current_col] = historical[is_current_col].astype(bool)

    incoming = incoming.copy()
    # If incoming provides an effective_from column, use it; else use now
    if effective_from_col in incoming.columns:
        incoming[effective_from_col] = pd.to_datetime(incoming[effective_from_col])
    else:
        incoming[effective_from_col] = now

    # Work with current rows only
    current = historical[historical[is_current_col]].copy()

    merge = pd.merge(current,
                     incoming,
                     on=keys,
                     how="right",
                     suffixes=("_hist", "_inc"),
                     indicator=True)

    # Determine which incoming rows are new (right_only)
    new_mask = merge["_merge"] == "right_only"
    new_incoming = merge.loc[new_mask, incoming.columns]

    # Determine which rows exist and need to be compared
    both_mask = merge["_merge"] == "both"
    both = merge.loc[both_mask]

    changed_keys = []
    for idx, row in both.iterrows():
        # Compare tracked columns between hist and inc
        diffs = False
        for col in tracked_cols:
            hcol = f"{col}_hist"
            icol = f"{col}_inc"
            if pd.isna(row.get(hcol)) and pd.isna(row.get(icol)):
                continue
            if row.get(hcol) != row.get(icol):
                diffs = True
                break
        if diffs:
            # capture key values
            if len(keys) > 1:
                k = {kcol: row[kcol] for kcol in keys}
            else:
                k = row[keys[0]]
            changed_keys.append(k)

    # Expire changed current rows in historical
    if changed_keys:
        # Build boolean mask to find rows to expire
        expire_mask = pd.Series(False, index=historical.index)
        for k in changed_keys:
            if isinstance(k, dict):
                m = pd.Series(True, index=historical.index)
                for key_col, val in k.items():
                    m &= historical[key_col] == val
                expire_mask |= m & historical[is_current_col]
            else:
                expire_mask |= (historical[keys[0]] == k) & historical[is_current_col]

        # Set effective_to to now - 1 second and is_current False
        cutoff = now - pd.Timedelta(seconds=1)
        historical.loc[expire_mask, effective_to_col] = cutoff
        historical.loc[expire_mask, is_current_col] = False

    # Prepare new rows: changed rows' incoming versions + right_only incoming
    new_rows = []
    # Add changed incoming rows
    if changed_keys:
        # Find incoming rows that correspond to changed keys
        for k in changed_keys:
            if isinstance(k, dict):
                cond = pd.Series(True, index=incoming.index)
                for key_col, val in k.items():
                    cond &= incoming[key_col] == val
                rows = incoming[cond]
            else:
                rows = incoming[incoming[keys[0]] == k]
            for _, r in rows.iterrows():
                new_rows.append(r.to_dict())

    # Add right_only (totally new) incoming rows
    for _, r in new_incoming.iterrows():
        new_rows.append(r.to_dict())

    # Convert new_rows to DataFrame and set SCD metadata
    inserted = pd.DataFrame(new_rows)
    if not inserted.empty:
        inserted[effective_from_col] = pd.to_datetime(inserted[effective_from_col])
        inserted[effective_to_col] = pd.NaT
        inserted[is_current_col] = True

    result = pd.concat([historical, inserted], ignore_index=True, sort=False)
    # Keep a stable sort order
    sort_cols = keys + [effective_from_col]
    result = result.sort_values(by=sort_cols).reset_index(drop=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SCD Type 2 merge example")
    parser.add_argument("--historical", default="data/historical.csv")
    parser.add_argument("--incoming", default="data/incoming.csv")
    parser.add_argument("--out", default="data/merged.csv")
    args = parser.parse_args()

    hist = pd.read_csv(args.historical)
    inc = pd.read_csv(args.incoming)

    merged = scd_type2_merge(hist, inc, keys=["id"], tracked_cols=["name", "city"]) 
    merged.to_csv(args.out, index=False)
    print(f"Merged output written to {args.out}")
