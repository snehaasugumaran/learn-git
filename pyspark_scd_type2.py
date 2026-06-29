from pyspark.sql import SparkSession, functions as F
from pyspark.sql import types as T


def _col_ne(col_h, col_i):
    return (
        (F.col(col_h).isNull() & F.col(col_i).isNotNull())
        | (F.col(col_h).isNotNull() & F.col(col_i).isNull())
        | (F.col(col_h) != F.col(col_i))
    )


def scd_type2_merge(historical_df, incoming_df, keys, tracked_cols,
                    effective_from_col='effective_from',
                    effective_to_col='effective_to',
                    is_current_col='is_current'):
    """Perform SCD Type 2 merge using PySpark DataFrames.

    Args:
        historical_df: Spark DataFrame containing existing SCD rows (may include older versions).
        incoming_df: Spark DataFrame with new/changed rows.
        keys: list of business key column names.
        tracked_cols: list of columns to detect changes on.

    Returns:
        Spark DataFrame with updated historical rows (expired rows updated and new rows inserted).
    """
    # Ensure incoming has an effective_from column
    inc = incoming_df
    if effective_from_col not in inc.columns:
        inc = inc.withColumn(effective_from_col, F.current_timestamp())

    # Normalize types for historical
    hist = historical_df
    # Work with current rows from history for change detection
    hist_current = hist.filter(F.col(is_current_col) == True)

    # Join hist_current with inc to find changed and new rows
    joined_hi = hist_current.alias('h').join(inc.alias('i'), keys, 'inner')

    # Build change expression on joined aliases
    change_expr = None
    for c in tracked_cols:
        expr = _col_ne(f'h.{c}', f'i.{c}')
        change_expr = expr if change_expr is None else (change_expr | expr)

    # Rows to expire: current historical rows that have changes
    changed_hist = joined_hi.filter(change_expr)

    # Map keys -> cutoff (incoming.effective_from - 1 second)
    cutoff = (F.col(f'i.{effective_from_col}') - F.expr("INTERVAL 1 seconds")).alias('cutoff')
    expire_keys = changed_hist.select(*[F.col(f'i.{k}').alias(k) for k in keys], cutoff).distinct()

    # Update historical: set effective_to and is_current=False for matching current rows
    hist_updated = hist.alias('h').join(expire_keys.alias('e'), keys, 'left')
    hist_updated = hist_updated.withColumn(
        effective_to_col,
        F.when((F.col('e.cutoff').isNotNull()) & (F.col(f'h.{is_current_col}') == True), F.col('e.cutoff')).otherwise(F.col(f'h.{effective_to_col}'))
    ).withColumn(
        is_current_col,
        F.when((F.col('e.cutoff').isNotNull()) & (F.col(f'h.{is_current_col}') == True), F.lit(False)).otherwise(F.col(f'h.{is_current_col}'))
    ).select('h.*')

    # New rows from incoming for changed keys
    new_from_changed = inc.alias('i').join(hist_current.alias('h'), keys, 'inner')
    # reuse change_expr but built against aliases i/h -> build equivalent expr
    change_expr_i_h = None
    for c in tracked_cols:
        expr = _col_ne(f'h.{c}', f'i.{c}')
        change_expr_i_h = expr if change_expr_i_h is None else (change_expr_i_h | expr)
    new_from_changed = new_from_changed.filter(change_expr_i_h)

    # New keys (completely new incoming rows)
    new_from_newkeys = inc.alias('i').join(hist_current.alias('h'), keys, 'left_anti')

    # Prepare inserted rows with SCD metadata
    def prepare_new(df):
        cols = [c for c in inc.columns]
        out = df.select(*[F.col(f'i.{c}').alias(c) for c in cols]) if 'i' in df.alias else df
        out = out.withColumn(effective_to_col, F.lit(None).cast(T.TimestampType()))
        out = out.withColumn(is_current_col, F.lit(True))
        return out

    new_rows = None
    if new_from_changed.rdd.isEmpty() and new_from_newkeys.rdd.isEmpty():
        new_rows = inc.limit(0)
    else:
        parts = []
        if not new_from_changed.rdd.isEmpty():
            parts.append(prepare_new(new_from_changed))
        if not new_from_newkeys.rdd.isEmpty():
            parts.append(prepare_new(new_from_newkeys))
        new_rows = parts[0]
        for p in parts[1:]:
            new_rows = new_rows.unionByName(p, allowMissingColumns=True)

    # Union updated history and new rows
    result = hist_updated.unionByName(new_rows, allowMissingColumns=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical', default='data/historical.csv')
    parser.add_argument('--incoming', default='data/incoming.csv')
    parser.add_argument('--out', default='data/merged_pyspark')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('scd_type2_example').getOrCreate()
    hist = spark.read.option('header', 'true').option('inferSchema', 'true').csv(args.historical)
    inc = spark.read.option('header', 'true').option('inferSchema', 'true').csv(args.incoming)

    # Ensure historical SCD columns exist and types are correct
    if 'is_current' not in hist.columns:
        hist = hist.withColumn('is_current', F.lit(True))
    hist = hist.withColumn('effective_from', F.to_timestamp('effective_from'))
    if 'effective_to' in hist.columns:
        hist = hist.withColumn('effective_to', F.to_timestamp('effective_to'))

    merged = scd_type2_merge(hist, inc, keys=['id'], tracked_cols=['name', 'city'])
    merged.coalesce(1).write.mode('overwrite').option('header', 'true').csv(args.out)
    print(f'Wrote merged output to {args.out}')
