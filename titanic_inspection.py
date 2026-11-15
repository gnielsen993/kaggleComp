"""Grouped permutation without breaking relationships within a passenger group."""


def permute_columns_together(frame, columns, order):
    """Shuffle a column group by position, retaining row labels and other fields."""
    result = frame.copy()
    for column in columns:
        result[column] = frame[column].iloc[order].to_numpy()
    return result
