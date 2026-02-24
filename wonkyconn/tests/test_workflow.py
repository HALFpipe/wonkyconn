from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from wonkyconn.workflow import load_data_frame


def test_load_data_frame(tmp_path: Path) -> None:
    # Create a sample phenotypes file
    phenotypes_path = tmp_path / "phenotypes.tsv"
    phenotypes_path.write_text("participant_id\tage\tgender\nsub-01\t25\tM\nsub-02\t30\tF\nsub-03\t35\tM\n")

    # Load the data frame
    data_frame = load_data_frame(Namespace(phenotypes=str(phenotypes_path)))

    # Check that the data frame has the expected shape and columns
    row_count, _ = data_frame.shape
    assert row_count == 3
    assert list(data_frame.reset_index().columns) == ["participant_id", "age", "gender"]

    full_data_frame = data_frame.reset_index()

    # Check that we throw an error for missing columns
    for missing_column in full_data_frame.columns:
        full_data_frame.drop(columns=missing_column).to_csv(phenotypes_path, sep="\t", index=False)
        with pytest.raises(ValueError):
            load_data_frame(Namespace(phenotypes=str(phenotypes_path)))

    # Check that we throw an error for duplicate participant_id entries
    data_frame = pd.concat([full_data_frame, full_data_frame.iloc[0:1]])  # Add a duplicate row
    data_frame.to_csv(phenotypes_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate participant_id"):
        load_data_frame(Namespace(phenotypes=str(phenotypes_path)))
