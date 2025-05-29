import pandas as pd
import logging
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# seann: changed type so it expicity allow None
def prepare_participants_file(
    tsv_path: str | Path, output_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Prepare participants.tsv file for WONKYCONN analysis:
    1. Validates required columns exist
    2. Ensures proper numeric encoding for gender (0=M, 1=F)
    3. Ensures age is float
    4. Validates subject IDs match BIDS format

    Parameters
    ----------
    tsv_path : str or Path
        Path to input participants.tsv file
    output_path : str or Path, optional
        Path to save cleaned participants.tsv. If None, overwrites input file

    Returns
    -------
    pd.DataFrame
        Cleaned participants dataframe
    """
    tsv_path = Path(tsv_path)
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = tsv_path

    logging.info(f"Reading participants file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    # Verify required columns
    required_cols = ["participant_id", "gender", "age"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean participant IDs
    if not df["participant_id"].str.startswith("sub-").all():
        logging.info("Adding 'sub-' prefix to participant IDs")
        df["participant_id"] = df["participant_id"].apply(
            lambda x: f"sub-{x}" if not str(x).startswith("sub-") else x
        )

    # Convert gender to numeric (M=0, F=1)
    df = df.copy()  # Make copy to avoid modifying view
    if df["gender"].dtype == object:  # If gender is string
        logging.info("Converting gender from M/F to 0/1")
        df.loc[df["gender"].str.upper() == "M", "gender"] = 0
        df.loc[df["gender"].str.upper() == "F", "gender"] = 1

    # Convert datatypes
    logging.info("Converting datatypes")
    df["gender"] = df["gender"].astype(float)
    df["age"] = df["age"].astype(float)

    # Validate data
    if len(df["gender"].unique()) < 2:
        raise ValueError("Need both gender categories (0 and 1) for analysis")

    if not df["gender"].isin([0, 1]).all():
        raise ValueError("Gender values must be 0 or 1")

    # Save cleaned file
    logging.info(f"Saving cleaned participants file to {output_path}")
    df.to_csv(output_path, sep="\t", index=False)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare participants.tsv file for WONKYCONN"
    )
    parser.add_argument(
        "input_tsv", help="Path to input participants.tsv file"
    )
    parser.add_argument(
        "--output", help="Path to save cleaned file (optional)"
    )
    args = parser.parse_args()

    prepare_participants_file(args.input_tsv, args.output)
