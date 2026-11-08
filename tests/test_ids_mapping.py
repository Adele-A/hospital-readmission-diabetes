"""Tests for parsing the stacked IDS_mapping.csv file."""

from pathlib import Path

from src.data.load import parse_ids_mapping

FIXTURE = Path(__file__).parent / "fixtures" / "ids_mapping_sample.csv"


def test_three_mapping_blocks_parsed():
    df = parse_ids_mapping(FIXTURE)
    assert set(df["mapping_type"].unique()) == {
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    }


def test_ids_are_integers_and_descriptions_stripped():
    df = parse_ids_mapping(FIXTURE)
    assert df["id"].dtype.kind in ("i", "u")
    # Leading spaces in the source (e.g. " Physician Referral") are stripped.
    src_val = df.loc[
        (df["mapping_type"] == "admission_source_id") & (df["id"] == 1), "description"
    ].iloc[0]
    assert src_val == "Physician Referral"


def test_death_hospice_codes_present_with_expected_descriptions():
    df = parse_ids_mapping(FIXTURE)
    disch = df[df["mapping_type"] == "discharge_disposition_id"].set_index("id")["description"]
    assert disch[11] == "Expired"
    assert disch[13] == "Hospice / home"
    assert disch[14] == "Hospice / medical facility"
    assert "hospice" in disch[19].lower()


def test_blank_separator_lines_are_not_rows():
    df = parse_ids_mapping(FIXTURE)
    # No empty descriptions and no row with id coming from a "," separator line.
    assert (df["description"].str.len() > 0).all()
