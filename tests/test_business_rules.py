"""Тесты бизнес-правил."""

from maternal_risk.business.rules import split_by_age


def test_split_by_age(sample_patients_df) -> None:
    young, older = split_by_age(sample_patients_df)
    assert (young["Age"] <= 42).all()
    assert (older["Age"] > 42).all()


def test_age_filter(sample_patients_df) -> None:
    filtered = sample_patients_df[sample_patients_df["Age"] >= 18]
    assert (filtered["Age"] >= 18).all()
    assert len(filtered) == len(sample_patients_df) - 1
