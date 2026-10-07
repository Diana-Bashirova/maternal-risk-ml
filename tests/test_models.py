"""Тесты модели."""

from maternal_risk.data.features import encode_target, engineer_features


def test_engineer_features(sample_patients_df) -> None:
    result = engineer_features(sample_patients_df)
    assert "PulsePressure" in result.columns
    assert (result["PulsePressure"] == result["SystolicBP"] - result["DiastolicBP"]).all()


def test_encode_target(sample_patients_df) -> None:
    df, class_names = encode_target(sample_patients_df)
    assert set(df["RiskLevel"]).issubset({0, 1, 2})
    assert class_names == {0: "low risk", 1: "mid risk", 2: "high risk"}
