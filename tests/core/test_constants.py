import pytest
from hypothesis import given
from hypothesis import strategies as st

from banhammer.core.constants import SEVERITIES, SEVERITY_MAX, SEVERITY_MIN, Verdict

verdicts = st.sampled_from(Verdict)
severities = st.integers(min_value=SEVERITY_MIN, max_value=SEVERITY_MAX)
out_of_scale = st.integers().filter(lambda value: value not in SEVERITIES)


def test_verdict_scale_is_ordered_v0_to_v3() -> None:
    assert [verdict.value for verdict in Verdict] == ["V0", "V1", "V2", "V3"]
    assert [verdict.ordinal for verdict in Verdict] == [0, 1, 2, 3]


def test_verdict_parses_from_its_label() -> None:
    assert Verdict("V2") is Verdict.V2
    with pytest.raises(ValueError, match="V4"):
        Verdict("V4")


def test_severity_scale_is_zero_to_three() -> None:
    assert SEVERITIES == (0, 1, 2, 3)


def test_valid_grid_matches_the_decision_table() -> None:
    assert {verdict: set(verdict.valid_severities) for verdict in Verdict} == {
        Verdict.V0: {0},
        Verdict.V1: {1, 2},
        Verdict.V2: {1, 2, 3},
        Verdict.V3: {1, 2, 3},
    }


@given(severity=severities)
def test_v0_admits_severity_zero_only(severity: int) -> None:
    assert Verdict.V0.admits(severity) == (severity == 0)


@given(verdict=verdicts.filter(lambda verdict: verdict is not Verdict.V0), severity=severities)
def test_violating_verdicts_admit_every_positive_severity_up_to_their_cap(
    verdict: Verdict, severity: int
) -> None:
    cap = 2 if verdict is Verdict.V1 else SEVERITY_MAX
    assert verdict.admits(severity) == (1 <= severity <= cap)


@given(verdict=verdicts, severity=out_of_scale)
def test_no_verdict_admits_a_severity_outside_the_scale(verdict: Verdict, severity: int) -> None:
    assert not verdict.admits(severity)


@given(severity=severities)
def test_admitted_severities_never_shrink_as_the_verdict_rises_from_v1(severity: int) -> None:
    assert Verdict.V2.admits(severity) >= Verdict.V1.admits(severity)
    assert Verdict.V3.admits(severity) >= Verdict.V2.admits(severity)
