import re
from pathlib import Path

import pytest

DRIVER_VERSION_FILE = Path("/proc/driver/nvidia/version")
# https://github.com/NVIDIA/product-security/tree/main/2026/5861: CVE-2026-47501 fix per branch
FIXED_RELEASES = {
    580: (580, 178, 4),
    595: (595, 91, 7),
    610: (610, 57, 4),
    615: (615, 71, 9),
}
DRIVER_VERSION = re.compile(r"NVRM version:.*?\s(\d+(?:\.\d+){1,2})\s")


def parse_driver_version(text: str) -> tuple[int, ...]:
    match = DRIVER_VERSION.search(text)
    assert match, f"no driver version in: {text!r}"
    return tuple(int(part) for part in match.group(1).split("."))


def is_vulnerable(version: tuple[int, ...]) -> bool:
    branch = version[0]
    if branch in FIXED_RELEASES:
        return version < FIXED_RELEASES[branch]
    return branch < max(FIXED_RELEASES)


def test_parses_the_version_from_the_nvrm_line() -> None:
    line = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.159.03  Release Build\n"
    assert parse_driver_version(line) == (580, 159, 3)


@pytest.mark.parametrize(
    ("version", "vulnerable"),
    [
        ((580, 159, 3), True),
        ((580, 178, 3), True),
        ((580, 178, 4), False),
        ((580, 200, 1), False),
        ((595, 91, 6), True),
        ((610, 57, 4), False),
        ((570, 999, 0), True),
        ((590, 1, 0), True),
        ((620, 1, 0), False),
    ],
)
def test_a_release_below_its_branch_fix_or_on_an_unlisted_older_branch_is_vulnerable(
    version: tuple[int, ...], vulnerable: bool
) -> None:
    assert is_vulnerable(version) is vulnerable


@pytest.mark.gpu
def test_host_driver_is_not_affected_by_cve_2026_47501() -> None:
    if not DRIVER_VERSION_FILE.exists():
        pytest.skip(f"{DRIVER_VERSION_FILE} does not exist")
    version = parse_driver_version(DRIVER_VERSION_FILE.read_text(encoding="utf-8"))
    assert not is_vulnerable(version), (
        f"driver {'.'.join(map(str, version))} is affected by CVE-2026-47501 "
        f"(fixed releases per branch: {FIXED_RELEASES}): upgrade before running GPU containers"
    )
