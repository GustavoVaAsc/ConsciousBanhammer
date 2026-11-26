from pathlib import Path

from hygiene_checker import HygieneChecker, RepositoryScanner

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_src_and_tests_follow_the_hygiene_rules() -> None:
    files = RepositoryScanner(REPOSITORY_ROOT).files()
    checker = HygieneChecker.for_files(files)
    violations = [violation for file in files for violation in checker.check(file)]
    assert not violations, "\n" + "\n".join(str(violation) for violation in violations)
