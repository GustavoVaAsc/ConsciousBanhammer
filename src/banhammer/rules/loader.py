"""Reads the curated ruleset files: one YAML file per ruleset, validated on load (D28)."""

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from banhammer.rules.constants import MERGE_TAG, RULESET_FILE_SUFFIX
from banhammer.rules.corpus import RuleCorpus
from banhammer.rules.errors import RuleCorpusError, RulesetFormatError
from banhammer.rules.records import Ruleset, describe_validation_error


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects a mapping key written twice instead of keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            if key_node.tag == MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, Hashable) and key in seen:
                message = f"duplicate key {key!r}"
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, message, key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class RuleCorpusLoader:
    """Loads the ruleset files a curator wrote and checks them as one corpus.

    Every failure names the file, so a curator can fix the record that caused it.
    """

    def load_ruleset(self, path: Path) -> Ruleset:
        """Parse and validate one ruleset file; its stem must equal its `ruleset_id`.

        Raises:
            RulesetFormatError: The file cannot be read or parsed, is not a mapping, does not
                match the record format, or is named differently from its ruleset id.
        """
        document = self._read_document(path)
        try:
            ruleset = Ruleset.model_validate(document)
        except ValidationError as error:
            message = f"{path}: {describe_validation_error(error)}"
            raise RulesetFormatError(message) from error
        if path.stem != ruleset.ruleset_id:
            message = f"{path}: file name must be {ruleset.ruleset_id}{RULESET_FILE_SUFFIX}"
            raise RulesetFormatError(message)
        return ruleset

    def load_directory(self, directory: Path) -> RuleCorpus:
        """Load every `.yaml` file of `directory` into one corpus; other files are ignored.

        Raises:
            RulesetFormatError: The directory is missing or holds no ruleset file, or a file
                fails as in `load_ruleset`.
            RuleCorpusError: Two files share a ruleset id or a rule id.
        """
        if not directory.is_dir():
            message = f"{directory}: ruleset directory is missing"
            raise RulesetFormatError(message)
        paths = sorted(directory.glob(f"*{RULESET_FILE_SUFFIX}"))
        if not paths:
            message = f"{directory}: no {RULESET_FILE_SUFFIX} ruleset files"
            raise RulesetFormatError(message)
        rulesets = tuple(self.load_ruleset(path) for path in paths)
        try:
            return RuleCorpus(rulesets=rulesets)
        except ValidationError as error:
            message = f"{directory}: {describe_validation_error(error)}"
            raise RuleCorpusError(message) from error

    def _read_document(self, path: Path) -> object:
        try:
            text = path.read_text(encoding="utf-8")
            # https://pyyaml.org/wiki/PyYAMLDocumentation: _UniqueKeyLoader subclasses SafeLoader
            document = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            message = f"{path}: cannot read ruleset: {error}"
            raise RulesetFormatError(message) from error
        if not isinstance(document, dict):
            message = f"{path}: top level must be a mapping"
            raise RulesetFormatError(message)
        return document
