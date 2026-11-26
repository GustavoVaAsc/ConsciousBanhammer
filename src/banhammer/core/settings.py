"""Settings models and the reader that fills them from `params.yaml` (D4).

`params.yaml` is the only source of tunables, so the models have no defaults and the reader
never consults environment variables. Each subpackage owns one top-level section of the file.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from banhammer.core.constants import SEED_LIMIT
from banhammer.core.errors import SettingsDefinitionError, SettingsError


class SettingsModel(BaseModel):
    """Base of every settings model: frozen, no unknown keys, and no defaults on its fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    # https://docs.pydantic.dev/latest/concepts/models/: pydantic fixes the hook signature
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Reject a subclass that gives any field a default, at class creation.

        Raises:
            SettingsDefinitionError: A field has a default or a default factory.
        """
        super().__pydantic_init_subclass__(**kwargs)
        defaulted = sorted(
            name for name, field in cls.model_fields.items() if not field.is_required()
        )
        if defaulted:
            message = f"{cls.__name__}: fields must not have defaults: {', '.join(defaulted)}"
            raise SettingsDefinitionError(message)


class CoreSettings(SettingsModel):
    """The `core` section of `params.yaml`: the seed and the root directories of the pipeline."""

    seed: int = Field(ge=0, lt=SEED_LIMIT)
    data_dir: Path
    models_dir: Path


class ParamsFile:
    """A `params.yaml` file from which settings sections are read and validated."""

    def __init__(self, path: Path) -> None:
        """Remember the location of the file; it is read on each `section` call."""
        self._path = path

    def section[SettingsT: SettingsModel](self, name: str, model: type[SettingsT]) -> SettingsT:
        """Validate the top-level section `name` of the file against `model`.

        Raises:
            SettingsError: The file cannot be read or parsed, the section is absent, or its
                content does not match `model`.
        """
        raw = self._read_section(name)
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            message = f"{self._path}: section '{name}' is invalid: {error}"
            raise SettingsError(message) from error

    def _read_section(self, name: str) -> object:
        document = self._read_document()
        if name not in document:
            message = f"{self._path}: section '{name}' is missing"
            raise SettingsError(message)
        return document[name]

    def _read_document(self) -> dict[str, object]:
        try:
            document = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            message = f"{self._path}: cannot read params: {error}"
            raise SettingsError(message) from error
        if not isinstance(document, dict):
            message = f"{self._path}: top level must be a mapping"
            raise SettingsError(message)
        return document
