"""Exceptions shared by every subpackage."""


class BanhammerError(Exception):
    """Base class of every exception this project raises on purpose."""


class SettingsError(BanhammerError):
    """The params file is missing, unreadable, or does not match its settings model."""


class SettingsDefinitionError(BanhammerError):
    """A settings model gives a field a default, so a key missing from `params.yaml` would pass."""


class UnknownKeyError(BanhammerError):
    """A key was looked up in a union-find that has never seen it."""


class InvalidKeyError(BanhammerError):
    """A union-find key is not a plain `str` or `int`, or not of the kind the instance holds."""


class ModelSnapshotError(BanhammerError):
    """A model snapshot could not be downloaded at its pinned revision."""
