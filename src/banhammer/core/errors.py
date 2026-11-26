"""Exceptions shared by every subpackage."""


class BanhammerError(Exception):
    """Base class of every exception this project raises on purpose."""


class SettingsError(BanhammerError):
    """The params file is missing, unreadable, or does not match its settings model."""


class SettingsDefinitionError(BanhammerError):
    """A settings model gives a field a default, so a key missing from `params.yaml` would pass."""
