from banhammer.core.errors import BanhammerError, SettingsDefinitionError, SettingsError


def test_settings_error_is_a_banhammer_error() -> None:
    assert issubclass(SettingsError, BanhammerError)


def test_banhammer_error_is_an_exception_not_a_base_exception() -> None:
    assert issubclass(BanhammerError, Exception)


def test_settings_definition_error_is_a_banhammer_error() -> None:
    assert issubclass(SettingsDefinitionError, BanhammerError)
