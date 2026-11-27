import time

import pytest

from banhammer.messages.text_cleaning import clean_text, dedup_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see https://example.com/a?b=1 now", "see now"),
        ("see HTTP://EXAMPLE.COM now and www.example.org/x too", "see now and too"),
        ("hi <@123456> and <@!42> in <#99> with <@&7>", "hi and in with"),
        ("hi @alice_1 and @bob", "hi and"),
        ("mail me@example.com", "mail me@example.com"),
        ("great 😂👍🏽 job ❤️", "great job"),
        ("flags 🇪🇸 and 👨‍👩‍👧 family", "flags and family"),
        ("wow <:pog:123456> and <a:dance:99> and :smile: ok", "wow and and ok"),
        ("run ```print(1)\nx = 2``` and `inline` ok", "run and ok"),
        ("¿Qué tal, señor? Ça va", "¿Qué tal, señor? Ça va"),
        ("  many   spaces\n\tand\nlines ", "many spaces and lines"),
        ("ｆｕｌｌｗｉｄｔｈ ① ﬁ", "fullwidth 1 fi"),
        ("^caret and ratio 3:30 stay", "^caret and ratio 3:30 stay"),
        ("seen <discord-link> and <@user> <@role> <#channel> ok", "seen and ok"),
        ("", ""),
        ("https://example.com 😂 <@1>", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent() -> None:
    raw = "hey <@1> https://x.io 😂 `c` ¡Hola!  qué  tal"
    assert clean_text(clean_text(raw)) == clean_text(raw)


def test_dedup_key_ignores_case_noise_and_whitespace() -> None:
    assert dedup_key("Hola   MUNDO https://a.io 😂") == dedup_key("hola mundo <@3>")


def test_dedup_key_keeps_accents_and_distinguishes_words() -> None:
    assert dedup_key("está") != dedup_key("esta")
    assert dedup_key("ß") == "ss"


@pytest.mark.parametrize(
    "hostile",
    ["``` " * 40_000, "```a" * 40_000, "`" * 200_000, "``` `" * 40_000, "```" + "`x" * 100_000],
)
def test_hostile_runs_of_backticks_are_cleaned_in_linear_time(hostile: str) -> None:
    started = time.perf_counter()
    clean_text(hostile)
    assert time.perf_counter() - started < 2.0


def test_a_fenced_block_with_inner_backticks_and_newlines_is_removed_whole() -> None:
    assert clean_text("a ```x = `y`\nz = 1``` b") == "a b"
