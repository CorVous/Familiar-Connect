"""Tests for SillyTavern macro substitution."""

from familiar_connect.macros import MacroContext, substitute


class TestCommentMacro:
    def test_comment_is_removed(self) -> None:
        result = substitute("Hello {{// this is a comment }}world", MacroContext())
        assert result == "Hello world"

    def test_multiword_comment_removed(self) -> None:
        result = substitute("{{// remove me }}", MacroContext())
        assert not result

    def test_comment_with_spaces(self) -> None:
        result = substitute("a{{// comment with lots of words here }}b", MacroContext())
        assert result == "ab"


class TestTrimMacro:
    def test_trim_removes_macro_and_strips_whitespace(self) -> None:
        result = substitute("  hello world  {{trim}}", MacroContext())
        assert result == "hello world"

    def test_trim_with_leading_whitespace(self) -> None:
        result = substitute("  {{trim}}  text  ", MacroContext())
        assert result == "text"

    def test_trim_only(self) -> None:
        result = substitute("{{trim}}", MacroContext())
        assert not result


class TestCharMacro:
    def test_char_replaced_with_name(self) -> None:
        ctx = MacroContext(char="Sapphire")
        result = substitute("My name is {{char}}.", ctx)
        assert result == "My name is Sapphire."

    def test_char_default_empty(self) -> None:
        result = substitute("{{char}}", MacroContext())
        assert not result


class TestUserMacro:
    def test_user_replaced_with_name(self) -> None:
        ctx = MacroContext(user="Alice")
        result = substitute("Hello, {{user}}!", ctx)
        assert result == "Hello, Alice!"

    def test_user_default(self) -> None:
        result = substitute("{{user}}", MacroContext())
        assert result == "User"


class TestScenarioMacro:
    def test_scenario_substituted(self) -> None:
        ctx = MacroContext(scenario="A rainy afternoon in a coffee shop.")
        result = substitute("Scene: {{scenario}}", ctx)
        assert result == "Scene: A rainy afternoon in a coffee shop."

    def test_scenario_default_empty(self) -> None:
        result = substitute("{{scenario}}", MacroContext())
        assert not result


class TestPersonalityMacro:
    def test_personality_substituted(self) -> None:
        ctx = MacroContext(personality="Cheerful and curious.")
        result = substitute("Traits: {{personality}}", ctx)
        assert result == "Traits: Cheerful and curious."

    def test_personality_default_empty(self) -> None:
        result = substitute("{{personality}}", MacroContext())
        assert not result


class TestDescriptionMacro:
    def test_description_substituted(self) -> None:
        ctx = MacroContext(description="A blue-haired mage.")
        result = substitute("Who: {{description}}", ctx)
        assert result == "Who: A blue-haired mage."

    def test_description_default_empty(self) -> None:
        result = substitute("{{description}}", MacroContext())
        assert not result


class TestUnknownMacro:
    def test_unknown_macro_passes_through(self) -> None:
        result = substitute("{{getvar::guidelines}}", MacroContext())
        assert result == "{{getvar::guidelines}}"

    def test_unknown_macro_with_text(self) -> None:
        result = substitute("before {{randomfuture}} after", MacroContext())
        assert result == "before {{randomfuture}} after"


class TestCombined:
    def test_multiple_macros_in_one_string(self) -> None:
        ctx = MacroContext(char="Sapphire", user="Alice", personality="Bright")
        text = (
            "{{// header }}{{char}} speaks to {{user}}. Traits: {{personality}}{{trim}}"
        )
        result = substitute(text, ctx)
        assert result == "Sapphire speaks to Alice. Traits: Bright"

    def test_comment_then_trim(self) -> None:
        result = substitute("  {{// ignore }}  hello  {{trim}}", MacroContext())
        assert result == "hello"
