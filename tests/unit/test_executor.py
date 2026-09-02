"""Unit tests for the AST-validated expression executor.

The executor is the safety-critical component: it evaluates arithmetic
expressions emitted by the LLM without ever using eval() or exec().
All edge cases and attack vectors must be covered before implementation.
"""

import pytest

from financial_qa.pipeline.executor import ExecutionError, Executor


class TestSafeArithmetic:
    """Happy-path arithmetic — these must all return correct floats."""

    def test_addition(self) -> None:
        assert Executor().run("1 + 2") == pytest.approx(3.0)

    def test_subtraction(self) -> None:
        assert Executor().run("10 - 4.5") == pytest.approx(5.5)

    def test_multiplication(self) -> None:
        assert Executor().run("3 * 4") == pytest.approx(12.0)

    def test_division(self) -> None:
        assert Executor().run("14.5 / 2") == pytest.approx(7.25)

    def test_percentage_change(self) -> None:
        assert Executor().run("(14.5 - 12.0) / 12.0 * 100") == pytest.approx(20.833, rel=1e-3)

    def test_nested_parentheses(self) -> None:
        assert Executor().run("(1 + 2) * (3 + 4)") == pytest.approx(21.0)

    def test_unary_negation(self) -> None:
        assert Executor().run("-5 + 10") == pytest.approx(5.0)

    def test_float_literals(self) -> None:
        assert Executor().run("0.1 + 0.2") == pytest.approx(0.3, abs=1e-9)


class TestDivisionByZero:
    def test_raises_on_division_by_zero(self) -> None:
        with pytest.raises(ExecutionError, match="division by zero"):
            Executor().run("1 / 0")

    def test_raises_on_modulo_by_zero(self) -> None:
        with pytest.raises(ExecutionError):
            Executor().run("5 % 0")


class TestForbiddenNodes:
    """Expressions containing anything beyond arithmetic must be rejected."""

    def test_rejects_function_call(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("abs(-1)")

    def test_rejects_import(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("__import__('os')")

    def test_rejects_string_literal(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("'hello'")

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("x.y")

    def test_rejects_comparison(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("1 == 1")

    def test_rejects_lambda(self) -> None:
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("lambda x: x")

    def test_rejects_name_lookup(self) -> None:
        """Variable names are not allowed — only numeric literals."""
        with pytest.raises(ExecutionError, match="forbidden"):
            Executor().run("revenue + costs")

    def test_rejects_multiline(self) -> None:
        with pytest.raises(ExecutionError):
            Executor().run("1 + 1\n2 + 2")


class TestMalformedInput:
    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ExecutionError):
            Executor().run("")

    def test_rejects_unparseable_expression(self) -> None:
        with pytest.raises(ExecutionError):
            Executor().run("1 +* 2")

    def test_returns_float_not_int(self) -> None:
        result = Executor().run("4 / 2")
        assert isinstance(result, float)
