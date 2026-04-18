from decimal import Decimal

import polars as pl
import pytest

from financials.loans import AmortizedLoan


def test_known_30_year_mortgage():
    loan = AmortizedLoan.from_terms(200_000, 0.06, 360)
    assert loan.payment == Decimal("1199.10")
    assert abs(loan.total_interest - Decimal("231676.38")) <= Decimal("2.00")


def test_schedule_shape_and_closeout():
    loan = AmortizedLoan.from_terms(200_000, 0.06, 360)
    assert loan.schedule.shape == (360, 5)
    assert loan.schedule["balance"][-1] == Decimal("0")
    assert list(loan.schedule.columns) == [
        "payment_number",
        "payment",
        "principal",
        "interest",
        "balance",
    ]


def test_totals_consistency():
    loan = AmortizedLoan.from_terms(250_000, 0.055, 180)
    interest_sum = loan.schedule["interest"].sum()
    payment_sum = loan.schedule["payment"].sum()
    assert loan.total_interest == interest_sum
    assert loan.total_cost == loan.principal + loan.total_interest
    assert abs(loan.total_cost - payment_sum) <= Decimal("0.02")


def test_zero_interest_loan():
    loan = AmortizedLoan.from_terms(12_000, 0, 24)
    assert loan.payment == Decimal("500.00")
    assert loan.total_interest == Decimal("0")
    assert loan.total_cost == Decimal("12000")
    assert (loan.schedule["interest"] == Decimal("0")).all()
    assert loan.schedule["balance"][-1] == Decimal("0")


def test_from_budget_roundtrips_with_from_terms():
    loan = AmortizedLoan.from_budget(1500, 0.06, 360)
    assert abs(loan.payment - Decimal("1500.00")) <= Decimal("0.01")
    assert loan.num_payments == 360
    assert loan.annual_rate == Decimal("0.06")


def test_from_budget_zero_interest():
    loan = AmortizedLoan.from_budget(500, 0, 24)
    assert loan.principal == Decimal("12000.00")
    assert loan.payment == Decimal("500.00")
    assert loan.total_interest == Decimal("0")


@pytest.mark.parametrize(
    "principal, rate",
    [
        (200_000, 0.06),
        ("200000", "0.06"),
        (Decimal("200000"), Decimal("0.06")),
        (200_000.0, 0.06),
    ],
)
def test_input_flexibility(principal, rate):
    loan = AmortizedLoan.from_terms(principal, rate, 360)
    assert loan.payment == Decimal("1199.10")
    assert loan.num_payments == 360
