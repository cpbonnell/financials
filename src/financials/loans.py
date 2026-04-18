from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

import polars as pl

Number = int | float | str | Decimal

_CENT = Decimal("0.01")


def _to_decimal(value: Number) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class AmortizedLoan:
    """A fixed-rate, fully-amortized loan (e.g. a mortgage or car loan).

    Holds the key summary figures a borrower cares about along with the full
    payment-by-payment schedule. Construct with :meth:`from_terms` rather than
    calling the dataclass directly.

    Attributes:
        principal: Amount borrowed.
        annual_rate: Annual percentage rate expressed as a fraction (0.06 = 6%).
        num_payments: Total number of monthly payments in the loan term.
        payment: Fixed monthly payment, rounded to cents. The final payment may
            differ by a few cents to close the balance exactly at zero.
        total_interest: Sum of all interest paid over the life of the loan.
        total_cost: ``principal + total_interest`` — everything paid to the lender.
        schedule: Polars DataFrame with one row per payment and columns
            ``payment_number``, ``payment``, ``principal``, ``interest``, ``balance``.
    """

    principal: Decimal
    annual_rate: Decimal
    num_payments: int
    payment: Decimal
    total_interest: Decimal
    total_cost: Decimal
    schedule: pl.DataFrame = field(repr=False)

    @classmethod
    def from_terms(
        cls,
        principal: Number,
        annual_rate: Number,
        num_payments: int,
    ) -> "AmortizedLoan":
        """Build an :class:`AmortizedLoan` from the standard loan terms.

        Args:
            principal: Amount borrowed. Accepts ``int``, ``float``, ``str``, or
                ``Decimal``; coerced to ``Decimal`` internally.
            annual_rate: APR as a fraction (``0.06`` for 6%). Accepts the same
                numeric types as ``principal``.
            num_payments: Number of monthly payments (e.g. ``360`` for a 30-year
                mortgage).

        Returns:
            A populated :class:`AmortizedLoan` whose schedule's final balance is
            exactly zero; the last payment absorbs any rounding residual.
        """
        p = _to_decimal(principal)
        apr = _to_decimal(annual_rate)
        n = int(num_payments)

        r = apr / 12

        if r == 0:
            payment = (p / n).quantize(_CENT, rounding=ROUND_HALF_UP)
        else:
            growth = (1 + r) ** n
            unrounded = p * r * growth / (growth - 1)
            payment = unrounded.quantize(_CENT, rounding=ROUND_HALF_UP)

        payment_numbers: list[int] = []
        payments: list[Decimal] = []
        principals: list[Decimal] = []
        interests: list[Decimal] = []
        balances: list[Decimal] = []

        balance = p
        for i in range(1, n + 1):
            interest_i = (balance * r).quantize(_CENT, rounding=ROUND_HALF_UP)
            if i == n:
                principal_i = balance
                payment_i = principal_i + interest_i
            else:
                principal_i = payment - interest_i
                payment_i = payment
            balance = balance - principal_i

            payment_numbers.append(i)
            payments.append(payment_i)
            principals.append(principal_i)
            interests.append(interest_i)
            balances.append(balance)

        money_dtype = pl.Decimal(precision=20, scale=2)
        schedule = pl.DataFrame(
            {
                "payment_number": payment_numbers,
                "payment": payments,
                "principal": principals,
                "interest": interests,
                "balance": balances,
            },
            schema={
                "payment_number": pl.UInt32,
                "payment": money_dtype,
                "principal": money_dtype,
                "interest": money_dtype,
                "balance": money_dtype,
            },
        )

        total_interest = sum(interests, Decimal("0"))
        total_cost = p + total_interest

        return cls(
            principal=p,
            annual_rate=apr,
            num_payments=n,
            payment=payment,
            total_interest=total_interest,
            total_cost=total_cost,
            schedule=schedule,
        )
