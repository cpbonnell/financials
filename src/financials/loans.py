from dataclasses import dataclass, field
from decimal import Decimal

import polars as pl


@dataclass(frozen=True)
class AmortizedLoan:
    principal: Decimal
    annual_rate: Decimal
    num_payments: int
    payment: Decimal
    total_interest: Decimal
    total_cost: Decimal
    schedule: pl.DataFrame = field(repr=False)
