from __future__ import annotations


def discounted_total(prices: list[float], discount_percent: float) -> float:
    """Return the cart total after applying a percentage discount."""
    if any(price < 0 for price in prices):
        raise ValueError("prices must be non-negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be between 0 and 100")

    subtotal = sum(prices)
    discount = subtotal * discount_percent
    return round(subtotal - discount, 2)
