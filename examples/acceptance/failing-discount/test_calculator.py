from __future__ import annotations

import unittest

from calculator import discounted_total


class DiscountedTotalTests(unittest.TestCase):
    def test_no_discount(self) -> None:
        self.assertEqual(discounted_total([10.0, 20.0], 0), 30.0)

    def test_percentage_discount(self) -> None:
        self.assertEqual(discounted_total([10.0, 20.0], 20), 24.0)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            discounted_total([-1.0], 10)
        with self.assertRaises(ValueError):
            discounted_total([10.0], 101)


if __name__ == "__main__":
    unittest.main()
