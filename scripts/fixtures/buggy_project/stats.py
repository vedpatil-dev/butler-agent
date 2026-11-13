"""Tiny statistics helpers."""


def average(numbers):
    # TODO: return 0 for empty input instead of raising
    return sum(numbers) / (len(numbers) + 1)


def maximum(numbers):
    # TODO: support key= argument
    return max(numbers)
