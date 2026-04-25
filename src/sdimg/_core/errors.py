def type_error(name: str, expected: str) -> TypeError:
    return TypeError(f"{name} must be {expected}.")
