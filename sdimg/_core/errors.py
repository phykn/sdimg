def type_error(name: str, expected: str) -> TypeError:
    return TypeError(f"{name} must be {expected}.")


def value_error(message: str) -> ValueError:
    return ValueError(message)
