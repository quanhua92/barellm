def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
