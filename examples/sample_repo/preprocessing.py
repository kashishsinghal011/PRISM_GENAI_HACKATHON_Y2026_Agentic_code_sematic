"""Input preprocessing before the main entry point."""


def normalize(text):
    """Normalize raw input: trim, lowercase, collapse whitespace, then forward to main."""
    cleaned = " ".join(text.strip().lower().split())
    return forward(cleaned)


def sanitize_input(user_input):
    """Sanitize user input (strip control characters, escape HTML) before it is passed to the API."""
    safe = escape_html(remove_control_chars(user_input))
    return call_api(safe)


def forward(value):
    return main(value)


def main(value):
    print(value)
