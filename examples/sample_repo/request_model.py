class InternalRequest:
    """Internal representation of an incoming request."""

    def __init__(self, user, action, params):
        self.user, self.action, self.params = user, action, params


def to_internal(raw_request):
    """Convert the incoming HTTP request into the internal representation."""
    body = raw_request.json()
    return InternalRequest(body["user"], body["action"], body.get("params", {}))
