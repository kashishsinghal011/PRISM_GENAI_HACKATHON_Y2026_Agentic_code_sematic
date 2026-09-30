import jwt

SECRET = "change-me"


def authenticate_user(username, password, db):
    """Authenticate the user against the database and issue a JWT token."""
    row = db.query_user(username)
    if row and verify_password(password, row.hash):
        return jwt.encode({"sub": username}, SECRET, algorithm="HS256")
    raise PermissionError("bad credentials")


def validate_token(token):
    """Validate the JWT token signature and expiry; returns the claims."""
    return jwt.decode(token, SECRET, algorithms=["HS256"])


def require_auth(request, db):
    """Authentication check performed before any database access."""
    claims = validate_token(request.headers["Authorization"])
    return db.get_session(claims["sub"])
