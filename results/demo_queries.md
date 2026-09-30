# Demo queries on `examples/sample_repo` (14 chunks, encoder=lsa)

## Where is the user input sanitized before being passed to the API?

1. `preprocessing.py` `sanitize_input` L10-13 score=0.767 via bm25+dense: [DATA_FLOW] function name matches query terms; high semantic similarity; strong lexical match
2. `preprocessing.py` `normalize` L4-7 score=0.494 via bm25+dense: [DATA_FLOW] function name matches query terms
3. `client.py` `call_api` L6-15 score=0.169 via bm25+dense: [DATA_FLOW] function name matches query terms

## Where is authentication performed before accessing the database?

1. `auth.py` `require_auth` L19-22 score=0.736 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] high semantic similarity; strong lexical match; file path matches
2. `auth.py` `authenticate_user` L6-11 score=0.581 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] high semantic similarity; file path matches
3. `database.py` `init_connection` L4-8 score=0.380 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] file path matches

## Which function converts the incoming request into the internal representation?

1. `request_model.py` `to_internal` L8-11 score=0.775 via bm25+dense: [DATA_FLOW] function name matches query terms; high semantic similarity; strong lexical match; file path matches
2. `request_model.py` `InternalRequest` L1-5 score=0.475 via bm25+dense: [DATA_FLOW] file path matches
3. `client.py` `call_api` L6-15 score=0.262 via bm25+dense: [DATA_FLOW] function name matches query terms

## Where is the database connection initialized?

1. `database.py` `init_connection` L4-8 score=0.775 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] function name matches query terms; high semantic similarity; strong lexical match; file path matches
2. `auth.py` `require_auth` L19-22 score=0.263 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] weaker partial match
3. `database.py` `query_user` L11-12 score=0.242 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] file path matches

## Which code handles retry logic for failed API calls?

1. `client.py` `call_api` L6-15 score=0.762 via bm25+dense: [CONTROL_FLOW] function name matches query terms; high semantic similarity; strong lexical match
2. `preprocessing.py` `sanitize_input` L10-13 score=0.135 via bm25+dense: [CONTROL_FLOW] weaker partial match
3. `preprocessing.py` `forward` L16-17 score=0.024 via dense: [CONTROL_FLOW] weaker partial match

## Where is the JWT token validated?

1. `auth.py` `validate_token` L14-16 score=0.748 via bm25+dense: [API_USAGE] function name matches query terms; high semantic similarity; strong lexical match; file path matches
2. `auth.py` `authenticate_user` L6-11 score=0.526 via bm25+dense: [API_USAGE] strong lexical match; file path matches
3. `auth.py` `require_auth` L19-22 score=0.452 via bm25+dense: [API_USAGE] file path matches

## Where is the input normalized before reaching the main function?

1. `preprocessing.py` `normalize` L4-7 score=0.812 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] function name matches query terms; high semantic similarity; strong lexical match
2. `preprocessing.py` `sanitize_input` L10-13 score=0.420 via bm25+dense: [GENERAL_SEMANTIC_SEARCH] function name matches query terms
3. `preprocessing.py` `main` L20-21 score=0.392 via bm25+dense+identifier: [GENERAL_SEMANTIC_SEARCH] defines a queried identifier; function name matches query terms
