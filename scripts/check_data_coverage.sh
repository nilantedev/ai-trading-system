#!/bin/bash
#
# Check QuestDB data coverage
#

QUESTDB_URL="http://localhost:9000/exec"

# Query market data count
MARKET=$(curl -s "${QUESTDB_URL}?query=SELECT+COUNT(*)+FROM+market_data" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['dataset'][0][0])" 2>/dev/null)

# Query options data count
OPTIONS=$(curl -s "${QUESTDB_URL}?query=SELECT+COUNT(*)+FROM+options_data" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['dataset'][0][0])" 2>/dev/null)

# Query news items count
NEWS=$(curl -s "${QUESTDB_URL}?query=SELECT+COUNT(*)+FROM+news_items" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['dataset'][0][0])" 2>/dev/null)

# Query social signals count
SOCIAL=$(curl -s "${QUESTDB_URL}?query=SELECT+COUNT(*)+FROM+social_signals" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['dataset'][0][0])" 2>/dev/null)

# Query unique symbols
SYMBOLS=$(curl -s "${QUESTDB_URL}?query=SELECT+COUNT(DISTINCT+symbol)+FROM+market_data" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['dataset'][0][0])" 2>/dev/null)

# Format output
if [ -n "$MARKET" ]; then
    echo "$MARKET|$OPTIONS|$NEWS|$SOCIAL|$SYMBOLS"
fi
