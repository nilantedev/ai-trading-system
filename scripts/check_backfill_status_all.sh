#!/bin/bash

# Backfill Status Check for All Data Types
# Checks watchlist backfill progress for equity, options, news, social, calendar

echo "═══════════════════════════════════════════════════════════════"
echo "  WATCHLIST BACKFILL STATUS - ALL DATA TYPES"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Get watchlist size
echo "📋 Watchlist Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
watchlist_count=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | xargs)
echo "Total symbols in watchlist: ${watchlist_count:-0}"
echo ""

# Check PostgreSQL backfill progress
echo "📊 EQUITY BACKFILL (PostgreSQL):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker exec -i trading-postgres psql -U trading_user -d trading_db -t << 'EOF'
SELECT 
    COUNT(*) as total_symbols,
    COUNT(*) FILTER (WHERE last_date IS NOT NULL) as symbols_with_data,
    COUNT(*) FILTER (WHERE last_date >= CURRENT_DATE - INTERVAL '7 days') as recent_updates,
    ROUND(100.0 * COUNT(*) FILTER (WHERE last_date IS NOT NULL) / COUNT(*), 1) as completion_pct
FROM historical_backfill_progress;
EOF

echo ""
echo "Sample symbols with most recent data:"
docker exec -i trading-postgres psql -U trading_user -d trading_db -c "SELECT symbol, last_date, updated_at FROM historical_backfill_progress WHERE last_date IS NOT NULL ORDER BY last_date DESC LIMIT 10;" 2>/dev/null

# Check QuestDB equity data
echo ""
echo "📊 EQUITY DATA IN QUESTDB:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

daily_bars_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20daily_bars" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
daily_bars_symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20count_distinct%28symbol%29%20FROM%20daily_bars" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
market_data_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20market_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "%-40s %'15d rows\n" "Daily bars (daily_bars)" "${daily_bars_count:-0}"
printf "%-40s %'15d symbols\n" "Unique symbols with daily bars" "${daily_bars_symbols:-0}"
printf "%-40s %'15d rows\n" "Intraday market data" "${market_data_count:-0}"

# Calculate coverage
if [ "${watchlist_count:-0}" -gt 0 ] && [ "${daily_bars_symbols:-0}" -gt 0 ]; then
    coverage=$(awk "BEGIN {printf \"%.1f\", (${daily_bars_symbols:-0} / ${watchlist_count:-1}) * 100}")
    echo ""
    echo "Equity Coverage: ${daily_bars_symbols}/${watchlist_count} symbols (${coverage}%)"
fi

# Check OPTIONS backfill
echo ""
echo "📊 OPTIONS BACKFILL (QuestDB):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

options_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20options_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
options_symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20count_distinct%28underlying%29%20FROM%20options_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "%-40s %'15d rows\n" "Options data records" "${options_count:-0}"
printf "%-40s %'15d symbols\n" "Underlying symbols with options" "${options_symbols:-0}"

if [ "${watchlist_count:-0}" -gt 0 ] && [ "${options_symbols:-0}" -gt 0 ]; then
    options_coverage=$(awk "BEGIN {printf \"%.1f\", (${options_symbols:-0} / ${watchlist_count:-1}) * 100}")
    echo ""
    echo "Options Coverage: ${options_symbols}/${watchlist_count} symbols (${options_coverage}%)"
fi

# Check NEWS backfill
echo ""
echo "📰 NEWS BACKFILL (PostgreSQL & QuestDB):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pg_news=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM news_events;" 2>/dev/null | xargs)
pg_news_symbols=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(DISTINCT symbol) FROM news_events WHERE symbol IS NOT NULL AND symbol != '';" 2>/dev/null | xargs)
questdb_news=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20news_items" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "%-40s %'15s articles\n" "PostgreSQL news_events" "${pg_news:-0}"
printf "%-40s %'15s symbols\n" "Symbols with news (PostgreSQL)" "${pg_news_symbols:-0}"
printf "%-40s %'15d articles\n" "QuestDB news_items" "${questdb_news:-0}"

# Check SOCIAL backfill
echo ""
echo "💬 SOCIAL SENTIMENT BACKFILL (QuestDB):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

social_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20social_signals" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
social_symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20count_distinct%28symbol%29%20FROM%20social_signals" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "%-40s %'15d signals\n" "Social signals" "${social_count:-0}"
printf "%-40s %'15d symbols\n" "Symbols with social data" "${social_symbols:-0}"

if [ "${watchlist_count:-0}" -gt 0 ] && [ "${social_symbols:-0}" -gt 0 ]; then
    social_coverage=$(awk "BEGIN {printf \"%.1f\", (${social_symbols:-0} / ${watchlist_count:-1}) * 100}")
    echo ""
    echo "Social Coverage: ${social_symbols}/${watchlist_count} symbols (${social_coverage}%)"
fi

# Check CALENDAR backfill
echo ""
echo "📅 CALENDAR EVENTS (QuestDB):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

earnings_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20earnings_calendar" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
dividends_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20dividends_calendar" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
splits_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20splits_calendar" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
ipo_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20ipo_calendar" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "%-40s %'15d events\n" "Earnings calendar" "${earnings_count:-0}"
printf "%-40s %'15d events\n" "Dividends calendar" "${dividends_count:-0}"
printf "%-40s %'15d events\n" "Splits calendar" "${splits_count:-0}"
printf "%-40s %'15d events\n" "IPO calendar" "${ipo_count:-0}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BACKFILL SUMMARY:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Watchlist Size: ${watchlist_count} symbols"
echo ""
echo "Data Type Coverage:"
printf "  %-30s %s symbols\n" "Equity (daily bars):" "${daily_bars_symbols:-0}"
printf "  %-30s %s symbols\n" "Options (underlying):" "${options_symbols:-0}"
printf "  %-30s %s symbols\n" "News (coverage):" "${pg_news_symbols:-0}"
printf "  %-30s %s symbols\n" "Social (coverage):" "${social_symbols:-0}"
echo ""
echo "Calendar Events: $((${earnings_count:-0} + ${dividends_count:-0} + ${splits_count:-0} + ${ipo_count:-0})) total"
echo ""
echo "═══════════════════════════════════════════════════════════════"
