#!/bin/bash

# Comprehensive Data Collection Status Report
# Verifies what data is being actively collected across all types

echo "═══════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE DATA COLLECTION STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check container status
echo "📦 Data Ingestion Container:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker ps --filter "name=data-ingestion" --format "{{.Names}}: {{.Status}}"
echo ""

# Check environment variables
echo "⚙️  Collection Enablement Flags:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker exec trading-data-ingestion env | grep -E "^ENABLE_" | grep -E "(EQUITY|OPTION|NEWS|SOCIAL|CALENDAR|QUOTE|STREAM)" | sort | while read line; do
    key=$(echo "$line" | cut -d= -f1)
    value=$(echo "$line" | cut -d= -f2)
    if [ "$value" = "true" ]; then
        echo "✅ $key"
    else
        echo "❌ $key"
    fi
done
echo ""

# Check recent collection activity (last 2 hours)
echo "📊 RECENT COLLECTION ACTIVITY (Last 2 Hours):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Equity bars
equity_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -i "daily delta fetched" | wc -l)
echo "Equity Bars (Daily Delta):     $equity_count fetches"

# Options data
options_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -iE "option.*collected|option.*fetched|option.*snapshot" | wc -l)
echo "Options Data:                   $options_count collections"

# News events
news_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -i "collected.*news articles" | wc -l)
news_articles=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -i "collected.*news articles" | tail -1 | grep -oP '\d+(?= news articles)')
echo "News Events:                    $news_count collections ($news_articles articles recent)"

# Social signals
social_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -i "collected.*social signals" | wc -l)
echo "Social Signals:                 $social_count collections"

# Calendar events
calendar_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -iE "calendar|earnings|dividend" | grep -iE "collected|fetched" | wc -l)
echo "Calendar Events:                $calendar_count collections"

# Quote stream
quote_count=$(docker logs trading-data-ingestion --since 2h 2>&1 | grep -iE "quote.*stream|live.*quote" | wc -l)
echo "Live Quotes (Streaming):        $quote_count updates"

echo ""

# DATABASE STORAGE STATUS
echo "💾 DATABASE STORAGE STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# QuestDB Summary
echo ""
echo "📊 QuestDB (Time-Series):"
questdb_total=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20daily_bars" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
questdb_options=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20options_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
questdb_news=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20news_events" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
questdb_social=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20social_signals" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
questdb_market=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20market_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)

printf "  %-30s %'15d rows\n" "Daily Bars (daily_bars)" "${questdb_total:-0}"
printf "  %-30s %'15d rows\n" "Market Data (market_data)" "${questdb_market:-0}"
printf "  %-30s %'15d rows\n" "Options Data (options_data)" "${questdb_options:-0}"
printf "  %-30s %'15d rows\n" "News Events (news_events)" "${questdb_news:-0}"
printf "  %-30s %'15d rows\n" "Social Signals (social_signals)" "${questdb_social:-0}"

total_questdb=$((${questdb_total:-0} + ${questdb_market:-0} + ${questdb_options:-0} + ${questdb_news:-0} + ${questdb_social:-0}))
printf "  %-30s %'15d rows\n" "TOTAL" "$total_questdb"

# PostgreSQL Summary
echo ""
echo "🐘 PostgreSQL (Relational):"
pg_news=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM news_events;" 2>/dev/null | xargs)
pg_bars=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM historical_daily_bars;" 2>/dev/null | xargs)
pg_backfill=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | xargs)

printf "  %-30s %'15s rows\n" "Historical Daily Bars" "${pg_bars:-0}"
printf "  %-30s %'15s rows\n" "News Events" "${pg_news:-0}"
printf "  %-30s %'15s symbols\n" "Backfill Progress" "${pg_backfill:-0}"

# Weaviate Summary
echo ""
echo "🔮 Weaviate (Vector Store):"
for collection in NewsArticle SocialSentiment OptionContract EquityBar; do
    # Get actual count via aggregation
    count=$(curl -s -X POST "http://localhost:8080/v1/graphql" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"{ Aggregate { ${collection} { meta { count } } } }\"}" 2>/dev/null | \
        jq -r ".data.Aggregate.${collection}[0].meta.count // 0" 2>/dev/null)
    printf "  %-30s %'15s objects\n" "$collection" "${count:-0}"
done

echo ""

# Check for collection gaps or issues
echo "⚠️  DATA COLLECTION HEALTH:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if loops are running
loops_running=$(docker logs trading-data-ingestion --since 10m 2>&1 | grep -E "loop started|scheduler started" | wc -l)
if [ "$loops_running" -gt 0 ]; then
    echo "✅ Collection loops are active"
else
    echo "⚠️  No loop startups in last 10 minutes (may be normal if stable)"
fi

# Check for errors
error_count=$(docker logs trading-data-ingestion --since 1h 2>&1 | grep -iE "error|failed|exception" | grep -v "last_error" | wc -l)
if [ "$error_count" -lt 10 ]; then
    echo "✅ Low error rate ($error_count errors in last hour)"
else
    echo "⚠️  Elevated error rate ($error_count errors in last hour)"
fi

# Check recent data flow
recent_flow=$(docker logs trading-data-ingestion --since 10m 2>&1 | grep -iE "collected|fetched|saved" | wc -l)
if [ "$recent_flow" -gt 10 ]; then
    echo "✅ Active data flow ($recent_flow collection events in last 10 min)"
else
    echo "⚠️  Low data flow ($recent_flow collection events in last 10 min)"
fi

echo ""

# Summary by data type
echo "📈 COLLECTION SUMMARY:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$equity_count" -gt 0 ]; then
    echo "✅ Equity Data:     COLLECTING ($equity_count recent fetches)"
else
    echo "⚠️  Equity Data:     LOW ACTIVITY"
fi

if [ "$options_count" -gt 0 ]; then
    echo "✅ Options Data:    COLLECTING ($options_count recent)"
else
    echo "⚠️  Options Data:    LOW ACTIVITY (may be daily schedule)"
fi

if [ "$news_count" -gt 0 ]; then
    echo "✅ News Events:     COLLECTING ($news_count cycles, ${news_articles:-0} articles)"
else
    echo "⚠️  News Events:     LOW ACTIVITY"
fi

if [ "$social_count" -gt 50 ]; then
    echo "✅ Social Signals:  ACTIVE ($social_count collections)"
else
    echo "⚠️  Social Signals:  LOW ACTIVITY ($social_count collections)"
fi

if [ "$calendar_count" -gt 0 ]; then
    echo "✅ Calendar Events: COLLECTING ($calendar_count recent)"
else
    echo "ℹ️  Calendar Events: LOW ACTIVITY (may be periodic)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report Complete"
echo "═══════════════════════════════════════════════════════════════"
