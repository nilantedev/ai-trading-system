#!/bin/bash

# QuestDB Comprehensive Data Report
# Checks all tables and generates detailed statistics

echo "═══════════════════════════════════════════════════════════════"
echo "  QUESTDB COMPREHENSIVE DATA REPORT"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Function to query QuestDB via HTTP
query_questdb() {
    local query="$1"
    curl -s "http://localhost:9000/exec?query=$(echo "$query" | sed 's/ /%20/g' | sed 's/\*/%2A/g' | sed 's/(/%28/g' | sed 's/)/%29/g' | sed 's/,/%2C/g' | sed 's/=/%3D/g' | sed 's/>/%3E/g' | sed "s/'/%27/g")" 2>/dev/null
}

# Get all tables
echo "📊 Fetching QuestDB Tables..."
tables=$(curl -s "http://localhost:9000/exec?query=SHOW+TABLES" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null | grep -v "^sys\." | grep -v "^telemetry")

if [ -z "$tables" ]; then
    echo "❌ No tables found or unable to query QuestDB"
    exit 1
fi

table_count=$(echo "$tables" | wc -l)
echo "✅ Found $table_count tables (excluding system tables)"
echo ""

# Initialize counters
declare -A data_types
data_types["equity"]=0
data_types["options"]=0
data_types["news"]=0
data_types["social"]=0
data_types["calendar"]=0
data_types["other"]=0

# Count rows in each table
echo "📈 DATA SUMMARY BY TABLE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-40s %20s %15s\n" "TABLE NAME" "TOTAL ROWS" "TYPE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

total_rows=0
while IFS= read -r table; do
    if [ -n "$table" ]; then
        # Get row count
        count_result=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20$table" 2>/dev/null)
        count=$(echo "$count_result" | jq -r '.dataset[]?[0]?' 2>/dev/null)
        
        # Classify table type
        type="Other"
        if [[ "$table" =~ (bar|daily_bars|market_data|price|ohlc|equity) ]]; then
            type="Equity"
            data_types["equity"]=$((${data_types["equity"]} + ${count:-0}))
        elif [[ "$table" =~ option ]]; then
            type="Options"
            data_types["options"]=$((${data_types["options"]} + ${count:-0}))
        elif [[ "$table" =~ news ]]; then
            type="News"
            data_types["news"]=$((${data_types["news"]} + ${count:-0}))
        elif [[ "$table" =~ (social|sentiment|reddit|twitter) ]]; then
            type="Social"
            data_types["social"]=$((${data_types["social"]} + ${count:-0}))
        elif [[ "$table" =~ (calendar|earnings|dividend|split|ipo) ]]; then
            type="Calendar"
            data_types["calendar"]=$((${data_types["calendar"]} + ${count:-0}))
        else
            data_types["other"]=$((${data_types["other"]} + ${count:-0}))
        fi
        
        if [ -n "$count" ] && [ "$count" != "null" ]; then
            printf "%-40s %'20d %15s\n" "$table" "$count" "$type"
            total_rows=$((total_rows + count))
        else
            printf "%-40s %20s %15s\n" "$table" "Error" "$type"
        fi
    fi
done <<< "$tables"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-40s %'20d\n" "TOTAL ROWS ACROSS ALL TABLES" "$total_rows"
echo ""

# Summary by data type
echo "📊 DATA SUMMARY BY TYPE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-30s %'20d rows\n" "Equity/Market Data" "${data_types["equity"]}"
printf "%-30s %'20d rows\n" "Options Data" "${data_types["options"]}"
printf "%-30s %'20d rows\n" "News Events" "${data_types["news"]}"
printf "%-30s %'20d rows\n" "Social Signals" "${data_types["social"]}"
printf "%-30s %'20d rows\n" "Calendar Events" "${data_types["calendar"]}"
printf "%-30s %'20d rows\n" "Other/Analytics" "${data_types["other"]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-30s %'20d rows\n" "GRAND TOTAL" "$total_rows"
echo ""

# Check data coverage by symbol for key tables
echo "🎯 DATA COVERAGE (Sample Symbols):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check daily_bars
if echo "$tables" | grep -q "daily_bars"; then
    symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20DISTINCT%20symbol%20FROM%20daily_bars%20LIMIT%2010" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null | tr '\n' ' ')
    symbol_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28DISTINCT%20symbol%29%20FROM%20daily_bars" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
    echo "Equity Bars: $symbol_count symbols"
    echo "  Sample: $symbols"
fi

# Check options_data
if echo "$tables" | grep -q "options_data"; then
    symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20DISTINCT%20underlying_symbol%20FROM%20options_data%20LIMIT%2010" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null | tr '\n' ' ')
    symbol_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28DISTINCT%20underlying_symbol%29%20FROM%20options_data" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
    if [ -n "$symbol_count" ] && [ "$symbol_count" != "null" ]; then
        echo "Options Data: $symbol_count underlying symbols"
        echo "  Sample: $symbols"
    fi
fi

# Check news_events
if echo "$tables" | grep -q "news_events"; then
    symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20DISTINCT%20symbol%20FROM%20news_events%20LIMIT%2010" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null | tr '\n' ' ')
    symbol_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28DISTINCT%20symbol%29%20FROM%20news_events" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
    if [ -n "$symbol_count" ] && [ "$symbol_count" != "null" ]; then
        echo "News Events: $symbol_count symbols"
        echo "  Sample: $symbols"
    fi
fi

# Check social_signals
if echo "$tables" | grep -q "social_signals"; then
    symbols=$(curl -s "http://localhost:9000/exec?query=SELECT%20DISTINCT%20symbol%20FROM%20social_signals%20LIMIT%2010" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null | tr '\n' ' ')
    symbol_count=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28DISTINCT%20symbol%29%20FROM%20social_signals" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
    if [ -n "$symbol_count" ] && [ "$symbol_count" != "null" ]; then
        echo "Social Signals: $symbol_count symbols"
        echo "  Sample: $symbols"
    fi
fi

echo ""

# Check recent activity (last 24 hours)
echo "⏰ RECENT ACTIVITY (Last 24 Hours):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for table in daily_bars options_data news_events social_signals calendar_events; do
    if echo "$tables" | grep -q "^${table}$"; then
        # Try timestamp column
        recent=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20${table}%20WHERE%20timestamp%20%3E%20dateadd%28%27d%27%2C%20-1%2C%20now%28%29%29" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
        if [ -z "$recent" ] || [ "$recent" = "null" ]; then
            # Try created_at column
            recent=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20${table}%20WHERE%20created_at%20%3E%20dateadd%28%27d%27%2C%20-1%2C%20now%28%29%29" 2>/dev/null | jq -r '.dataset[]?[0]?' 2>/dev/null)
        fi
        if [ -n "$recent" ] && [ "$recent" != "null" ]; then
            printf "%-30s %'15d rows\n" "$table" "$recent"
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report Complete"
echo "═══════════════════════════════════════════════════════════════"
