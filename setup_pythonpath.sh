#!/bin/bash
# Setup PYTHONPATH for the AI Trading System

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export PYTHONPATH to include all necessary directories
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/shared/python-common:${PROJECT_ROOT}/services:${PYTHONPATH}"

echo "PYTHONPATH set to: $PYTHONPATH"