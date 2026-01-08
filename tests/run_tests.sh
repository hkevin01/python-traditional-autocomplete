#!/bin/bash
# Test Execution Script for Python Traditional Autocomplete
# Run all tests with coverage and generate reports

set -e

echo "========================================="
echo "Python Traditional Autocomplete Test Suite"
echo "========================================="
echo ""

# Activate venv
source venv/bin/activate

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🧪 Running Test Suite...${NC}"
echo ""

# Run tests with various options
if [ "$1" == "quick" ]; then
    echo "Running quick tests (excluding slow tests)..."
    python -m pytest tests/ -v -m "not slow" --tb=short
elif [ "$1" == "coverage" ]; then
    echo "Running tests with coverage..."
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
elif [ "$1" == "failed" ]; then
    echo "Re-running only failed tests..."
    python -m pytest tests/ -v --lf --tb=short
elif [ "$1" == "verbose" ]; then
    echo "Running tests with verbose output..."
    python -m pytest tests/ -vv --tb=long
else
    echo "Running all tests..."
    python -m pytest tests/ -v --tb=short
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed (exit code: $EXIT_CODE)${NC}"
    echo ""
    echo "To re-run only failed tests:"
    echo "  ./tests/run_tests.sh failed"
    echo ""
    echo "To see more details:"
    echo "  ./tests/run_tests.sh verbose"
fi

echo ""
echo "Available options:"
echo "  ./run_tests.sh         - Run all tests"
echo "  ./run_tests.sh quick   - Skip slow tests"
echo "  ./run_tests.sh coverage - Generate coverage report"
echo "  ./run_tests.sh failed  - Re-run only failed tests"
echo "  ./run_tests.sh verbose - Show detailed output"

exit $EXIT_CODE
