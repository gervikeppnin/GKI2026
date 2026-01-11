#!/bin/bash
# Comprehensive test suite for GKÍ Evaluator
# Runs all unit tests, integration tests, and baseline validation

set -e  # Exit on first error

# Colors for output
GREEN='\\033[0;32m'
RED='\\033[0;31m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

echo "============================================================"
echo "GKÍ EVALUATOR - COMPREHENSIVE TEST SUITE"
echo "============================================================"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local name="$1"
    local command="$2"

    echo ""
    echo "${BLUE}Running: $name${NC}"
    echo "------------------------------------------------------------"

    ((TOTAL_TESTS++))

    if eval "$command"; then
        echo "${GREEN}✓ $name passed${NC}"
        ((PASSED_TESTS++))
    else
        echo "${RED}✗ $name failed${NC}"
        ((FAILED_TESTS++))
    fi
}

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "${RED}Error: Virtual environment not found${NC}"
    echo "Run: uv venv && uv pip install -e ."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================
# PHASE 1: Unit Tests
# ============================================================

echo "============================================================"
echo "PHASE 1: UNIT TESTS"
echo "============================================================"

run_test "Context Building Tests" "python tests/test_context_building.py"
run_test "Scoring Calculation Tests" "python tests/test_scoring.py"
run_test "Zip Validation Tests" "python tests/test_zip_validation.py"
run_test "Edge Case Tests" "python tests/test_edge_cases.py"

# ============================================================
# PHASE 2: Baseline Validation
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 2: BASELINE VALIDATION"
echo "============================================================"

# Package dummy baseline if needed
if [ ! -f "examples/dummy/output/submission.zip" ]; then
    echo "Packaging dummy baseline..."
    cd examples/dummy && python package.py && cd ../..
fi

run_test "Dummy Baseline Evaluation" \\
    "python -m gki_evaluator.evaluate --submission examples/dummy/output/submission.zip --test-data data/ --output output/dummy_score.json 2>&1 | grep -q '8.00'"

# Train and package MLP if needed
if [ ! -f "examples/mlp/output/submission.zip" ]; then
    echo "Training MLP baseline..."
    cd examples/mlp && python train.py && cd ../..
fi

run_test "MLP Baseline Evaluation" \\
    "python -m gki_evaluator.evaluate --submission examples/mlp/output/submission.zip --test-data data/ --output output/mlp_score.json 2>&1 | tail -5 | head -1 | grep -q 'SCORE'"

# ============================================================
# PHASE 3: Score Validation
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 3: SCORE VALIDATION"
echo "============================================================"

run_test "Score Validation Script" "python tests/validate_scores.py"

# ============================================================
# PHASE 4: Integration Tests (Informational)
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 4: INTEGRATION TESTS (INFORMATIONAL)"
echo "============================================================"

echo ""
echo "${BLUE}Generating security test submissions...${NC}"
python tests/test_security.py > /dev/null
echo "✓ Security test submissions created in /tmp"

echo ""
echo "${BLUE}Generating protocol error test submissions...${NC}"
python tests/test_integration.py > /dev/null
echo "✓ Protocol error test submissions created in /tmp"

echo ""
echo "${BLUE}Note: Security and protocol tests must be run manually${NC}"
echo "See tests/test_security.py and tests/test_integration.py for details"

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "============================================================"
echo "TEST SUMMARY"
echo "============================================================"
echo ""
echo "Total tests run: $TOTAL_TESTS"
echo "${GREEN}Passed: $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo "${RED}Failed: $FAILED_TESTS${NC}"
else
    echo "Failed: $FAILED_TESTS"
fi
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "${GREEN}============================================================${NC}"
    echo "${GREEN}ALL TESTS PASSED ✓${NC}"
    echo "${GREEN}============================================================${NC}"
    exit 0
else
    echo "${RED}============================================================${NC}"
    echo "${RED}SOME TESTS FAILED ✗${NC}"
    echo "${RED}============================================================${NC}"
    exit 1
fi
