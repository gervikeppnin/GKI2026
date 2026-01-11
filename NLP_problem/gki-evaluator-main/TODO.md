# Testing Framework TODO

## Status: Planning Complete ✓

## Next Steps (In Order)

### Phase 1: Environment Setup
- [ ] Build Docker image
- [ ] Install evaluator package
- [ ] Prepare test dataset

### Phase 2: Baseline Validation
- [ ] Test dummy baseline (verify ~8.0 score)
- [ ] Test MLP baseline (verify < 8.0 score)

### Phase 3: Advanced Model Testing
- [ ] Create GPT-2 submission wrapper
- [ ] Package and test GPT-2 submission

### Phase 4: Unit Tests
- [ ] Create unit tests for context building
- [ ] Create unit tests for scoring calculation

### Phase 5: Security & Integration Tests
- [ ] Create security tests (network, filesystem, resources)
- [ ] Create zip validation tests
- [ ] Create protocol error handling tests

### Phase 6: Edge Cases
- [ ] Create edge case and stress tests

### Phase 7: Automation
- [ ] Create automated test suite and validation scripts

### Phase 8: Documentation
- [ ] Create test report
- [ ] Create scoring analysis documentation

## Quick Start Commands

```bash
# Build Docker image
docker build -t gki_evaluator -f docker/Dockerfile docker/

# Install package
pip install -e .

# Test dummy baseline
cd examples/dummy && python package.py
python -m gki_evaluator.evaluate --submission examples/dummy/output/submission.zip --test-data data/

# Test MLP baseline
cd examples/mlp && python train.py
python -m gki_evaluator.evaluate --submission examples/mlp/output/submission.zip --test-data data/
```

## Notes
- Full testing plan in `TESTING_PLAN.md`
- Continue on Claude iOS app after pushing this code
