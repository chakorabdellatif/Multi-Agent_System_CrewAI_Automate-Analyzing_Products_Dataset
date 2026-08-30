const { getHealthStatus } = require('../monitoring/health');
const { runEvaluation } = require('../scripts/eval_harness');

describe('Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset Observability & QA', () => {
  test('health status returns UP', () => {
    const health = getHealthStatus();
    expect(health.status).toBe('UP');
    expect(health.service).toBe('Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset');
  });

  test('evaluation harness produces passing report', () => {
    const results = runEvaluation();
    expect(results.status).toBe('PASSED');
    expect(results.metrics.qualityIndex).toBeGreaterThan(0.5);
  });
});
