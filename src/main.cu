#include "common/sample.cuh"
#include "common/container.cuh"
#include "model/black_scholes.cuh"
#include "product/european.cuh"

using namespace bonsai;

// Test code

int main() {
  const double spot = 100.0;
  const double strike = 100.0;
  const double vol = 0.2;
  const double rate = 0.03;
  const double ttm = 1.0 / 12.0; // 1 month
  const bool isSpotMeasure = false; // risk neutral measure
  const int numPaths = 32000;

  product::EuropeanCall<double> call(strike, ttm);
  const int numPayoffs = call.GetPayoffLabels().size();
  container<container<double>> results(numPaths, numPayoffs);

  model::BlackScholesModel<double> bsm(spot, vol, isSpotMeasure, rate); // No dividend yield
  bsm.Allocate(call.GetTimeline(), call.GetDefline());
  bsm.Initialise(call.GetTimeline(), call.GetDefline());

  return 0;
}
