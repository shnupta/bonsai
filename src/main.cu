#include "common/sample.cuh"
#include "common/container.cuh"
#include "model/black_scholes.cuh"
#include "product/european.cuh"

#include <curand.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <utility>

#define THREADBLOCK_SIZE 800

using namespace bonsai;

// Test code

__global__ void AllocateInitialise(model::BlackScholesModel<double>** bsm, 
    double spot, double vol, bool isSpotMeasure, double rate,
    product::EuropeanCall<double>** call, double strike, double ttm) {

  int id = threadIdx.x + blockIdx.x * gridDim.x;
  if (id != 0) return;

  *bsm = new model::BlackScholesModel<double>(spot, vol, isSpotMeasure, rate);
  *call = new product::EuropeanCall<double>(strike, ttm);

  (*bsm)->Allocate((*call)->GetTimeline(), (*call)->GetDefline());
  (*bsm)->Initialise((*call)->GetTimeline(), (*call)->GetDefline());
}

__global__ void AllocateResults(container<container<double> >**results,
    int numPaths, product::EuropeanCall<double>** call) {

  int id = threadIdx.x + blockIdx.x * gridDim.x;
  if (id != 0) return;

  *results = new container<container<double> >(numPaths);
}


// TODO
__global__ void Value(model::BlackScholesModel<double>** bsm,
    product::EuropeanCall<double>** call, double* gauss) {

}

__global__ void TestKernel(model::BlackScholesModel<double>** bsmp) {
  auto* bsm = *bsmp;
  int simdim = bsm->GetSimulationDimension();
  printf("simdim = %d\n", simdim);
}


int main() {
  const double spot = 100.0;
  const double strike = 100.0;
  const double vol = 0.2;
  const double rate = 0.03;
  const double ttm = 1.0 / 12.0; // 1 month
  const bool isSpotMeasure = false; // risk neutral measure
  const int numPaths = 32000;
  
  product::EuropeanCall<double> host_call(strike, ttm);
  model::BlackScholesModel<double> host_bsm(spot, vol, isSpotMeasure, rate);

  host_bsm.Allocate(host_call.GetTimeline(), host_call.GetDefline());
  host_bsm.Initialise(host_call.GetTimeline(), host_call.GetDefline());

  const int simDim = host_bsm.GetSimulationDimension();

  model::BlackScholesModel<double>** dev_bsm;
  product::EuropeanCall<double>** dev_call;
  // TODO: Just have an array of doubles for the result, don't need to worry
  // about size I think
  container<container<double> >** dev_results;

  checkCudaErrors(cudaMalloc((void**) &dev_bsm,
        sizeof(model::BlackScholesModel<double>*)));
  checkCudaErrors(cudaMalloc((void**) &dev_call,
        sizeof(product::EuropeanCall<double>*)));
  checkCudaErrors(cudaMalloc((void**) &dev_results,
        sizeof(container<container<double> >*)));

  AllocateInitialise<<<1,1>>>(dev_bsm, spot, vol, isSpotMeasure, rate,
      dev_call, strike, ttm);
  cudaDeviceSynchronize();
  getLastCudaError("AllocateInitialise");
  AllocateInitialise<<<1,1>>>(dev_bsm, spot, vol, isSpotMeasure, rate,
      dev_call, strike, ttm);
  cudaDeviceSynchronize();
  getLastCudaError("AllocateInitialise");
  AllocateResults<<<1,1>>>(dev_results, numPaths, dev_call);
  cudaDeviceSynchronize();
  getLastCudaError("AllocateResults");

  /// Random variable generation ///
  const int totalThreads = numPaths * THREADBLOCK_SIZE;
  const int randomVarsCount = numPaths * simDim;
  // Need to allocate numPaths * simDim random variables
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(gen, simDim);
  double* devZs;
  checkCudaErrors(cudaMalloc((void**) &devZs,
        randomVarsCount * sizeof(double)));
  cudaDeviceSynchronize();
  curandGenerateNormalDouble(gen, devZs, randomVarsCount, 0, 1);
  /// ///

  cudaFree(devZs);
  curandDestroyGenerator(gen);


  TestKernel<<<1,1>>>(dev_bsm);
  cudaDeviceSynchronize();
  getLastCudaError("Test kernel failed");

  cudaFree(dev_bsm);
  cudaFree(dev_call);

  return 0;
}
