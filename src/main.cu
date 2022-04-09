#include "common/sample.cuh"
#include "common/container.cuh"
#include "model/black_scholes.cuh"
#include "product/european.cuh"

#include <curand.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <chrono>
#include <iostream>

#define THREADBLOCK_SIZE 800

using namespace bonsai;

// Test code

__global__ void AllocateInitialise(model::BlackScholesModel<double>** bsm, 
    double spot, double vol, bool isSpotMeasure, double rate,
    product::EuropeanCall<double>** call, double strike, double ttm) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id != 0) return;

  *bsm = new model::BlackScholesModel<double>(spot, vol, isSpotMeasure, rate);
  *call = new product::EuropeanCall<double>(strike, ttm);

  (*bsm)->Allocate((*call)->GetTimeline(), (*call)->GetDefline());
  (*bsm)->Initialise((*call)->GetTimeline(), (*call)->GetDefline());
}

__global__ void Value(model::BlackScholesModel<double>** bsm,
    product::EuropeanCall<double>** call, double* gauss, int numPaths,
    int numPayoffs, double* results) {
  Scenario<double> lpath;
  AllocatePath((*call)->GetDefline(), lpath);
  InitialisePath(lpath);

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < numPaths) {
    const int payoffIdx = idx * numPayoffs;
    (*bsm)->GeneratePath(&gauss[payoffIdx], lpath);
    (*call)->ComputePayoffs(lpath, &results[payoffIdx]);
  }
}

__global__ void TestKernel() {
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

  const int numPayoffs = host_call.GetPayoffLabels().size();
  const int simDim = host_bsm.GetSimulationDimension();

  model::BlackScholesModel<double>** dev_bsm;
  product::EuropeanCall<double>** dev_call;
  double* dev_results;

  checkCudaErrors(cudaMalloc((void**) &dev_bsm,
        sizeof(model::BlackScholesModel<double>*)));
  checkCudaErrors(cudaMalloc((void**) &dev_call,
        sizeof(product::EuropeanCall<double>*)));
  checkCudaErrors(cudaMalloc((void**) &dev_results,
        numPaths * numPayoffs * sizeof(double)));

  AllocateInitialise<<<1,1>>>(dev_bsm, spot, vol, isSpotMeasure, rate,
      dev_call, strike, ttm);
  cudaDeviceSynchronize();
  getLastCudaError("AllocateInitialise");

  /// Random variable generation ///
  /* const int totalThreads = numPaths * THREADBLOCK_SIZE; */
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

  const int blocks = (numPaths / THREADBLOCK_SIZE) + 1;

  printf("Blocks = %d\n", blocks);
  printf("Kernel start\n");
  auto start = std::chrono::steady_clock::now();
  Value<<<blocks, THREADBLOCK_SIZE>>>(dev_bsm, dev_call, devZs, numPaths, 
      numPayoffs, dev_results);
  cudaDeviceSynchronize();
  getLastCudaError("Value kernel failed");
  auto end = std::chrono::steady_clock::now();
  printf("Kernel end\n");

  double results[numPaths * numPayoffs];
  checkCudaErrors(cudaMemcpy(results, dev_results,
        numPaths * numPayoffs * sizeof(double), cudaMemcpyDeviceToHost));

  double payoffs[numPayoffs];

  for (int i = 0; i < numPayoffs; ++i) {
   double val = 0;
   for (int j = 0; j < numPaths; j += numPayoffs) {
     val += results[j];
   }
   payoffs[i] = val / numPaths;
  }

  auto final_end = std::chrono::steady_clock::now();

  printf("final price = %f\n", payoffs[0]);
  std::cout << "Value kernel time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  start = std::chrono::steady_clock::now();
  TestKernel<<<blocks, THREADBLOCK_SIZE>>>();
  end = std::chrono::steady_clock::now();
  std::cout << "Test kernel time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


  curandDestroyGenerator(gen);
  cudaFree(devZs);
  cudaFree(dev_bsm);
  cudaFree(dev_call);
  cudaFree(dev_results);

  return 0;
}
