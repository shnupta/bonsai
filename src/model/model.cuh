#pragma once

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

namespace bonsai {
  namespace model {

    template <class T>
    class IModel {
      public:

        // Separate allocation and initialisation from product timeline and
        // defline
        __host__ __device__
          virtual void Allocate(const container<Time>& prdTimeline,
              const container<SampleDef>& prdDefline) = 0;
        __host__ __device__
          virtual void Initialise(const container<Time>& prdTimeline,
              const container<SampleDef>& prdDefline) = 0;

        __host__ __device__
          virtual int GetSimulationDimension() const = 0;

        // Simulate a path whilst consuming the container<double> of independent
        // gaussians
        __device__
          virtual void GeneratePath(const container<double>& gaussVec,
              Scenario<T>& path) const = 0;

        // TODO: Check if I need these functions on __device__
        __host__ __device__
          virtual const container<T*>& GetParameters() = 0;
        __host__
          virtual const container<std::string>& GetParameterLabels() = 0;
        __host__ __device__
          virtual int GetNumParameters() const {
            return const_cast<IModel*>(this)->GetParameters().size();
          }
    };

  } // namespace model
} // namespace bonsai
