#pragma once

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585
#define ONE_MONTH 0.08333333333

namespace bonsai {
  namespace product {

    template <class T>
    class IProduct {
      public:
        __host__ __device__
          virtual const container<Time>& GetTimeline() const = 0; 
        __host__ __device__
          virtual const container<SampleDef>& GetDefline() const = 0;

        __host__ __device__
          virtual const container<std::string>& GetPayoffLabels() const = 0;

        __device__
          virtual void ComputePayoffs(const Scenario<T>& path,
              T* payoffs) const = 0;
    };

  } // namespace product
} // namespace bonsai
