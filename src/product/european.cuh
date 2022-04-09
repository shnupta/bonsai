#pragma once

#include "product.cuh"

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

#include <sstream>

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585

namespace bonsai {
  namespace product {

    template <class T>
    class EuropeanCall : IProduct<T> {
      public:

        __host__ __device__
          EuropeanCall(const double strike, const Time exerciseDate,
              const Time settlementDate)
            : strike_(strike),
              exerciseDate_(exerciseDate),
              settlementDate_(settlementDate),
              labels_(1)
          {
            // Timeline is just the exercise date
            timeline_.resize(1);
            timeline_[0] = exerciseDate;

            // Defline: one sample on the exersice date
            defline_.resize(1);
            defline_[0].numeraire = true;
            defline_[0].forwardMats.resize(1);
            defline_[0].forwardMats[0] = settlementDate_;
            defline_[0].discountMats.resize(1);
            defline_[0].discountMats[0] = settlementDate_;

            /* std::ostringstream ss; */
            /* ss.precision(2); */
            /* ss << std::fixed; */
            /* if (settlementDate_ == exerciseDate_) { */
            /*   ss << "call " << strike << " " << exerciseDate_; */
            /* } else { */
            /*   ss << "call " << strike << " " << exerciseDate_ */
            /*     << " " << settlementDate_; */
            /* } */
            /* labels_.resize(1); */
            /* labels_[0] = ss.str(); */
          }

        __host__ __device__
          EuropeanCall(const double strike, const Time exerciseDate)
            : EuropeanCall(strike, exerciseDate, exerciseDate) {}

        __host__ __device__
          const container<Time>& GetTimeline() const override {
            return timeline_;
          }

        __host__ __device__
          const container<SampleDef>& GetDefline() const override {
            return defline_;
          }

        __host__ __device__
          const container<std::string>& GetPayoffLabels() const override {
            return labels_;
          }

        // Per-thread responsability to compute the payoff from this path
        // simulation
        __device__
          void ComputePayoffs(const Scenario<T>& path,
              T* payoffs) const override {
            payoffs[0] = max(path[0].forwards[0] - strike_, 0.0)
              * path[0].discounts[0] / path[0].numeraire;
          }

      private:
        double strike_;
        Time exerciseDate_;
        Time settlementDate_;

        container<Time> timeline_;
        container<SampleDef> defline_;
        container<std::string> labels_; // TODO: string on gpu
    };

  } // namespace product
} // namespace bonsai
