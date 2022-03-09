#pragma once

#include "product.cuh"

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585

namespace bonsai {
  namespace product {

    template <class T>
    class EuropeanCall : IProduct<T> {
      public:

        __host__
        EuropeanCall(const double strike, const Time exerciseDate,
            const Time settlementDate)
          : strike_(strike),
            exerciseDate_(exerciseDate),
            settlementDate_(settlementDate) 
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

          std::ostringstream ss;
          ss.precision(2);
          ss << std::fixed;
          if (settlementDate_ == exerciseDate_) {
            ss << "call " << strike << " " << exerciseDate_;
          } else {
            ss << "call " << strike << " " << exerciseDate_
              << " " << settlementDate_;
          }
          labels_.resize(1);
          labels_[0] = ss.str();
        }

        __host__ 
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

        __host__
        const container<std::string>& GetPayoffLabels() const override {
          return labels_;
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
