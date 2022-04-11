#pragma once

#include "product.cuh"

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

#include <sstream>

namespace bonsai {
  namespace product {

    template <class T>
    class BinaryAsian : IProduct<T> {
      public:

        __host__ __device__
          BinaryAsian(const double strike, const Time monitorFreq,
              const Time maturity)
            : strike_(strike),
              monitorFreq_(monitorFreq),
              maturity_(maturity),
              labels_(1)
          {
            // timeline_
            // + 2 for now and exerciseDate
            const int n = ((maturity_ - ONE_HOUR) / monitorFreq_) + 2;
            timeline_.resize(n);
            timeline_[0] = SYSTEM_TIME;

            double t = SYSTEM_TIME + monitorFreq_;
            int i = 1;
            while (maturity_ - t > ONE_HOUR) {
              timeline_[i] = t;
              t += monitorFreq_;
              ++i;
            }
            timeline_[n-1] = maturity_;

            // defline_
            defline_.resize(n);
            for (int i = 0; i < n; ++i) {
              defline_[i].forwardMats.resize(1);
              defline_[i].numeraire = false;
              defline_[i].forwardMats[0] = timeline_[i];
            }
            defline_[n-1].numeraire = true; // Last date requires payment
          }

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
            // TODO
            // Compute arithmetic average of stock price across all forwards
            T avg = 0.0;
            const int n = path.size();
            for (int i = 0; i < n; ++i) {
              avg = (i * avg + path[i].forwards[0]) / (i + 1.0);
            }
            payoffs[0] = avg > strike_ ? 1.0 : 0.0;
            /* payoffs[0] = path[0].forwards[0] - strike_ > 0 ? 1.0 : 0.0; */
          }

      private:
        double strike_;
        Time maturity_;
        Time monitorFreq_;

        container<Time> timeline_;
        container<SampleDef> defline_;
        container<std::string> labels_; // TODO: string on gpu
    };

  } // namespace product
} // namespace bonsai
