#pragma once

#include "model.cuh"

#include <cmath>

namespace bonsai {
  namespace model {

    template <class T>
    class BlackScholesModel : IModel<T> {
      public:
        
        template <class U>
        __host__
        BlackScholesModel(const U spot, const U vol,
            const bool isSpotMeasure, const U rate = U(0.0),
            const U dividendYield = U(0.0))
          : spot_(spot), vol_(vol), rate_(rate), dividendYield_(dividendYield),
            isSpotMeasure_(isSpotMeasure), parameters_(4), parameterLabels_(4)
        {
          parameterLabels_[0] = "spot";
          parameterLabels_[1] = "vol";
          parameterLabels_[2] = "rate";
          parameterLabels_[3] = "dividend yield";

          SetParameterPointers();
        }

        __host__ __device__
          const T GetSpot() const {
            return spot_;
          }
        __host__ __device__
          const T GetVol() const {
            return vol_;
          }
        __host__ __device__
          const T GetRate() const {
            return rate_;
          }
        __host__ __device__
          const T GetDividendYield() const {
            return dividendYield_;
          }
        __host__
          const container<T*>& GetParameters() override {
            return parameters_;
          } 
        __host__
          const container<std::string>& GetParameterLabels() override {
            return parameterLabels_;
          }

        __host__
          void Allocate(const container<Time>& prdTimeline,
              const container<SampleDef>& prdDefline) override {
            // TODO (next)
          }

        __host__
          void Initialise(const container<Time>& prdTimeline,
              const container<SampleDef>& prdDefline) override {
            // TODO (next)
          }

        __host__ __device__
          int GetSimulationDimension() const override {
            return timeline_.size() - 1;
          }

        __device__
          void GeneratePath(const container<double>& gaussVec,
              Scenario<T>& path) const override {
            // TODO
          }

      private:
        __host__
          void SetParameterPointers() {
            parameters_[0] = &spot_;
            parameters_[1] = &vol_;
            parameters_[2] = &rate_;
            parameters_[3] = &dividendYield_;
          }


        T spot_;
        T vol_;
        T rate_;
        T dividendYield_; 

        // false = risk neutral measure
        // true = spot measure
        const bool isSpotMeasure_;

        // Simulation timeline = today + product timeline
        container<Time> timeline_;
        bool isTodayOnTimeline_;
        
        // Reference to product's defline
        const container<SampleDef>* defline_;

        // Precalculated on initialisation
        container<T> stds_;
        container<T> drifts_;

        // For mapping spot to sample
        container<container<T>> forwardFactors_;

        // Precalculated
        container<T> numeraires_;
        container<container<T>> discounts_;
        container<container<T>> libors_;

        // Exported parameters
        container<T*> parameters_;
        container<std::string> parameterLabels_;
    };

  } // namespace model
} // namespace bonsai
