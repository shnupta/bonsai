#pragma once

#include "model.cuh"
#include "../common/common.cuh"

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
            timeline_.clear();
            int count = 0;
            for (int i = 0; i < prdTimeline.size(); ++i) {
              if (prdTimeline[i] > SYSTEM_TIME) count++;
            }
            timeline_.resize(count);
            count = 0;
            for (int i = 0; i < prdTimeline.size(); ++i) {
              if (prdTimeline[i] > SYSTEM_TIME) {
                timeline_[0] = prdTimeline[i];
                count++;
              }
            }

            isTodayOnTimeline_ = (prdTimeline[0] == SYSTEM_TIME);
            defline_ = &prdDefline;

            stds_.resize(timeline_.size() - 1);
            drifts_.resize(timeline_.size() - 1);

            const int n = prdTimeline.size();
            numeraires_.resize(n);
            discounts_.resize(n);
            forwardFactors_.resize(n);
            libors_.resize(n);
            for (int i = 0; i < n; ++i) {
              discounts_[i].resize(prdDefline[i].discountMats.size());
              forwardFactors_[i].resize(prdDefline[i].forwardMats.size());
              libors_[i].resize(prdDefline[i].liborDefs.size());
            }
          }

        __host__
          void Initialise(const container<Time>& prdTimeline,
              const container<SampleDef>& prdDefline) override {
            const T mu = rate_ - dividendYield_;
            const int n = timeline_.size() - 1;

            for (int i = 0; i < n; ++i) {
              const double dt = timeline_[i+1] - timeline_[i];
              stds_[i] = vol_ * sqrt(dt);

              if (isSpotMeasure_) {
                drifts_[i] = (mu + 0.5 * vol_ * vol_) * dt;
              } else {
                drifts_[i] = (mu - 0.5 * vol_ * vol_) * dt;
              }
            }

            // Precompute the forwards, discounts and libors over the timeline
            const int m = timeline_.size();
            for (int i = 0; i < m; ++i) {
              if (prdDefline[i].numeraire) {
                if (isSpotMeasure_) {
                  numeraires_[i] = exp(dividendYield_ * prdTimeline[i]) / spot_;
                } else {
                  numeraires_[i] = exp(rate_ * prdTimeline[i]);
                }
              }

              // Forward factors
              const int pFF = prdDefline[i].forwardMats.size();
              for (int j = 0; j < pFF; ++j) {
                forwardFactors_[i][j] =
                  exp(mu * (prdDefline[i].forwardMats[j] - prdTimeline[i]));
              }

              // Discounts
              const int pDF = prdDefline[i].discountMats.size();
              for (int j = 0; j < pDF; ++j) {
                discounts_[i][j] =
                  exp(-rate_ * (prdDefline[i].discountMats[j] 
                        - prdTimeline[i]));
              }

              // Libors
              const int pL = prdDefline[i].liborDefs.size();
              for (int j = 0; j < pL; ++j) {
                const double dt = prdDefline[i].liborDefs[j].end
                  - prdDefline[i].liborDefs[j].start;
                libors_[i][j] = (exp(rate_ * dt) - 1.0) / dt;
              }
            }
          }

        __host__ __device__
          int GetSimulationDimension() const override {
            return timeline_.size() - 1;
          }

        // Per-thread responsability to consume the vector of gaussians 
        // and generate a path
        // NOTE: Given this vector of gaussians is generated from Sobol
        // I'll need to change the ordering of the curand generated output
        // So that each thread has a container of _timeSteps_ random variables,
        // one from each dimension of Sobol
        __device__
          void GeneratePath(const container<double>& gaussVec,
              Scenario<T>& path) const override {
            T spot = spot_;
            int index = 0;
            if (isTodayOnTimeline_) {
              FillScenario(index, spot, path[index], (*defline_)[index]);  
              ++index;
            }

            const int n = timeline_.size() - 1;
            for (int i = 0; i < n; ++i) {
              spot = spot * exp(drifts_[i] + stds_[i] * gaussVec[i]);
              FillScenario(index, spot, path[index], (*defline_)[index]);
              ++index;
            }
          }

      private:
        __host__
          void SetParameterPointers() {
            parameters_[0] = &spot_;
            parameters_[1] = &vol_;
            parameters_[2] = &rate_;
            parameters_[3] = &dividendYield_;
          }

        // Fill a sample in a Scenario given the definition and spot
        __device__
          inline void FillScenario(const int index, const T& spot, 
              Sample<T>& scen, const SampleDef& def) const {
            if (def.numeraire) {
              scen.numeraire = numeraires_[index];
              if (isSpotMeasure_) scen.numeraire *= spot;
            }

            // Fill forwards
            for (int j = 0; j < forwardFactors_[index].size(); ++j) {
              scen.forwards[j] = spot * forwardFactors_[index][j];
            }

            // Fill discounts and libors
            for (int j = 0; j < discounts_[index].size(); ++j) {
              scen.discounts[j] = discounts_[index][j];
            }
            for (int j = 0; j < libors_[index].size(); ++j) {
              scen.libors[j] = libors_[index][j];
            }
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
