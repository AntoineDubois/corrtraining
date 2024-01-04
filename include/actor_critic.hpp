#pragma once
#include <Eigen/Dense>
#include "state.hpp"
#include "action.hpp"
#include "environment.hpp"
#include "dynet/dynet.h"
#include "buffer.hpp"
#include "model.hpp"
#include "agent.hpp"

class AAC: public Agent 
{
    private:
        Environment env;
        Model model;
    public:
        AAC(unsigned n_max, bool isDet, unsigned batch_size = 64, unsigned capacity = 10'000);
        void Train(unsigned Ntrain = 100, unsigned train_each = 1, unsigned display_each = 10, unsigned Ntests_display = 10);
        double Test(unsigned Ntests = 10);
        Action Policy(State& state);
        bool& IsSto();
        bool const& IsSto() const;
    private:
        Action PolicySto(State& state, std::vector<dynet::real>& proba1, std::vector<dynet::real>& proba2);
        Action PolicyGreedy(State& state, std::vector<dynet::real>& proba1, std::vector<dynet::real>& proba2);
        void Backpropagation();
};