#pragma once
#include <Eigen/Dense>
#include "state.hpp"

class Reward
{
    private:
        bool isDet;
        double reward(State& state);
        double fct(Eigen::MatrixXd block);
    public:
        double init_reward, final_reward;
        Reward(bool isDet);
        double Init(State& init_state);
        double Final(State& final_state);
        double Score();
};