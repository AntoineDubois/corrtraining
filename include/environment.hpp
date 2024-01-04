#pragma once
#include <Eigen/Dense>
#include "state.hpp"
#include "action.hpp"
#include "agent.hpp"
#include "buffer.hpp"
#include "reward.hpp"


class Environment
{
    private:
        unsigned n, b, n_max;
        Reward reward;
        Agent* p_agent;
    public:
        Environment(unsigned n_max, Agent* p_agent);
        double Episode();
        void Transition(State& state, Action& action);
        bool notTerminal(unsigned t);
};