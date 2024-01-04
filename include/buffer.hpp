#pragma once
#include "state.hpp"
#include "action.hpp"
#include <vector>
#include <deque>
#include <vector>
#include "dynet/dynet.h"


struct Transition
{
    State state;
    Action action;
    double g;
    Transition(State state, Action action);
    double& G();
};

class Buffer
{
    private:
        unsigned capacity;
        unsigned batch_size;
        unsigned ep_step;
    public:
        std::deque<Transition> data;
        Buffer(unsigned capacity, unsigned batch_size);
        void PushBack(State& state, Action& action);
        void Backprob(double final_reward);
        void RandomSample(std::vector<dynet::real>& state_buffer, std::vector<dynet::real>& action1_buffer, 
                                std::vector<dynet::real>& action2_buffer, std::vector<dynet::real>& value_buffer);
};