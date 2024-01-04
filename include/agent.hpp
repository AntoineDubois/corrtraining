#pragma once
#include "state.hpp"
#include "action.hpp"
#include "buffer.hpp"

class Agent 
{
    protected:
        unsigned n_max;
        bool isDet;
        bool isSto = true;
    public:
        Buffer buffer;
        Agent(unsigned n_max, bool isDet, unsigned batch_size = 64, unsigned capacity = 10'000);
        virtual Action Policy(State& state);
        bool const& IsDet() const;
        bool const& IsSto() const;
};