#pragma once
#include <vector>
#include "dynet/dynet.h"

class Action
{
    private:
        unsigned first = 0, second = 0;
        unsigned n_max;
    public:
        Action(unsigned n_max);
        // Read first and second
        unsigned const& First() const;
        unsigned const& Second() const;
        // Modify first and second
        unsigned& First(); 
        unsigned& Second(); 
        void Encode(std::vector<dynet::real>& p1_vec, std::vector<dynet::real>& p2_vec);
        unsigned WhichBatchFirst(unsigned b);
        unsigned WhichBatchSecond(unsigned b);
};