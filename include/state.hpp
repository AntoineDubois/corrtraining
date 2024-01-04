#pragma once
#include <Eigen/Dense>
#include "dynet/dynet.h"

class State
{
    private:
        Eigen::MatrixXd values;
        unsigned n, b, n_max;
    public:
        State(unsigned n, unsigned b, unsigned n_max, bool isGen = true);
        
        unsigned const& N() const;
        unsigned const& B() const;
        unsigned const& Nmax() const;

        void Swap(unsigned i, unsigned j);
        void Encode(std::vector<dynet::real>& p_vec);
        double at(unsigned i, unsigned j);
        Eigen::MatrixXd block(unsigned from, unsigned size);
};