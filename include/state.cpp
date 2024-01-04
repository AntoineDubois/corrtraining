#include "state.hpp"
#include <Eigen/Dense>
#include "dynet/dynet.h"


State::State(unsigned n, unsigned b, unsigned n_max, bool isGen): values(n_max, n_max), n(n), b(b), n_max(n_max)
{
    if(isGen){
        values.block(0,0, n, n).setRandom();
        values.block(0,0, n, n) = values.block(0,0, n, n).transpose() * values.block(0,0, n, n); 
    }
}
unsigned const& State::N() const
{
    return n;
}
unsigned const& State::B() const
{
    return b;
}
unsigned const& State::Nmax() const
{
    return n_max;
}
void State::Swap(unsigned i, unsigned j)
{
    Eigen::MatrixXd buffer = values.col(i);
    values.col(i) = values.col(j);
    values.col(j) = buffer;

    buffer = values.row(i);
    values.row(i) = values.row(j);
    values.row(j) = buffer;
}
void State::Encode(std::vector<dynet::real>& p_vec)
{
    for(unsigned i=0; i<n_max; ++i){
        for(unsigned j=0; j<n_max; ++j){
            p_vec.push_back(values(i, j));
        }
    }
    p_vec.push_back(static_cast<dynet::real>(b));
}
double State::at(unsigned i, unsigned j)
{
    return values(i,j);
}
Eigen::MatrixXd State::block(unsigned from, unsigned size)
{
    return values.block(from, from, size, size);
}