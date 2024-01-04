#include "utils.hpp"
#include <vector>
#include "dynet/dynet.h"
#include <random>
#include <limits>
#include <cmath>


unsigned ceil(unsigned n, unsigned b)
{
    if (n % b == 0)
        return n/b;
    return n/b +1;
}
unsigned categorical(std::vector<double>& proba)
{
    std::discrete_distribution<unsigned> distr(proba.begin(), proba.end());
    return distr(utils_generator);
}
unsigned categorical(std::vector<dynet::real>& proba)
{
    std::uniform_real_distribution<dynet::real> distr(0.0, 1.0);
    dynet::real x = distr(utils_generator);
    dynet::real sum = 0.0;
    for( unsigned i = 0; i < proba.size(); ++i )
    {
        sum += proba[i];
        if ( x <= sum )
            return i;
    }
}
unsigned argmax(std::vector<double>& proba)
{
    dynet::real max = proba[0];
    unsigned arg_max = 0;
    for(unsigned i = 1; i < proba.size(); ++i)
    {
        if( max < proba[i] )
        {
            max = proba[i];
            arg_max = i;
        }
    }
    return arg_max;
}
unsigned argmax(std::vector<dynet::real>& proba)
{
    dynet::real max = proba[0];
    unsigned arg_max = 0;
    for(unsigned i = 1; i < proba.size(); ++i)
    {
        if( max < proba[i] )
        {
            max = proba[i];
            arg_max = i;
        }
    }
    return arg_max;
}

bool equal(double x, double y, double tol)
{
    if( std::abs(x -y) < tol )
        return true;
    return false;
}