#include "action.hpp"
#include <vector>
#include "dynet/dynet.h"

int main()
{
    int error = 0;

    unsigned n_max = 3;
    Action action(n_max);

    action.First() = 1;
    action.Second() = 2;

    if ( action.First() != 1 )
        ++error;
    if ( action.Second() != 2 )
        ++error;
    
    std::vector<dynet::real> p1_vec, p2_vec;
    action.Encode(p1_vec, p2_vec);
    if( p1_vec.size() != n_max || p2_vec.size() != n_max )
        ++error;
    if( p1_vec[0] != 0.0 || p1_vec[1] != 1.0 || p1_vec[2] != 0.0)
        ++error;
    if( p2_vec[0] != 0.0 || p2_vec[1] != 0.0 || p2_vec[2] != 1.0)
        ++error;
    
    unsigned b = 2;
    if ( action.WhichBatchFirst(b) != 0 )
        ++error;
    if ( action.WhichBatchSecond(b) != 1 )
        ++error;

    return error;
}