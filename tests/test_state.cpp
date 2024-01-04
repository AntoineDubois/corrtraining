#include "state.hpp"
#include <Eigen/Dense>
#include "dynet/dynet.h"


int main()
{
    int error = 0;

    unsigned n=3, b=2, n_max=4;
    State state(n, b, n_max);

    if( state.N() != 3 )
        ++error;
    if( state.B() != 2 )
        ++error;
    if( state.Nmax() != 4 )
        ++error;
    
    
    Eigen::MatrixXd matrix =  state.block(0, n_max);
    state.Swap(0, 1);
    
    if (
           matrix(0, 0) == state.at(1, 1)
        && matrix(0, 1) == state.at(1, 0)
        && matrix(0, 2) == state.at(1, 2)
        && matrix(0, 3) == state.at(1, 3)

        && matrix(1, 0) == state.at(0, 1)
        && matrix(1, 1) == state.at(0, 0)
        && matrix(1, 2) == state.at(0, 2)
        && matrix(1, 3) == state.at(0, 3)

        && matrix(2, 0) == state.at(2, 1)
        && matrix(2, 1) == state.at(2, 0)
        && matrix(2, 2) == state.at(2, 2)
        && matrix(2, 3) == state.at(2, 3)

        && matrix(3, 0) == state.at(3, 1)
        && matrix(3, 1) == state.at(3, 0)
        && matrix(3, 2) == state.at(3, 2)
        && matrix(3, 3) == state.at(3, 3)
    ){
        // nothing
    } else {
        ++error;
    }
    
    std::vector<dynet::real> p_vec;
    state.Encode(p_vec);
    if ( p_vec.size() != n_max*n_max +1 )
        ++error;
    
    for(unsigned i=0; i<n_max; ++i){
        for(unsigned j=0; j<n_max; ++j){
            if ( p_vec[i*n_max +j] != static_cast<dynet::real>(state.at(i, j)) )
                ++error;
        }
    }

    if ( p_vec[n_max * n_max] != static_cast<dynet::real>(b) )
        ++error;
    
    return error;
}