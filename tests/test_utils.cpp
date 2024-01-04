#include "utils.hpp"
#include <vector>
#include "dynet/dynet.h"

int main()
{
    int error = 0;

    if ( ceil(6, 2) != 3 )
        ++error;
    if ( ceil(6, 4) != 2 )    
        ++error;

    std::vector<dynet::real> proba0r {1, 0, 0};
    if ( categorical(proba0r) != 0 )
        ++error;

    std::vector<dynet::real> proba1r {0, 1, 0};
    if ( categorical(proba1r) != 1 )
        ++error;
    
    std::vector<dynet::real> proba2r {0, 0, 1};
    if ( categorical(proba2r) != 2 )
        ++error;
    
    if ( argmax(proba0r) != 0)
        ++error;
    
    if ( argmax(proba1r) != 1)
        ++error;
    
    if ( argmax(proba2r) != 2)
        ++error;
    

    std::vector<double> proba0d {1, 0, 0};
    if ( categorical(proba0d) != 0 )
        ++error;

    std::vector<double> proba1d {0, 1, 0};
    if ( categorical(proba1d) != 1 )
        ++error;
    
    std::vector<double> proba2d {0, 0, 1};
    if ( categorical(proba2d) != 2 )
        ++error;
    
    if ( argmax(proba0d) != 0)
        ++error;
    
    if ( argmax(proba1d) != 1)
        ++error;
    
    if ( argmax(proba2d) != 2)
        ++error;


    if ( !equal(0.0, 0.0) )
        ++error;
    if ( !equal(0.1, 0.0, 0.5) )
        ++error;

    return error;
}