#include "action.hpp"
#include <vector>
#include "dynet/dynet.h"

Action::Action(unsigned n_max): n_max(n_max)
{

}
unsigned const& Action::First() const
{
    return first;
}
unsigned const& Action::Second() const
{
    return second;
}
unsigned& Action::First()
{
    return first;
}
unsigned& Action::Second()
{
    return second;
}
void Action::Encode(std::vector<dynet::real>& p1_vec, std::vector<dynet::real>& p2_vec)
{
    for(unsigned i=0; i<n_max; ++i)
    {
        if(first == i){
            p1_vec.push_back(1.0);
        }else{
            p1_vec.push_back(0.0);
        }
        if(second == i){
            p2_vec.push_back(1.0);
        }else{
            p2_vec.push_back(0.0);
        }
    }
}
unsigned Action::WhichBatchFirst(unsigned b)
{
    return first / b;
}
unsigned Action::WhichBatchSecond(unsigned b)
{
    return second / b;
}