#include "agent.hpp"
#include "action.hpp"
#include "state.hpp"


int main()
{
    int error = 0;

    unsigned n = 4, b = 2, n_max = 5;
    bool isDet = true;
    unsigned batch_size = 32;
    unsigned capacity = 10'000;

    Agent agent(n_max, isDet, batch_size, capacity);

    if ( agent.IsDet() != isDet )
        ++error;
    if ( agent.IsSto() != true )
        ++error;
    
    State state(n, b, n_max);
    Action action = agent.Policy(state);
    if ( action.First() != 0 || action.Second() != 0 )
        ++error;

    return error;
}