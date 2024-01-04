#include "actor_critic.hpp"
#include "agent.hpp"
#include <algorithm>
#include <iostream>
#include "action.hpp"
#include "state.hpp"
#include "environment.hpp"
#include "dynet/dynet.h"
#include "utils.hpp"

int main(int argc, char** argv)
{
    int error = 0;

    dynet::initialize(argc, argv);
    
    unsigned n_max = 10;
    bool isDet = true;
    unsigned batch_size = 32;
    unsigned capacity = 10'000;

    AAC agent(n_max, isDet, batch_size, capacity);

    unsigned n=5, b=3;
    State state(n, b, n_max);

    agent.IsSto() = false;
    Action action = agent.Policy(state);

    if ( action.First() >= n || action.Second() >= n )
        ++error; 
    if ( action.WhichBatchFirst(b) == action.WhichBatchSecond(b) )
        ++error;
    

    agent.IsSto() = true;
    action = agent.Policy(state);

    if ( action.WhichBatchFirst(b) == action.WhichBatchSecond(b) )
        ++error;
    if ( action.First() >= n || action.Second() >= n )
        ++error;

    unsigned Ntests = 2;
    agent.Test(Ntests);

    unsigned Ntrain = 5, train_each = 4, display_each = 4, Ntests_display = 1;
    agent.Train(Ntrain, train_each, display_each, Ntests_display);

    return 0;
}