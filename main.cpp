#include <iostream>

#include "actor_critic.hpp"

int main(int argc, char** argv) 
{
    int error = 0;
    dynet::initialize(argc, argv);
    
    unsigned n_max = 10;
    bool isDet = true;
    unsigned batch_size = 32;
    unsigned capacity = 10'000;
    
    AAC agent(n_max, isDet, batch_size, capacity);
    
    unsigned Ntrain = 100, train_each = 2, display_each = 10, Ntests_display = 1;
    agent.Train(Ntrain, train_each, display_each, Ntests_display);

    
    return 0;
}