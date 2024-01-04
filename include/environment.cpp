#include "environment.hpp"
#include "state.hpp"
#include "action.hpp"
#include "agent.hpp"
#include "utils.hpp"
#include <random>
#include <iostream>

Environment::Environment(unsigned n_max, Agent* p_agent): 
                                n_max(n_max), p_agent(p_agent), reward(p_agent -> IsDet())
{
    if (n_max < 10){
        std::cout << "n_max must exceed 10, but received " << n_max << std::endl;
        exit(0);
    }
}
double Environment::Episode()
{
    std::uniform_int_distribution<unsigned> distribN(10, n_max);
    n = distribN(utils_generator);

    std::uniform_int_distribution<unsigned> distribNBatches(2, n/4);
    b = n / distribNBatches(utils_generator);

    State state(n, b, n_max);
    Action action(n_max);
    reward.Init(state);

    unsigned t = 0;
    while (notTerminal(t))
    {
        action = p_agent -> Policy(state);
        if ( p_agent -> IsSto() ){
            p_agent -> buffer.PushBack(state, action);
        }
        //std::cout << "(" << action.First() << ", " << action.Second() << ")" << std::endl;
        Transition(state, action);
        ++t;
    }
    if ( p_agent -> IsSto() ){
        p_agent -> buffer.Backprob( reward.Final(state) );
    } else {
        reward.Final(state);
    }
    //std::cout << "score = " << reward.Score() << std::endl;
    return reward.Score();
}
void Environment::Transition(State& state, Action& action)
{
    state.Swap(action.First(), action.Second());
}
bool Environment::notTerminal(unsigned t)
{
    if (t < n)
        return true;
    return false;
}
