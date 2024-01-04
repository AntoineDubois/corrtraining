#include "agent.hpp"
#include "action.hpp"
#include "state.hpp"

Agent::Agent(unsigned n_max, bool isDet, unsigned batch_size, unsigned capacity): 
                    n_max(n_max), isDet(isDet), buffer(capacity, batch_size)
{
     
}
bool const& Agent::IsDet() const
{
    return isDet;
}
bool const& Agent::IsSto() const
{
    return isSto;
}
Action Agent::Policy(State& state)
{
    return Action(n_max);
}