#include "environment.hpp"
#include "state.hpp"
#include "action.hpp"
#include "agent.hpp"
#include "utils.hpp"


int main()
{
    int error = 0;

    unsigned n = 4, b = 2, n_max = 20;
    bool isDet = true;
    unsigned batch_size = 32;
    unsigned capacity = 10'000;

    Agent agent(n_max, isDet, batch_size, capacity);

    Environment env(n_max, &agent);
    env.Episode();

    unsigned t = 0;
    if ( !env.notTerminal(t) )
        ++error;
    t = n_max +1;
    if ( env.notTerminal(t) )
        ++error;

    return error;
}