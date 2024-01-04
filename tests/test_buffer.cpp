#include "buffer.hpp"
#include "state.hpp"
#include "action.hpp"

int main()
{
    int error = 0;

    unsigned n = 4, b = 2, n_max = 5;
    Action action(n_max); action.First() = 0; action.Second() = 1;
    State state(n, b, n_max);

    Transition transition(state, action);
    transition.G() = 0.0;

    if ( transition.action.First() != 0 || transition.action.Second() != 1 || transition.g != 0.0 )
        ++error;

    if( transition.state.N() != 4 || transition.state.B() != 2 || transition.state.Nmax() != 5 )
        ++error;
    
    Buffer buffer(100, 1);
    buffer.PushBack(state, action);
    action.First() = 1; action.Second() = 2;
    state.Swap(0, 2);
    buffer.PushBack(state, action);

    
    if ( buffer.data[0].action.First() != 0
            || buffer.data[0].action.Second() != 1
            || buffer.data[1].action.First() != 1
            || buffer.data[1].action.Second() != 2 )
        ++error;
    
    buffer.Backprob(5.0);
    

    action.First() = 1; action.Second() = 1;
    state.Swap(0, 2);
    buffer.PushBack(state, action);
    buffer.Backprob(4.0);

    if ( buffer.data[0].g != 5.0 || buffer.data[1].g != 5.0 || buffer.data[2].g != 4.0 )
        ++error;

    std::vector<dynet::real> state_buffer, action1_buffer, action2_buffer, value_buffer;
    buffer.RandomSample(state_buffer, action1_buffer, action2_buffer, value_buffer);

    if( (action1_buffer[0] == 1 && action1_buffer[1] == 0 && action1_buffer[2] == 0 && action1_buffer[3] == 0 && action1_buffer[4] == 0  
               && action2_buffer[0] == 0 && action2_buffer[1] == 1 && action2_buffer[2] == 0 && action2_buffer[3] == 0 && action2_buffer[4] == 0)
        || (action1_buffer[0] == 0 && action1_buffer[1] == 1 && action1_buffer[2] == 0 && action1_buffer[3] == 0 && action1_buffer[4] == 0  
               && action2_buffer[0] == 0 && action2_buffer[1] == 0 && action2_buffer[2] == 1 && action2_buffer[3] == 0 && action2_buffer[4] == 0)
        || (action1_buffer[0] == 0 && action1_buffer[1] == 1 && action1_buffer[2] == 0 && action1_buffer[3] == 0 && action1_buffer[4] == 0  
               && action2_buffer[0] == 0 && action2_buffer[1] == 1 && action2_buffer[2] == 0 && action2_buffer[3] == 0 && action2_buffer[4] == 0)
      )
    {
        // do nothing
    } else {
        ++error;
    }

    return error;
}