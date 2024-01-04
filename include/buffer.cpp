#include "buffer.hpp"
#include "utils.hpp"
#include "state.hpp"
#include "action.hpp"
#include <vector>
#include <deque>
#include <random>
#include "dynet/dynet.h"


Transition::Transition(State state, Action action): state(state), action(action)
{

}
double& Transition::G()
{
    return g;
}

Buffer::Buffer(unsigned capacity, unsigned batch_size): capacity(capacity), batch_size(batch_size), ep_step(0)
{

}
void Buffer::PushBack(State& state, Action& action)
{
    if( data.size() >= capacity )
        data.pop_front();
    data.push_back( Transition(state, action) );
    ++ep_step;
}
void Buffer::Backprob(double final_reward)
{
    for(int i = data.size()-1; i >= static_cast<int>(data.size() -ep_step); --i)
    {
        data[i].G() = final_reward;
    }
    ep_step = 0;
}
void Buffer::RandomSample(std::vector<dynet::real>& state_buffer, std::vector<dynet::real>& action1_buffer, 
                            std::vector<dynet::real>& action2_buffer, std::vector<dynet::real>& value_buffer)
{
    unsigned index;
    std::uniform_int_distribution<unsigned> distrib(0, data.size() -1);

    state_buffer.clear();
    action1_buffer.clear();
    action2_buffer.clear();
    value_buffer.clear();
    for( int i = 0; i < batch_size; ++i )
    {
        index = distrib(utils_generator);
        data[index].state.Encode(state_buffer);
        data[index].action.Encode(action1_buffer, action2_buffer);
        value_buffer.push_back(data[index].g);
    }
}