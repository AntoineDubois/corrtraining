#include "model.hpp"
#include "action.hpp"
#include "state.hpp"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "utils.hpp"
#include "buffer.hpp"
#include <vector>
#include <utility>


int main(int argc, char** argv)
{
    int error = 0;
    
    dynet::initialize(argc, argv);
    unsigned n = 4, b = 2, n_max = 6;
    unsigned batch_size = 1;
    Model model(n_max, batch_size);

    State state(n, b, n_max);
    std::pair<std::vector<dynet::real>, std::vector<dynet::real> > proba = model.ProbaPairVec(state);

    double sum1 = 0.0, sum2 = 0.0; 
    for(int i = 0; i < n_max; ++i)
    {
        sum1 += proba.first.at(i);
        sum2 += proba.second.at(i);
    }
    if ( !equal(sum1, 1.0, 1e-5) || !equal(sum2, 1.0, 1e-5) )
        ++error;

    double v = model.Value(state);
    if ( v < 0.0 )
        ++error;
    
    unsigned capacity = 100;
    Buffer buffer(capacity, batch_size);
    Action action_b(n_max); action_b.First() = 0; action_b.Second() = 1;
    State state_b(n, b, n_max);
    buffer.PushBack(state_b, action_b);
    action_b.First() = 1; action_b.Second() = 2;
    state_b.Swap(0, 2);
    buffer.PushBack(state_b, action_b);
    double final_reward = 5.0;
    buffer.Backprob(final_reward);
    
    buffer.RandomSample(model.s_value, model.p1_value, model.p2_value, model.v_value);
    model.Backpropagation(); // I am still not convinced that it really works
    
    return error;
}