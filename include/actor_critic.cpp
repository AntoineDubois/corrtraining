#include "actor_critic.hpp"
#include "agent.hpp"
#include <algorithm>
#include <iostream>
#include "action.hpp"
#include "state.hpp"
#include "environment.hpp"
#include "dynet/dynet.h"
#include "utils.hpp"

AAC::AAC(unsigned n_max, bool isDet, unsigned batch_size, unsigned capacity): 
                        Agent(n_max, isDet, batch_size, capacity),
                        env(n_max, this), model(n_max, batch_size)
{

}
bool const& AAC::IsSto() const
{
    return isSto;
}
bool& AAC::IsSto()
{
    return isSto;
}
void AAC::Train(unsigned Ntrain, unsigned train_each, unsigned display_each, unsigned Ntests_display)
{
    isSto = true;
    for(unsigned i = 1; i <= Ntrain; ++i)
    {
        env.Episode();
        if ( i % train_each == 0 ){
            Backpropagation();
        }
            
        if( i % display_each == 0 ){
            double G = 0.0;
            G += Test(Ntests_display);
            std::cout << "After " << i << " training loops, av G=" << G/static_cast<double>(Ntests_display) << std::endl;
            isSto = true;
        }
    }
}
double AAC::Test(unsigned Ntests)
{
    isSto = false;
    double G, sum = 0.0;
    std::cout << "G: ";
    for(unsigned i = 1; i <= Ntests; ++i)
    {
        G = env.Episode();
        sum += G;
        std::cout << G;
        if ( i % 15 == 0 ){
            std::cout << "\n";
        } else if ( i == Ntests ) {
            std::cout << std::endl;
        } else {
            std::cout << ", ";
        }
    }
    return sum;
}
Action AAC::Policy(State& state)
{
    std::pair<std::vector<dynet::real>, std::vector<dynet::real> > proba = model.ProbaPairVec(state);
    if (isSto)
        return PolicySto(state, proba.first, proba.second);
    return PolicyGreedy(state, proba.first, proba.second);
}
Action AAC::PolicySto(State& state, std::vector<dynet::real>& proba1, std::vector<dynet::real>& proba2)
{
    Action action(n_max);
    dynet::real sum = 0.0;
    for(unsigned i = state.N(); i < state.Nmax(); ++i)
    {
        sum += proba1[i];
        proba1[i] = dynet::real(0.0);
    }
    sum = dynet::real(1.0) -sum;
    for(unsigned i = 0; i < state.N(); ++i)
    {
        proba1[i] /= sum;
    }

    sum = 0.0;
    for( unsigned i =0; i< state.Nmax(); ++i)
    {
        sum += proba1[i];
    }
    proba1.resize(state.N());
    action.First() = categorical(proba1);
    
    unsigned from = action.WhichBatchFirst(state.B()) * state.B();
    unsigned until = from + state.B() <= state.N() ? from + state.B() : state.N();
    
    sum = 0.0;
    for(unsigned i = from; i < until; ++i)
    {
        sum += proba2[i];
        proba2[i] = dynet::real(0.0);
    }
    for(unsigned i = state.N(); i < state.Nmax(); ++i)
    {
        sum += proba2[i];
        proba2[i] = dynet::real(0.0);
    }
    sum = dynet::real(1.0) -sum;
    for(unsigned i = 0; i < from; ++i)
    {
        proba2[i] /= sum;
    }
    for(unsigned i = until; i < state.N(); ++i)
    {
        proba2[i] /= sum;
    }

    proba2.resize(state.N());
    action.Second() = categorical(proba2);

    return action;
}
Action AAC::PolicyGreedy(State& state, std::vector<dynet::real>& proba1, std::vector<dynet::real>& proba2)
{
    Action action(n_max);
    proba1.resize(state.N()); // keep the the probabilities from 0 to N-1
    proba2.resize(state.N());

    action.First() = argmax(proba1);
    
    unsigned from = action.WhichBatchFirst(state.B()) * state.B();
    unsigned until = from + state.B() <= state.N() ? from + state.B() : state.N();
    for(unsigned i = from; i < until; ++i)
    {
        proba2.at(i) = 0.0;
    }
    action.Second() = argmax(proba2);

    return action;
}
void AAC::Backpropagation()
{
    buffer.RandomSample(model.s_value, model.p1_value, model.p2_value, model.v_value);
    model.Backpropagation();
}