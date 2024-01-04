#include "reward.hpp"
#include "state.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>

Reward::Reward(bool isDet): isDet(isDet)
{

}
double Reward::reward(State& state)
{
    unsigned nb_batches = ceil(state.N(), state.B());
    unsigned from, size;
    double r = 0.0;
    for(int i=0; i<nb_batches; ++i)
    {
        from = i * state.B();
        size = from + state.B() <= state.N() ? state.B() : state.N() -from;
        
        r += fct( state.block(from, size) );
    }
    return r;
}
double Reward::Init(State& init_state)
{
    init_reward = reward(init_state);
}
double Reward::Final(State& final_state)
{
    final_reward = reward(final_state);
    return final_reward;
}
double Reward::Score()
{
    return (final_reward/init_reward ) -1.0;
}
double Reward::fct(Eigen::MatrixXd block)
{
    if(isDet){
        return std::pow(block.llt().matrixL().determinant(), 2); // faster determinant with Cholesky decomposition
    }
    return -block.sum();
}