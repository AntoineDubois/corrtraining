#include "reward.hpp"
#include "state.hpp"
#include "utils.hpp"


int main()
{
    int error = 0;
    
    bool isDet = true;
    bool isSum = !isDet;
    Reward rewardSum(isSum);
    Reward rewardDet(isDet);

    unsigned n=3, b=2, n_max=5;
    State state(n, b, n_max);

    rewardSum.Init(state);
    rewardDet.Init(state);
    double init_reward_sum = -state.at(0, 0) -state.at(1, 1) -state.at(0, 1) -state.at(1, 0) 
                          -state.at(2, 2);
    double init_reward_det = state.at(0, 0) * state.at(1, 1) -state.at(0, 1) * state.at(1, 0)
                          + state.at(2, 2);

    
    state.Swap(0, 2);
    rewardSum.Final(state);
    rewardDet.Final(state);
    double final_reward_sum = - state.at(0, 0) - state.at(1, 1) - state.at(0, 1) - state.at(1, 0) 
                          - state.at(2, 2);
    double final_reward_det = state.at(0, 0) * state.at(1, 1) -state.at(0, 1) * state.at(1, 0)
                          + state.at(2, 2);
    

    double true_score_sum = (final_reward_sum / init_reward_sum) -1.0;
    double true_score_det = (final_reward_det / init_reward_det) -1.0;
    double score_sum = rewardSum.Score();
    double score_det = rewardDet.Score();

    
    if ( !equal(score_sum, true_score_sum, 1e-10) )
        ++error;
    if ( !equal(score_det, true_score_det, 1e-10) )
        ++error;
    

    return error;
}