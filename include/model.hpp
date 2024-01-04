#pragma once
#include "action.hpp"
#include "state.hpp"
#include "dynet/training.h"
#include "dynet/expr.h"
#include <vector>
#include <utility>

class Model
{
    private:
        unsigned n_max;
        unsigned batch_size;

        dynet::ParameterCollection pc;
        dynet::RMSPropTrainer trainer;
        dynet::ComputationGraph cg;

        dynet::Dim s_dim;
        dynet::Dim v_dim;
        dynet::Dim p1_dim, p2_dim;

        dynet::Expression s;
        dynet::Expression Wh;
        dynet::Expression bh;
        dynet::Expression h;
        
        dynet::Expression Wv;
        dynet::Expression bv;
        dynet::Expression v;
        dynet::Expression v_pred;
        dynet::Expression advantage;

        dynet::Expression Wpi1;
        dynet::Expression bpi1;
        dynet::Expression p1;
        dynet::Expression p1_pred;

        dynet::Expression Wpi2;
        dynet::Expression bpi2;
        dynet::Expression p2;
        dynet::Expression p2_pred;

        dynet::Expression loss;
        dynet::Expression sum_loss;

    public: 
        std::vector<dynet::real> s_value;
        std::vector<dynet::real> v_value;
        std::vector<dynet::real> p1_value;
        std::vector<dynet::real> p2_value;
        
        Model(unsigned n_max, unsigned batch_size);
        std::pair<std::vector<dynet::real>, std::vector<dynet::real> > ProbaPairVec(State& state);
        dynet::real Value(State& state);
        void Backpropagation();
};