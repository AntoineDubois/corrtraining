#include "model.hpp"
#include "action.hpp"
#include "state.hpp"
#include "dynet/training.h"
#include "dynet/expr.h"
#include <vector>
#include <utility>

Model::Model(unsigned n_max, unsigned batch_size): n_max(n_max), batch_size(batch_size),
            trainer(pc, 0.000000000),
            s_value(n_max * n_max * batch_size), p1_value(n_max * batch_size), p2_value(n_max * batch_size), v_value(1 * batch_size),
            s_dim({n_max * n_max}, batch_size),
            p1_dim({n_max, 1}, batch_size),
            p2_dim({n_max, 1}, batch_size), 
            v_dim({1}, batch_size)
{
    unsigned input_size = n_max * n_max;
    unsigned hidden_size = input_size/2;
    
    s = dynet::input(cg, s_dim, &s_value);
    Wh = dynet::parameter(cg, pc.add_parameters({hidden_size, input_size}));
    bh = dynet::parameter(cg, pc.add_parameters({hidden_size, 1}));
    h = dynet::rectify(Wh * s + bh); // rectify = ReLU

    
    Wv = dynet::parameter(cg, pc.add_parameters({1, hidden_size}));
    bv = dynet::parameter(cg, pc.add_parameters({1}));
    
    v = dynet::input(cg, v_dim, &v_value);
    v_pred = dynet::rectify(Wv * h + bv);
    advantage = v -v_pred;

    Wpi1 = dynet::parameter(cg, pc.add_parameters({n_max, hidden_size}));
    bpi1 = dynet::parameter(cg, pc.add_parameters({n_max}));
    
    p1 = dynet::input(cg, p1_dim, &p1_value);
    p1_pred = dynet::softmax(Wpi1 * h + bpi1);

    Wpi2 = dynet::parameter(cg, pc.add_parameters({n_max, hidden_size}));
    bpi2 = dynet::parameter(cg, pc.add_parameters({n_max}));
    
    p2 = dynet::input(cg, p2_dim, &p2_value);
    p2_pred = dynet::softmax(Wpi2 * h + bpi2);

    loss = dynet::dot_product( p1, p1_pred ); // p1 = {6}, p1_pred = {6, 1}
    loss = dynet::square(advantage) - advantage * (dynet::dot_product( p1, dynet::log(p1_pred) ) +
                                                  dynet::dot_product( p2, dynet::log(p2_pred) ) );
    //loss = dynet::square(advantage) - advantage * ( p1 * dynet::log(p1_pred) +  p2 * dynet::log(p2_pred) );
    sum_loss = dynet::sum_batches(loss);
}
std::pair<std::vector<dynet::real>, std::vector<dynet::real> > Model::ProbaPairVec(State& state)
{
    state.Encode(s_value);
    return std::pair<std::vector<dynet::real>, std::vector<dynet::real> >( dynet::as_vector(cg.forward(p1_pred)), 
                                                                                dynet::as_vector(cg.forward(p2_pred)) );
}
dynet::real Model::Value(State& state)
{
    state.Encode(s_value);
    return dynet::as_scalar(cg.forward(v_pred));
}
void Model::Backpropagation()
{
    dynet::real l = dynet::as_scalar(cg.forward(sum_loss));
    std::cout << "loss =" << l << std::endl;
    
    cg.backward(sum_loss);
    trainer.update();

    s_value.clear();
    v_value.clear();
    p1_value.clear();
    p2_value.clear();
}