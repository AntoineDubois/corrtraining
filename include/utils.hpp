#pragma once
#include <vector>
#include "dynet/dynet.h"
#include <limits>
#include <random>

static std::random_device utils_rd;
static std::mt19937 utils_generator(utils_rd());

unsigned ceil(unsigned n, unsigned b);

unsigned categorical(std::vector<double>& proba);
unsigned categorical(std::vector<dynet::real>& proba);

unsigned argmax(std::vector<double>& proba);
unsigned argmax(std::vector<dynet::real>& proba);

bool equal(double x, double y, double tol = std::numeric_limits< double >::min() );