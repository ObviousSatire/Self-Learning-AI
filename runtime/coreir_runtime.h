#ifndef COREIR_RUNTIME_H
#define COREIR_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>

// Core types for self-learning runtime

typedef struct {
    void* data;
    size_t size;
    int* shape;
    int ndim;
} Tensor;

typedef struct {
    Tensor* (*forward)(void* params, Tensor* input);
    Tensor* (*backward)(void* params, Tensor* grad_output);
    void* params;
    size_t param_size;
} Differentiable;

typedef struct {
    void** candidates;
    double* scores;
    int num_candidates;
    double temperature;
} SearchSpace;

typedef struct {
    void* old_code;
    void* new_code;
    bool (*verifier)(void* old_code, void* new_code);
} RewriteRequest;

// Core API
void coreir_init(void);
void coreir_shutdown(void);

// Diff primitive
Differentiable* coreir_diff_register(
    Tensor* (*forward)(void*, Tensor*),
    Tensor* (*backward)(void*, Tensor*),
    void* params,
    size_t param_size
);
Tensor* coreir_diff_forward(Differentiable* fn, Tensor* input);
Tensor* coreir_diff_backward(Differentiable* fn, Tensor* grad);

// Search primitive
SearchSpace* coreir_search_create(double temperature);
void coreir_search_add(SearchSpace* space, void* candidate, double score);
void* coreir_search_sample(SearchSpace* space);
void coreir_search_free(SearchSpace* space);

// Rewrite primitive
bool coreir_rewrite_attempt(RewriteRequest* req);
void coreir_rewrite_commit(RewriteRequest* req);
void coreir_rewrite_rollback(RewriteRequest* req);

// Self-learning loop
void coreir_learn_step(void);
double coreir_get_performance(void);

#endif
