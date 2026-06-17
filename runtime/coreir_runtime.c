#include "coreir_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static double current_performance = 0.0;

void coreir_init(void) {
    printf("[CoreIR] Runtime initialized\n");
    current_performance = 0.0;
}

void coreir_shutdown(void) {
    printf("[CoreIR] Runtime shutdown. Final performance: %f\n", current_performance);
}

Differentiable* coreir_diff_register(
    Tensor* (*forward)(void*, Tensor*),
    Tensor* (*backward)(void*, Tensor*),
    void* params,
    size_t param_size
) {
    Differentiable* diff = malloc(sizeof(Differentiable));
    diff->forward = forward;
    diff->backward = backward;
    diff->params = malloc(param_size);
    memcpy(diff->params, params, param_size);
    diff->param_size = param_size;
    return diff;
}

Tensor* coreir_diff_forward(Differentiable* fn, Tensor* input) {
    return fn->forward(fn->params, input);
}

Tensor* coreir_diff_backward(Differentiable* fn, Tensor* grad) {
    return fn->backward(fn->params, grad);
}

SearchSpace* coreir_search_create(double temperature) {
    SearchSpace* space = malloc(sizeof(SearchSpace));
    space->candidates = NULL;
    space->scores = NULL;
    space->num_candidates = 0;
    space->temperature = temperature;
    return space;
}

void coreir_search_add(SearchSpace* space, void* candidate, double score) {
    space->num_candidates++;
    space->candidates = realloc(space->candidates, space->num_candidates * sizeof(void*));
    space->scores = realloc(space->scores, space->num_candidates * sizeof(double));
    space->candidates[space->num_candidates - 1] = candidate;
    space->scores[space->num_candidates - 1] = score;
}

void* coreir_search_sample(SearchSpace* space) {
    if (space->num_candidates == 0) return NULL;
    // Simple temperature-scaled softmax sampling
    double max_score = space->scores[0];
    for (int i = 1; i < space->num_candidates; i++) {
        if (space->scores[i] > max_score) max_score = space->scores[i];
    }
    double sum = 0.0;
    double* probs = malloc(space->num_candidates * sizeof(double));
    for (int i = 0; i < space->num_candidates; i++) {
        probs[i] = exp((space->scores[i] - max_score) / space->temperature);
        sum += probs[i];
    }
    double r = (double)rand() / RAND_MAX * sum;
    double accum = 0.0;
    for (int i = 0; i < space->num_candidates; i++) {
        accum += probs[i];
        if (r <= accum) {
            free(probs);
            return space->candidates[i];
        }
    }
    free(probs);
    return space->candidates[space->num_candidates - 1];
}

void coreir_search_free(SearchSpace* space) {
    free(space->candidates);
    free(space->scores);
    free(space);
}

bool coreir_rewrite_attempt(RewriteRequest* req) {
    if (req->verifier(req->old_code, req->new_code)) {
        printf("[CoreIR] Rewrite verified, applying...\n");
        return true;
    }
    printf("[CoreIR] Rewrite verification failed, rejecting\n");
    return false;
}

void coreir_rewrite_commit(RewriteRequest* req) {
    void* temp = req->old_code;
    req->old_code = req->new_code;
    free(temp);
    printf("[CoreIR] Rewrite committed\n");
}

void coreir_rewrite_rollback(RewriteRequest* req) {
    free(req->new_code);
    printf("[CoreIR] Rewrite rolled back\n");
}

void coreir_learn_step(void) {
    printf("[CoreIR] Learn step executed. Performance: %f\n", current_performance);
    current_performance += 0.01;
}

double coreir_get_performance(void) {
    return current_performance;
}
