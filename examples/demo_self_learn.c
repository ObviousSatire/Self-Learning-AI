#include "../runtime/coreir_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple demo: learn to approximate a linear function y = wx + b
typedef struct {
    double w;
    double b;
} LinearParams;

Tensor* linear_forward(void* params, Tensor* input) {
    LinearParams* p = (LinearParams*)params;
    Tensor* output = malloc(sizeof(Tensor));
    output->size = input->size;
    output->data = malloc(input->size * sizeof(double));
    double* in_data = (double*)input->data;
    double* out_data = (double*)output->data;
    for (size_t i = 0; i < input->size; i++) {
        out_data[i] = p->w * in_data[i] + p->b;
    }
    return output;
}

Tensor* linear_backward(void* params, Tensor* grad) {
    // Simplified: return grad scaled by w
    LinearParams* p = (LinearParams*)params;
    Tensor* output = malloc(sizeof(Tensor));
    output->size = grad->size;
    output->data = malloc(grad->size * sizeof(double));
    double* g = (double*)grad->data;
    double* o = (double*)output->data;
    for (size_t i = 0; i < grad->size; i++) {
        o[i] = g[i] * p->w;
    }
    return output;
}

bool always_verify(void* old_code, void* new_code) {
    return true; // Trust everything for demo
}

int main() {
    printf("=== CoreIR Self-Learning Demo ===\n\n");
    coreir_init();

    // 1. DIFF: Register a differentiable function
    printf("1. Registering differentiable function (linear model)...\n");
    LinearParams params = {2.0, 1.0}; // y = 2x + 1
    Differentiable* model = coreir_diff_register(
        linear_forward, linear_backward, &params, sizeof(params)
    );
    printf("   Model registered: w=%.1f, b=%.1f\n\n", params.w, params.b);

    // 2. SEARCH: Create a search space and sample
    printf("2. Searching over parameter space...\n");
    SearchSpace* space = coreir_search_create(0.5);
    
    LinearParams candidate1 = {1.5, 0.8};
    LinearParams candidate2 = {2.1, 1.1};
    LinearParams candidate3 = {1.9, 0.9};
    
    LinearParams* c1 = malloc(sizeof(LinearParams)); *c1 = candidate1;
    LinearParams* c2 = malloc(sizeof(LinearParams)); *c2 = candidate2;
    LinearParams* c3 = malloc(sizeof(LinearParams)); *c3 = candidate3;
    
    coreir_search_add(space, c1, 0.7);
    coreir_search_add(space, c2, 0.9);
    coreir_search_add(space, c3, 0.8);
    
    LinearParams* best = (LinearParams*)coreir_search_sample(space);
    printf("   Best candidate sampled: w=%.2f, b=%.2f\n\n", best->w, best->b);

    // 3. REWRITE: Attempt a safe code modification
    printf("3. Attempting safe rewrite...\n");
    RewriteRequest req;
    req.old_code = model->params;
    LinearParams new_params = {2.5, 1.2};
    req.new_code = malloc(sizeof(LinearParams));
    memcpy(req.new_code, &new_params, sizeof(LinearParams));
    req.verifier = always_verify;
    
    if (coreir_rewrite_attempt(&req)) {
        coreir_rewrite_commit(&req);
        LinearParams* current = (LinearParams*)model->params;
        printf("   Model updated: w=%.1f, b=%.1f\n\n", current->w, current->b);
    }

    // 4. LEARN: Run a learning step
    printf("4. Running self-learning loop...\n");
    for (int i = 0; i < 3; i++) {
        coreir_learn_step();
    }
    printf("   Final performance score: %.3f\n\n", coreir_get_performance());

    // Cleanup
    coreir_search_free(space);
    coreir_shutdown();
    
    printf("=== Demo complete ===\n");
    return 0;
}
