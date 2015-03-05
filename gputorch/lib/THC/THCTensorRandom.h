#ifndef TH_GPU_TENSOR_RANDOM_INC
#define TH_GPU_TENSOR_RANDOM_INC

#include "THCTensor.h"
#include <random>

/* Generator */
typedef struct _Generator {
  std::mt19937* gen_states;
  int initf;
  unsigned long initial_seed;
} Generator;

typedef struct THGPURNGState {
  /* One generator per GPU */
  Generator* gen;
  Generator* current_gen;
  int num_devices;
} THGPURNGState;

THC_API void THCRandom_init(THGPURNGState* state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(THGPURNGState* state);
THC_API void THCRandom_setGenerator(THGPURNGState* state, int device);
THC_API void THCRandom_resetGenerator(THGPURNGState* state);
THC_API unsigned long THCRandom_seed(THGPURNGState* state);
THC_API unsigned long THCRandom_seedAll(THGPURNGState* state);
THC_API void THCRandom_manualSeed(THGPURNGState* state, unsigned long the_seed_);
THC_API void THCRandom_manualSeedAll(THGPURNGState* state, unsigned long the_seed_);
THC_API unsigned long THCRandom_initialSeed(THGPURNGState* state);
THC_API void THCRandom_getRNGState(THGPURNGState* state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(THGPURNGState* state, THByteTensor *rng_state);
THC_API void THGPUTensor_geometric(THGPURNGState* state, THGPUTensor *self, double p);
THC_API void THGPUTensor_bernoulli(THGPURNGState* state, THGPUTensor *self, double p);
THC_API void THGPUTensor_uniform(THGPURNGState* state, THGPUTensor *self, double a, double b);
THC_API void THGPUTensor_normal(THGPURNGState* state, THGPUTensor *self, double mean, double stdv);
THC_API void THGPUTensor_exponential(THGPURNGState* state, THGPUTensor *self, double lambda);
THC_API void THGPUTensor_cauchy(THGPURNGState* state, THGPUTensor *self, double median, double sigma);
THC_API void THGPUTensor_logNormal(THGPURNGState* state, THGPUTensor *self, double mean, double stdv);

#endif

