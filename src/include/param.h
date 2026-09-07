/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define NCCL_PARAM(name/*函数名称*/, env/*环境变量名称(未加前缀）*/, deftVal/*默认值*/) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; /*用于标明是否已cache*/\
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized;/*用于缓存结果*/ \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
    	/*无cache的情况下，通过*/\
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

#endif
