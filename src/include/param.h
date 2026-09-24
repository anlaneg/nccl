/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>
#include "compiler.h"

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char* ncclGetEnv(const char* name);

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache);

#define NCCL_PARAM(name/*函数名称*/, env/*环境变量名称(未加前缀）*/, deftVal/*默认值*/) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; /*用于标明是否已cache*/\
    static int8_t noCache = /*uninitialized*/ -1; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; /*用于缓存结果*/\
    if (COMPILER_EXPECT(COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed) == uninitialized, false)) { \
      /*无cache的情况下，通过*/\
      return ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

#endif
