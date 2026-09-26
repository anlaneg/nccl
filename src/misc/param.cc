/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "param/param.h"
#include "debug.h"
#include "env.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <mutex>
#include <unordered_set>
#include "os.h"

const char* userHomeDir() {
  return getenv("HOME");
}

/*打开给定的文件，将此文件中的kv行做为环境变量设置*/
void setEnvFile(const char* fileName) {
  FILE* file = fopen(fileName, "r");
  if (file == NULL) return;/*打开文件失败，直接返回*/

  char line[4096];
  char envVar[1024];
  char envValue[1024];
  while (fgets(line, (int)sizeof(line), file) != NULL) {
    size_t len = strlen(line);
    /*以换行符结尾的，指定字符串结尾*/
    if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
    if (len > 0 && line[len - 1] == '\r') line[--len] = '\0';
    if (line[0] == '#') continue;/*跳过注释行*/
    int s = 0;
    while (line[s] != '\0' && line[s] != '=') s++;/*使lines[s]指向'\0'或者指向'='*/
    if (line[s] == '\0') continue;/*这一行有误，没有发现'='符号，忽略*/
    strncpy(envVar, line, std::min(1023, s));/*s指向的是'='即复制变量名称到envVar中*/
    envVar[std::min(1023, s)] = '\0';
    s++;
    strncpy(envValue, line + s, 1023);/*自line+s之后为value*/
    envValue[1023] = '\0';
    ncclOsSetEnv(envVar, envValue);/*设置k,v为环境变量*/
  }
  fclose(file);
}

/*初始化环境变量*/
static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = std::getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
	  /*通过环境变量指明了配置文件的，取配置文件路径*/
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);/*读取此配置文件中有kv对，设置为环境变量*/
  } else {
	  /*没有指定配置文件，按默认的home目录读取配置*/
    const char* userDir = userHomeDir();
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  /*容许etc下的配置覆盖以上设置的环境变量*/
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  static std::once_flag once;
  std::call_once(once, initEnvFunc);
}

static void ncclGetCachePolicy(char const* env, int8_t* noCache) {
  *noCache = ncclParamIsCacheDisabled(env) ? /*noCache*/ 1 : /*cache*/ 0;
}

/*nccl参数加载环境变量（优化了下，保证不每次从env中拿，且均为数字）*/
int64_t ncclLoadParam(char const* env, int64_t deftVal/*默认值*/, int64_t uninitialized/*cache未初始化时值*/, int64_t* cache, int8_t* noCache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // noCache is only load/stored within the mutex, no need for atomic
  if (*noCache == /*uninitialized*/ -1) ncclGetCachePolicy(env, noCache);

  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) != uninitialized) {
    return COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed);
  }

  // Read the environment variable
  const char* str = ncclGetEnv(env);/*取环境变量*/
  int64_t value = deftVal;/*这里用于验证提供的默认值类型兼容*/

  if (str && strlen(str) > 0) {
    errno = 0;
    char* end = nullptr;
    value = strtoll(str, &end, 0);/*转为数字*/
    // Preserve numeric-prefix parsing while rejecting non-numeric values.
    if (errno || end == str) {
      value = deftVal;/*转数字失败，使用默认值*/
      ATTN("Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
    } else {
      INFO(NCCL_ENV, "%s set by environment to %lld.", env, (long long)value);
    }
  }
  /*存为cache(以后就不用从env插件中拿了）*/
  if (*noCache == /*cache*/ 0) COMPILER_ATOMIC_STORE(cache, value, std::memory_order_relaxed);
  return value;
}

/*取环境变量*/
const char* ncclGetEnv(const char* name) {
  ncclInitEnv();/*尝试初始化*/
  return ncclEnvPluginGetEnv(name);/*取环境变量对应的值*/
}
