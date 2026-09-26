/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <errno.h>
#include <stdlib.h>
#include <mutex>
#include <atomic>

#include "checks.h"
#include "debug.h"
#include "env.h"
#include "param.h"
#include "plugin.h"

extern ncclEnv_t* getNcclEnv_v2(void* lib);
extern ncclEnv_t* getNcclEnv_v1(void* lib);

static void* envPluginLib = nullptr;
static ncclEnv_t* ncclEnvPlugin = nullptr;
extern ncclEnv_v2_t ncclIntEnv_v2;

#define EXT_ENV_PLUGIN 0
#define INT_ENV_PLUGIN 1
#define NUM_ENV_PLUGIN 2
/*用于获取环境变量的插件*/
static ncclEnv_t* ncclEnvPlugins[NUM_ENV_PLUGIN] = {nullptr/*容许定制（扩展）的插件位置*/, &ncclIntEnv_v2/*默认插件*/ };

enum {
  envPluginLoadFailed = -1,
  envPluginLoadReady = 0,
  envPluginLoadSuccess = 1,
};
static int envPluginStatus = envPluginLoadReady;

/*加载用于获取env的插件*/
static ncclResult_t ncclEnvPluginLoad(void) {
  const char* envName = nullptr;
  /*之前尝试过，已失败，直接返回不再尝试*/
  bool envPluginRequested = false;
  if (envPluginStatus != envPluginLoadReady) goto exit;

  if ((envName = std::getenv("NCCL_ENV_PLUGIN")) != nullptr) {
	  /*设置了env插件*/
    envPluginRequested = true;
    INFO(NCCL_ENV, "NCCL_ENV_PLUGIN set by environment to %s", envName);
    if (strcasecmp(envName, "none") == 0) {
      goto fail;/*env插件名称不得为none*/
    }
  }
  /*打开env插件*/
  envPluginLib = ncclOpenEnvPluginLib(envName);
  if (nullptr == envPluginLib) {
    goto fail;/*加载失败*/
  } else if (ncclPluginLibPaths[ncclPluginTypeEnv]) {
	  /*取lib路径*/
    envName = ncclPluginLibPaths[ncclPluginTypeEnv];
  }

  /*取插件操作api结构体*/
  ncclEnvPlugins[EXT_ENV_PLUGIN] = getNcclEnv_v2(envPluginLib);
  if (ncclEnvPlugins[EXT_ENV_PLUGIN] == nullptr) {
    ncclEnvPlugins[EXT_ENV_PLUGIN] = getNcclEnv_v1(envPluginLib);
    if (ncclEnvPlugins[EXT_ENV_PLUGIN] == nullptr) {
      if (envPluginRequested) ATTN("External env plugin %s is unsupported", envName);
      else INFO(NCCL_INIT, "External env plugin %s is unsupported", envName);
      goto fail;
    }
  }
  INFO(NCCL_INIT, "Successfully loaded external env plugin %s", envName);

  envPluginStatus = envPluginLoadSuccess;/*通过环境变量加载插件成功*/

exit:
  return ncclSuccess;
fail:
  // Fallback to internal/default plugin
  if (envPluginLib) NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  envPluginLib = nullptr;
  envPluginStatus = envPluginLoadFailed;/*加载env插件失败*/
  goto exit;/*只设置加载状态为失败，仍跳exit返回ncclSuccess*/
}

static ncclResult_t ncclEnvPluginUnload(void) {
  if (ncclEnvPlugin) {
    INFO(NCCL_DESTROY, "ENV/Plugin: Closing env plugin %s", ncclEnvPlugin->name);
  }
  if (ncclEnvPlugins[EXT_ENV_PLUGIN]) {
    ncclEnvPlugin = ncclEnvPlugins[INT_ENV_PLUGIN];
    ncclEnvPlugins[EXT_ENV_PLUGIN] = nullptr;
  }
  NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void);

static bool initialized;

ncclResult_t ncclEnvPluginInit(void) {
  /*初始化环境变量*/
  initEnv();
  /*加载env插件*/
  NCCLCHECK(ncclEnvPluginLoad());
  /*如果evn插件加载成功，则表示可以用扩展的env插件，否则用默认的env插件*/
  ncclEnvPlugin =
    (envPluginLoadSuccess == envPluginStatus) ? ncclEnvPlugins[EXT_ENV_PLUGIN] : ncclEnvPlugins[INT_ENV_PLUGIN]/*未成功，返回默认*/;
  /*env插件初始化*/
  NCCLCHECK(ncclEnvPlugin->init(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_SUFFIX, ncclDebugLog));
  atexit(ncclEnvPluginFinalize);
  /*指明已初始化*/
  COMPILER_ATOMIC_STORE(&initialized, true, std::memory_order_release);
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void) {
  if (ncclEnvPlugin->finalize) {
    ncclEnvPlugin->finalize();
    ncclEnvPluginUnload();
  }
}

/*通过env插件获取环境变量（env插件有默认插件实现）*/
const char* ncclEnvPluginGetEnv(const char* name) {
  return ncclEnvPlugin->getEnv(name);
}

bool ncclEnvPluginInitialized(void) {
  return COMPILER_ATOMIC_LOAD(&initialized, std::memory_order_acquire);
}
