/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "os.h"

#include "debug.h"
#include "plugin.h"

#define MAX_STR_LEN 255

#define NUM_LIBS 6
static char* libNames[NUM_LIBS];/*按类型记录lib名称*/
char* ncclPluginLibPaths[NUM_LIBS];/*按类型记录的lib路径*/
static void* libHandles[NUM_LIBS];/*按类型记录的lib handle*/
/*各类型插件对应的名称*/
static const char* pluginNames[NUM_LIBS] = {"NET", "GIN", "RMA", "TUNER", "PROFILER", "ENV"};
/*各类型plugin对应的默认名称*/
static const char* pluginPrefix[NUM_LIBS] = {"libnccl-net",   "libnccl-gin",      "libnccl-rma",
                                             "libnccl-tuner", "libnccl-profiler", "libnccl-env"};
static const char* pluginFallback[NUM_LIBS] = {"", "", "", "", "", ""};
static unsigned long subsys[NUM_LIBS] = {
  NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_TUNING, NCCL_INIT,
  NCCL_INIT | NCCL_ENV
};

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;/*名称不能为空*/
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;/*静态插件时，名称置为NULL,以方便直接打开*/
  }

  /*打开lib*/
  void* handle = ncclOsDlopen(name);
  if (nullptr == handle) {
    const char* dlErr = ncclOsDlerror();
    if (dlErr) {
      strncpy(errStr, dlErr, MAX_STR_LEN);
      errStr[MAX_STR_LEN] = '\0';
    } else {
      errStr[0] = '\0';
    }
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    if (name && strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static void appendNameToList(char* nameList, int* leftChars, char* name) {
  snprintf(nameList + PATH_MAX - *leftChars, *leftChars, " %s", name);
  *leftChars -= strlen(name) + 1;
}

#if defined(NCCL_OS_LINUX)
/*取lib路径*/
static char* getLibPath(void* handle) {
  struct link_map* lm;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0) return nullptr;
  else return strdup(lm->l_name);
}
#elif defined(NCCL_OS_WINDOWS)
static char* getLibPath(void* handle) {
  char path[PATH_MAX];
  DWORD len = GetModuleFileNameA((HMODULE)handle, path, PATH_MAX);
  if (len == 0 || len >= PATH_MAX) return nullptr;
  return strdup(path);
}
#endif

/*尝试加载指定名称的插件*/
static void* openPluginLib(enum ncclPluginType type, const char* libName/*库名称*/) {
  int openErr, len = PATH_MAX;
  char libName_[MAX_STR_LEN] = {0};
  char openErrStr[MAX_STR_LEN + 1] = {0};
  char eNoEntNameList[PATH_MAX] = {0};

  if (libName && strlen(libName)) {
	  /*指定lib名称的情况*/
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
  } else {
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
  }

  /*设置此type对应的lib handle*/
  libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
  if (libHandles[type]) {
	  /*打开成功，设置此类型对应的lib名称，libpath,返回对应的handle*/
    libNames[type] = strdup(libName_);
    ncclPluginLibPaths[type] = getLibPath(libHandles[type]);/*设置lib路径*/
    return libHandles[type];
  }
  if (openErr == ENOENT) {
	  /*指定的插件不存在*/
    appendNameToList(eNoEntNameList, &len, libName_);
  } else {
	  /*加载插件时发生其它错误*/
    INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
  }

  // libName can't be a relative or absolute path (start with '.' or contain any '/'). It can't be a
  // library name either (start with 'lib' or end with '.so')
  if (libName && strlen(libName) && strchr(libName, '/') == nullptr &&
      (strncmp(libName, "lib", strlen("lib")) || strlen(libName) < strlen(".so") ||
       strncmp(libName + strlen(libName) - strlen(".so"), ".so", strlen(".so")))) {
	  /*libName不能lib开头或者不包含.so,则按类型增加前缀及lib名称再查一次*/
    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);

    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      libNames[type] = strdup(libName_);
      ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
    	/*仍然没有这个文件*/
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
    	/*打开这个文件失败*/
      INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
    }
  }

  if (strlen(eNoEntNameList)) {
    INFO(subsys[type], "%s/Plugin: Could not find:%s%s%s", pluginNames[type], eNoEntNameList,
         (strlen(pluginFallback[type]) > 0 ? ". " : ""), pluginFallback[type]);
  } else if (strlen(pluginFallback[type])) {
    INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], pluginFallback[type]);
  }
  return nullptr;
}

void* ncclOpenNetPluginLib(const char* name) {
	/*打开netplugin*/
  return openPluginLib(ncclPluginTypeNet, name);
}

void* ncclOpenGinPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeGin, name);
}

void* ncclOpenRmaPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeRma, name);
}

void* ncclOpenTunerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeTuner, name);
}

void* ncclOpenProfilerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeProfiler, name);
}

void* ncclOpenEnvPluginLib(const char* name) {
	/*尝试打开env插件*/
  return openPluginLib(ncclPluginTypeEnv, name);
}

/*取得此类型的so handle*/
void* ncclGetGinPluginLib(enum ncclPluginType type) {
  if (libNames[ncclPluginTypeGin]) {
    // increment the reference counter of the gin library
	  /*复用gin信息*/
    libNames[type] = strdup(libNames[ncclPluginTypeGin]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeGin]);
    libHandles[type] = ncclOsDlopen(libNames[ncclPluginTypeGin]);
  }
  return libHandles[type];
}

void* ncclGetNetPluginLib(enum ncclPluginType type) {
  if (libNames[ncclPluginTypeNet]) {
    // increment the reference counter of the net library
	  /*复用net信息*/
    libNames[type] = strdup(libNames[ncclPluginTypeNet]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeNet]);
    libHandles[type] = ncclOsDlopen(libNames[ncclPluginTypeNet]);
  }
  return libHandles[type];
}

ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type) {
  if (handle && libHandles[type] == handle) {
    ncclOsDlclose(handle);
    libHandles[type] = nullptr;
    free(ncclPluginLibPaths[type]);
    ncclPluginLibPaths[type] = nullptr;
    free(libNames[type]);
    libNames[type] = nullptr;
  }
  return ncclSuccess;
}

const char* ncclGetPluginLibName(enum ncclPluginType type) {
  return libNames[type];
}
