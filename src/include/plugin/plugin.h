/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PLUGIN_H_
#define NCCL_PLUGIN_H_

#include "nccl.h"

enum ncclPluginType {
  ncclPluginTypeNet,
  /*GIN = GPU‑Initiated Networking（GPU 发起的网络通信）
   * GIN 模式：GPU 设备核函数内部直接发起远程
   *  RMA (Put/Get、signal、barrier)，不需要 CPU 介入每一次数据收发。
   * */
  ncclPluginTypeGin,
  ncclPluginTypeRma,
  ncclPluginTypeTuner,
  ncclPluginTypeProfiler,
  ncclPluginTypeEnv,
};

void* ncclOpenNetPluginLib(const char* name);
void* ncclOpenGinPluginLib(const char* name);
void* ncclOpenRmaPluginLib(const char* name);
void* ncclOpenTunerPluginLib(const char* name);
void* ncclOpenProfilerPluginLib(const char* name);
void* ncclOpenEnvPluginLib(const char* name);
void* ncclGetNetPluginLib(enum ncclPluginType type);
void* ncclGetGinPluginLib(enum ncclPluginType type);
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type);

extern char* ncclPluginLibPaths[];
const char* ncclGetPluginLibName(enum ncclPluginType type);

#endif
