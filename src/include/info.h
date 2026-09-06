/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;/** 操作符类型 */
  const char* opName;/** 操作符名称 */
  // NCCL Coll Args
  const void* sendbuff;/** 发送数据指针 */
  void* recvbuff;/** 接收数据指针 */
  size_t count;/** 数据数量 */
  /** 数据类型 */
  ncclDataType_t datatype;
  ncclRedOp_t op;
  /** 对端gpu编号（仅在点对点操作中有效） */
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
};

#endif
