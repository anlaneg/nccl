/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "shmutils.h"
#include "comm.h"
#include "checks.h"
#include <stdio.h>
#include <stdlib.h>
#include "compiler.h"

/*创建共享内存*/
ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize/*共享内存大小*/, void** shmPtr, void** devShmPtr/*入出参，此值为空时，不映射给gpu；否则返回GPU映射后地址*/,
                         int refcount/*引用计数*/, ncclShmHandle_t* handle/*出参，共享内存详细信息*/) {
  struct ncclShmHandleInternal* internalHandle = NULL;
  ncclResult_t ret = ncclOsShmOpen(shmPath, shmPathSize, shmSize, shmPtr, devShmPtr, refcount, &internalHandle);
  *handle = (ncclShmHandle_t)internalHandle;
  return ret;
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  return ncclOsShmClose((struct ncclShmHandleInternal*)handle);
}

ncclResult_t ncclShmUnlink(ncclShmHandle_t handle) {
  return ncclOsShmUnlink((struct ncclShmHandleInternal*)handle);
}

ncclResult_t ncclShmemAllgather(struct ncclComm* comm, struct ncclShmemCollBuff* shmem, void* sendbuff, void* recvbuff,
                                size_t typeSize) {
  ncclResult_t ret = ncclSuccess;
  int nextRound = shmem->round + 1;
  int curIndex = shmem->round % 2;
  bool done;
  int index = 0;
  size_t maxTypeSize = shmem->maxTypeSize;

  if (comm == NULL || shmem == NULL || sendbuff == NULL || recvbuff == NULL || maxTypeSize < typeSize) {
    ret = ncclInvalidArgument;
    goto exit;
  }

  memcpy((char*)shmem->ptr[curIndex] + comm->localRank * maxTypeSize, sendbuff, typeSize);
  /* reset the previous round and notify I arrive this round */
  COMPILER_ATOMIC_STORE((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * comm->localRank), nextRound,
                        std::memory_order_release);

  do {
    done = true;
    for (int i = index; i < comm->localRanks; ++i) {
      if (i != comm->localRank && COMPILER_ATOMIC_LOAD((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * i),
                                                       std::memory_order_acquire) < nextRound) {
        done = false;
        index = i;
        break;
      }
    }
  } while (!done);

  for (int i = 0; i < comm->localRanks; ++i) {
    memcpy((uint8_t*)recvbuff + i * typeSize, (uint8_t*)shmem->ptr[curIndex] + i * maxTypeSize, typeSize);
  }
  shmem->round = nextRound;

exit:
  return ret;
}
