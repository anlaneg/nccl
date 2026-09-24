/*************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REGISTER_INLINE_H_
#define NCCL_REGISTER_INLINE_H_

#include "comm.h"
#include "register.h"

/*查找<data,data+size>这一段内存对应的regCache*/
static inline ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** outReg) {
  struct ncclRegCache* cache = &comm->regCache;
  *outReg = NULL;
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population) return ncclSuccess;/*达到结尾*/
    struct ncclReg *reg = cache->slots[slot];
    if ((uintptr_t)data < reg->begAddr) return ncclSuccess;
    if ((uintptr_t)data + size <= reg->endAddr) {
      *outReg = reg;/*找到*/
      return ncclSuccess;
    }
  }
}


#endif
