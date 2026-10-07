/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "register.h"
#include "transport.h"

ncclResult_t ncclRegisterP2pNetBuffer(
  struct ncclComm* comm, void* userbuff/*buffer地址*/, size_t size/*buffer长度*/, struct ncclConnector* conn, int* regFlag, void** handle,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  ncclResult_t ret = ncclSuccess;

  *regFlag = 0;
  if (comm->netDeviceType != NCCL_NET_DEVICE_UNPACK) {
    if (comm->planner.persistent && ncclParamGraphRegister()) {
    	/*为此连接主动注册buffer*/
      ncclNetGraphRegisterBuffer(comm, userbuff, size, &conn, 1/*仅一个连接*/, regFlag, handle, cleanupQueue, NULL);
    }
    if (*regFlag == 0/*未注册成功*/ && ncclParamLocalRegister()) {
    	/*主动注册不成功，检查是否在一个更大的buffer中此段已被注册*/
      ncclNetLocalRegisterBuffer(comm, userbuff, size, &conn, 1, regFlag, handle);
    }
  }
  return ret;
}

ncclResult_t ncclRegisterP2pIpcBuffer(
  struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  ncclResult_t ret = ncclSuccess;
  uintptr_t offset = 0;
  uintptr_t* peerRmtAddrs = NULL;

  *regFlag = 0;
  if (comm->planner.persistent && ncclParamGraphRegister()) {
    ncclIpcGraphRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs,
                               reinterpret_cast<void*>(cleanupQueue), NULL);
  }
  if (*regFlag == 0 && ncclParamLocalRegister()) {
    ncclIpcLocalRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs);
  }

  if (*regFlag) *regAddr = (void*)((uintptr_t)peerRmtAddrs + offset);
  return ret;
}
