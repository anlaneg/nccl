/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "comm.h"
#include "allocator.h"
#include "register.h"
#include "utils.h"

#define NCCL_COMM_GROUP_INVALID 0x01

ncclResult_t ncclGroupErrCheck(ncclResult_t ret);
void ncclGroupCommJoin(struct ncclComm* comm, int type);
void ncclGroupCommPreconnect(struct ncclComm* comm);
ncclResult_t ncclGroupCommLeave(struct ncclComm* comm);
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob);
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob);

typedef ncclResult_t (*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank,
                           int cudaDev);

ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                             void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*), ncclComm_t comm);

struct ncclGroupJob {
  struct ncclAsyncJob base;
  int groupRefCount;/*被外部引用的引数*/
  bool nonBlockingInit;/*是否非阻塞初始化*/
  bool joined;
  /**按类型划分的任务链表头（放在此对列的会被并行执行） */
  struct ncclComm* groupCommHead[ncclGroupTaskTypeNum];
  /*记录preConnect类任务的链表头节点（放在此对列会被并行执行）*/
  struct ncclComm* groupCommPreconnectHead;
  ncclResult_t groupError;
  bool abortFlag;
  /*执行时，先存放ncclAsyncJobs，之后groupCommPreconnectHead也会被转换为ncclPreconnectJob存放进来*/
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncJobs;
};

ncclResult_t ncclCollPreconnect(struct ncclComm* comm, bool* algoNeedConnect);

ncclResult_t doLaunches(struct ncclComm* head, int taskType = ncclGroupTaskTypeCollective);

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL/**如果不传参，此值为NULL */);
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////

extern thread_local int ncclGroupDepth; // depth of ncclGroupStart nesting
extern thread_local ncclResult_t ncclGroupError;
extern thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum];
extern thread_local struct ncclComm* ncclGroupCommPreconnectHead;
extern thread_local int ncclGroupBlocking;
extern thread_local struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;/**增加ncclGroupDepth，表示当前线程进入了一个新的group */
  return ncclSuccess;
}

inline bool ncclGroupEnabled() {
  return ncclGroupDepth != 0;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
	  /*depth大于0，则在当前在某一个group内,如有错误，则置error*/
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;
}

// Add comm to this thread's group
inline void ncclGroupCommJoin(struct ncclComm* comm, int type) {
  if (comm->groupNext[type] == reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID)) {
    // Insert comm into ncclGroupCommHead adjacent to sibling comms. This preserves
    // the users program order yet insures siblings occur consecutively. This
    // is required by doLaunches() in "group.cc".
    struct ncclComm** pp = &ncclGroupCommHead[type];
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0) pp = &(*pp)->groupNext[type];

    // didn't find its clique, we need to insert it with ascending order based on commHash
    if (*pp == nullptr) {
      pp = &ncclGroupCommHead[type];
      while (*pp != nullptr && (*pp)->commHash < comm->commHash) pp = &(*pp)->groupNext[type];
    }
    comm->groupNext[type] = *pp;
    *pp = comm;
    // Comms gets a new memory stack scope upon joining. Each task batched for
    // this comm is allocated there.
    if (type == ncclGroupTaskTypeCollective || type == ncclGroupTaskTypeRawTask) {
      // Initialize planner
      ncclMemoryStackPush(&comm->memScoped);
      ncclKernelPlanner::Peer* tmp = comm->planner.peers;
      ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* tmpRmaQueues = comm->planner.rmaTaskQueues;
      int numRmaCtx = comm->config.numRmaCtx;
      memset(&comm->planner, 0, sizeof(comm->planner));
      comm->planner.peers = tmp;
      comm->planner.bcast_info.minBcastPeer = INT_MAX;
      comm->planner.bcast_info.maxBcastPeer = INT_MIN;
      comm->planner.rmaTaskQueues = tmpRmaQueues;
      if (comm->planner.rmaTaskQueues != NULL) {
        for (int i = 0; i < numRmaCtx; i++) {
          ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
        }
      }
    }
  }
  ncclGroupBlocking = comm->config.blocking;
}

// Add comm to this thread's group needing preconnect
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    /** 如果当前comm不是已加入的preConnect类任务链表头节点 ，设置为链表头节点 */
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}

// Comm has left group
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm, int type) {
  comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID);
  if (type == ncclGroupTaskTypeCollective || type == ncclGroupTaskTypeRawTask) ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}

#endif
