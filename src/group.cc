/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "enqueue/task_sched.h"
#include "transport.h"
#include "channel.h"
#include "bootstrap.h"
#include "ce_coll.h"
#include "profiler.h"
#include "nvtx.h"
#include "compiler.h"
#include "rma/rma.h"
#include "argcheck.h"
#include <assert.h>
#include <chrono>
#include <thread>
#include "os.h"

#define GROUP_MAX_RECLAIM_STEPS 10

/*标记当前线程所在的group depth*/
thread_local int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
/*以下几个变量用于支持group实现，groupdepth被start调用,以实现增加
 * 当group被增加后，容许嵌套增加，而end调用会使groupdepth减少。直到group depth
 * 被减少到0时，end函数才整整的收集下列变量保存的内容，用于执行
 * groupLocalResetJobState负责以下变量的重置问题
 * */
thread_local ncclResult_t ncclGroupError = ncclSuccess;/*用于记录错误码*/
/*用于记录当前线程在一个group内保存的comm head列表（会在group end时被处理）
 * ncclGroupCommJoin负责向此变量中增加元素
 * */
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};
/*用于记录设置的preconnectHead，ncclGroupCommPreconnect用于增加元素*/
thread_local struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
/*用于存放异步job，ncclAsyncLaunch用于增加元素*/
thread_local struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
thread_local int ncclGroupBlocking = -1; /* default mode */
void* ncclAsyncJobMain(void* arg);

/*如果当前ncclGroupDepth为0，则直接调用，否则创建初始化job并入队到ncclAsyncJobs*/
ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func/*完成job必须函数*/)(struct ncclAsyncJob*),
                             void (*undo/*job执行失败后，用于回退，可选*/)(struct ncclAsyncJob*), void (*destructor/*job未执行/job执行完成/undo执行完后后，用于清理，可选*/)(void*), ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;

  job->destroyFlag = comm->destroyFlag;
  if (ncclGroupDepth == 0) {
	/*depth为0，直接调用func*/
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);/*如果失败执行undo*/
    if (destructor) destructor(job);/*执行销毁*/
  } else {
	/*初始化job*/
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->abortFlagDev = comm->abortFlagDev;
    job->childAbortFlag = comm->childAbortFlag;
    job->childAbortFlagDev = comm->childAbortFlagDev;
    job->state = ncclGroupJobRunning;/*置为running*/
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (comm->destroyFlag) {
      ncclGroupBlocking = 1;/*阻塞方式*/
    } else if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->config.blocking;
    } else if (ncclGroupBlocking != comm->config.blocking) {
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;/*配置冲突（阻塞与非阻塞冲突）*/
    }
    if (ret == ncclSuccess) {
    	/*异步job入队*/
      ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
    } else {
      // no need to undo, the job hasn't run
      if (destructor) destructor(job);
    }
  }

  return ret;
}

ncclResult_t ncclGroupJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsMain,
                                volatile bool* groupAbortFlag) {
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;

  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    if (job->next == nullptr) {
      job->isThreadMain = true;
      ncclAsyncJobMain(job);
      job->state = ncclGroupJobJoined;
      return job->result;
    }
    do {
      STDTHREADCREATE(job->thread, ncclAsyncJobMain, job);
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);
      do {
        ncclGroupJobState_t state = COMPILER_ATOMIC_LOAD(&job->state, std::memory_order_acquire);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;
        } else if (state == ncclGroupJobDone) {
          int err;
          if ((err = ncclThreadJoin(job->thread)) != ncclSuccess) {
            WARN("ncclGroupJobLaunch: failed to join thread for job");
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;
          if (job->result != ncclSuccess && ret == ncclSuccess) {
            ret = job->result;
            errorJobAbortFlag = true;
          }
        } else {
          /* safety check */
          assert(state == ncclGroupJobJoined);
        }

        if (!job->destroyFlag &&
            (COMPILER_ATOMIC_LOAD(groupAbortFlag, std::memory_order_acquire) || errorJobAbortFlag == true)) {
          COMPILER_ATOMIC_STORE(job->abortFlag, uint32_t(1), std::memory_order_release);
          COMPILER_ATOMIC_STORE(job->abortFlagDev, uint32_t(1), std::memory_order_release);
          if (job->childAbortFlag) {
            COMPILER_ATOMIC_STORE(job->childAbortFlag, uint32_t(1), std::memory_order_release);
            COMPILER_ATOMIC_STORE(job->childAbortFlagDev, uint32_t(1), std::memory_order_release);
          }
        }

        job = job->next;
      } while (job != nullptr);
      // Let preconnect threads progress.
      if (jobsDone == false) std::this_thread::sleep_for(std::chrono::microseconds(1));
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

exit:
  return ret;
fail:
  goto exit;
}

/**直接调用job的func函数来处理此job */
void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);/*调用job函数来处理此job,并设返回值*/
  if (job->result != ncclSuccess) {
    /**如果不成功，显示错误信息 */
    INFO_LOC(NCCL_INIT, "-> %d [Async thread]", job->result);
  }
  /*标记状态为done*/
  COMPILER_ATOMIC_STORE(&job->state, static_cast<ncclGroupJobState_t>(ncclGroupJobDone), std::memory_order_release);
  return arg;
}

ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;
  NCCLCHECK(ncclThreadJoin(job->thread));
  if (job->result != ncclSuccess) {
    WARN("ncclAsyncJobComplete: job %p failed, job error %d", job, job->result);
  }
  ret = job->result;
  if (job->destructor) job->destructor((void*)job);
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclResult_t ret = ncclSuccess;
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(ncclGroupStartInternal());/**仅增加层数（容许嵌套） */
  TRACE_CALL("ncclGroupStart()");
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  ncclResult_t ret = ncclSuccess;
  NCCL_NVTX3_FUNC_RANGE;
  NCCLCHECKGOTO(ncclGroupEndInternal(/**这里实际上默认传NULL了 */), ret, exit);/**执行group end实际流程（仅在深度达到0时执行） */
  TRACE_CALL("ncclGroupEnd()");
exit:
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupSimulateEnd, ncclSimInfo_t* simInfo);
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  NCCL_NVTX3_FUNC_RANGE;
  NCCLCHECKGOTO(ncclGroupEndInternal(simInfo), ret, exit);/**这种容许传入的simInfo不为NULL */
  TRACE_CALL("ncclGroupSimulateEnd()");
exit:
  return ret;
}

struct ncclPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;/*指向其从属的comm*/
  bool* algoNeedConnect;
};

struct ncclPrepareTasksAndCollPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  ncclSimInfo_t* simInfo;
};

struct ncclTaskPrepareJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  ncclSimInfo_t* simInfo;
};

struct ncclMgmtTaskJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};

ncclResult_t ncclP2PPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));/**设置当前线程的cuda设备为comm->cudaDev */
  /*设置cpu亲和性*/
  if (!job_->isThreadMain && ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

ncclResult_t ncclCollPreconnect(struct ncclComm* comm, bool* algoNeedConnect) {
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (algoNeedConnect[i]) {
      switch (i) {
      case NCCL_ALGO_RING:
        {
          NCCLCHECK(ncclTransportRingConnect(comm));
          break;
        }
      case NCCL_ALGO_TREE:
        {
          NCCLCHECK(ncclTransportTreeConnect(comm));
          break;
        }
      case NCCL_ALGO_NVLS:
        {
          /* If we are using NVLS_TREE algo, we must mark NVLS algo to set up
           * NVLS intra-node buffer */
          NCCLCHECK(ncclNvlsBufferSetup(comm));
          break;
        }
      case NCCL_ALGO_NVLS_TREE:
        {
          NCCLCHECK(ncclNvlsTreeConnect(comm));
          break;
        }
      case NCCL_ALGO_COLLNET_CHAIN:
        {
          NCCLCHECK(ncclCollNetChainBufferSetup(comm));
          break;
        }
      case NCCL_ALGO_COLLNET_DIRECT:
        {
          NCCLCHECK(ncclCollNetDirectBufferSetup(comm));
          break;
        }
      case NCCL_ALGO_PAT:
        {
          NCCLCHECK(ncclTransportPatConnect(comm));
          if (comm->localRanks > 1 && ncclNvlsTransportEnabled(comm)) {
            NCCLCHECK(ncclNvlsBufferSetup(comm));
          }
          break;
        }
        // Yes, it's a dead code.  That's fine...
        // coverity[dead_error_begin]
      default:
        {
          NCCLCHECK(ncclInternalError);
        }
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclPrepareTasksAndCollPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPrepareTasksAndCollPreconnectJob* job = (ncclPrepareTasksAndCollPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  bool needConnect;
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
  memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, job->simInfo));
  if (comm->cuMemSupport && needConnect) NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  return ncclSuccess;
}

static ncclResult_t ncclTaskPrepareJobFunc(struct ncclAsyncJob* job_) {
  struct ncclTaskPrepareJob* job = (struct ncclTaskPrepareJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  NCCLCHECK(ncclTaskPrepare(comm, job->simInfo));
  return ncclSuccess;
}

static ncclResult_t ncclMgmtTaskJobFunc(struct ncclAsyncJob* job_) {
  struct ncclMgmtTaskJob* job = (struct ncclMgmtTaskJob*)job_;
  struct ncclComm* comm = job->comm;
  struct ncclAsyncJob* task = nullptr;
  ncclResult_t ret = ncclSuccess;

  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  if (comm->destroyFlag) {
    task = ncclIntruQueueDequeue(&job->comm->mgmtTaskQueue);
    NCCLCHECKGOTO(task->func(task), ret, fail);
    if (task->destructor) task->destructor((void*)task);
  } else {
    while (!ncclIntruQueueEmpty(&job->comm->mgmtTaskQueue)) {
      task = ncclIntruQueueDequeue(&job->comm->mgmtTaskQueue);
      NCCLCHECKGOTO(task->func(task), ret, fail);
      if (task->destructor) task->destructor((void*)task);
    }
  }
exit:
  return ret;
fail:
  if (task && task->destructor) task->destructor((void*)task);
  goto exit;
}

ncclResult_t ncclCollPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (!job_->isThreadMain) CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  NCCLCHECKGOTO(ncclCollPreconnect(comm, job->algoNeedConnect), ret, fail);

exit:
  free(job->algoNeedConnect);
  return ret;
fail:
  goto exit;
}

struct ncclGroupSymmetricJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;/*指明从属的comm*/
};

struct ncclGroupDebugJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};

ncclResult_t ncclCommGroupArgsGlobalCheck(struct ncclAsyncJob* job_) {
  struct ncclGroupDebugJob* job = (struct ncclGroupDebugJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;
  while (!ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
    struct ncclArgsInfo* argsInfo = ncclIntruQueueDequeue(&comm->argsInfoQueue);
    NCCLCHECKGOTO(ncclArgsGlobalCheck(argsInfo), ret, fail);
    free(argsInfo);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommGroupRegisterSymmetric(struct ncclAsyncJob* job_) {
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&comm->devrState.regTaskQueue);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, task->userPtr, task->userSize, task->winFlags, task->outWinDev),
                  ret, fail);
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) {
    struct ncclDevrCommCreateTask* task = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue);
    NCCLCHECKGOTO(ncclDevrCommCreateInternal(comm, task->reqs, task->outDevComm, /*isInternal=*/false,
                                             task->deviceCodeVersion),
                  ret, fail);
    freeDevCommRequirements(task->reqs); // free additional task memory for reqs
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    NCCLCHECKGOTO(ncclCeInit(task->comm), ret, fail);
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* task = ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue);
    NCCLCHECKGOTO(ncclRmaCeInit(task->comm), ret, fail);
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->suspendTaskQueue);
    struct ncclComm* taskComm = task->comm;
    free(task);
    NCCLCHECKGOTO(ncclCommMemSuspend(taskComm), ret, fail);
  }

  while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->resumeTaskQueue);
    struct ncclComm* taskComm = task->comm;
    free(task);
    NCCLCHECKGOTO(ncclCommMemResume(taskComm), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t doLaunches(struct ncclComm* head, int taskType) {
  ncclResult_t result = ncclSuccess;
  struct ncclComm* cliqueHead = head;
  struct ncclComm* cliqueNextHead;
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
  // This outer loop iterates over cliques of comms which are siblings of the
  // same global entity. We calculate a clique as all comms which have the same
  // `intraComm0` value.
  do {
    struct ncclComm* comm = cliqueHead;
    bool capturingYes = false, capturingNo = false;
    do {
      (ncclCudaGraphValid(comm->planner.capturingGraph) ? capturingYes : capturingNo) = true;
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);/**设置当前线程的cuda设备为comm->cudaDev */
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);/**准备launch任务 */
      if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
      comm = comm->groupNext[taskType];/**获取下一个collective任务 */
    } while (comm != nullptr && comm != reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID) &&
             comm->intraComm0 == cliqueHead->intraComm0);
    cliqueNextHead = comm;

    if (capturingYes && capturingNo) {
      // We have entered barriers but are aborting without leaving them. Thus
      // these comms are permanently trashed. We need a good mechanism for
      // tracking and reporting that.
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      result = ncclInvalidUsage;
      goto failure;
    }

    while (true) {
      // Iterate rounds of launches for clique.
      bool moreRounds = false;
      comm = cliqueHead;
      do {
        // Iterate clique members.
        struct ncclComm* next = comm->groupNext[taskType];
        if (useBarrier) {
          // Barrier reduction result tells us if this was the final round.
          moreRounds = 0 != ncclCommIntraBarrierOut(comm);
        } else {
          moreRounds |= comm->planner.unlaunchedPlansHead != nullptr;
        }
        if (moreRounds) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->planner.unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            if (plan->isCeColl) {
              NCCLCHECKGOTO(ncclLaunchCeColl(comm, plan), result, failure);
            } else if (plan->isRma) {
              NCCLCHECKGOTO(ncclLaunchRma(comm, plan), result, failure);
            } else {
              NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
            }
          }
          // Barrier reduction input indicates if we require further rounds.
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->planner.unlaunchedPlansHead != nullptr ? 1 : 0);
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        } else {
          // Final round.
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
        }
        comm = next;
      } while (comm != reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID) && comm != cliqueNextHead);
      if (!moreRounds) break;
    }
    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr && cliqueHead != reinterpret_cast<struct ncclComm*>(NCCL_COMM_GROUP_INVALID));
failure:
  return result;
}

static inline void groupLocalResetJobState() {
  ncclGroupError = ncclSuccess;
  /*重置group comm head列表*/
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) ncclGroupCommHead[type] = NULL;
  ncclGroupCommPreconnectHead = NULL;/*重置comm preconnected列表*/
  ncclGroupBlocking = -1;
  /*重置ncclAsyncJobs*/
  ncclIntruQueueConstruct(&ncclAsyncJobs);
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr,
                         struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr,
                         ncclResult_t error) {
  struct ncclComm* comm;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    comm = groupCommHeadPtr[type];
    // reset groupCommHeadPtr[type]
    groupCommHeadPtr[type] = nullptr;
    while (comm != nullptr) {
      struct ncclComm* next = comm->groupNext[type];
      (void)ncclGroupCommLeave(comm, type); // overwrites comm->groupNext
      // We don't know if preconnect succeeded or happened at all, so clear
      // the flags that let `taskAppend()` skip over checking if preconnect
      // is needed.
      if (type == ncclGroupTaskTypeCollective || type == ncclGroupTaskTypeRawTask) {
        comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
        for (int i = 0; i < comm->nRanks; i++) {
          comm->connectSend[i] = 0UL;
          comm->connectRecv[i] = 0UL;
        }
        // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
        // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
        while (!ncclIntruQueueEmpty(&comm->planner.planQueue)) {
          struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planner.planQueue);
          // Persistent plans will be reclaimed via the callbackQueue when the
          // graph drops its UserObject reference.
          if (!plan->persistent) {
            while (!ncclIntruQueueEmpty(&plan->proxyOpQueue)) {
              struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->proxyOpQueue);
              ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
            }
            ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
          }
        }

        { // Reset comm->planner to empty.
          ncclKernelPlanner::Peer* tmp = comm->planner.peers;
          ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* tmpRmaQueues = comm->planner.rmaTaskQueues;
          int numRmaCtx = comm->config.numRmaCtx;

          memset(&comm->planner, 0, sizeof(comm->planner));

          comm->planner.peers = tmp;
          if (comm->planner.peers != NULL) {
            memset(comm->planner.peers, 0, comm->nRanks * sizeof(comm->planner.peers[0]));
          }
          comm->planner.bcast_info.minBcastPeer = INT_MAX;
          comm->planner.bcast_info.maxBcastPeer = INT_MIN;

          comm->planner.rmaTaskQueues = tmpRmaQueues;
          if (comm->planner.rmaTaskQueues != NULL) {
            for (int i = 0; i < numRmaCtx; i++) {
              ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
            }
          }
        }

        if (type == ncclGroupTaskTypeRawTask) {
          while (!ncclIntruQueueEmpty(&comm->rawTaskQueue.genericQueue)) {
            struct ncclRawTask* task = ncclIntruQueueDequeue(&comm->rawTaskQueue.genericQueue);
            ncclMemoryPoolFree(&comm->memPool_ncclRawTask, task);
          }
          while (!ncclIntruQueueEmpty(&comm->rawTaskQueue.bcastQueue)) {
            struct ncclRawTask* task = ncclIntruQueueDequeue(&comm->rawTaskQueue.bcastQueue);
            ncclMemoryPoolFree(&comm->memPool_ncclRawTask, task);
          }
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.symTaskQueue);
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.legacyTaskQueue);
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.allgathervTaskQueue);
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.p2pTaskQueue);
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.rmaTaskQueue);
          ncclIntruQueueConstruct(&comm->classifiedTaskQueues.ceTaskQueue);
        }
      } else if (type == ncclGroupTaskTypeMgmtTask) {
        while (!ncclIntruQueueEmpty(&comm->mgmtTaskQueue)) {
          struct ncclAsyncJob* task = ncclIntruQueueDequeue(&comm->mgmtTaskQueue);
          if (task->destructor) task->destructor((void*)task);
        }
      }

      if (!comm->config.blocking) (void)ncclCommSetAsyncError(comm, error);
      comm = next;
    }
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking) (void)ncclCommSetAsyncError(job->comm, error);
    if (job->destructor) job->destructor((void*)job);
  }

  return;
}

/*由于多个job是通过job的next指针串起来的，
 * 因此当有多个job时，
 * asyncJobLanch函数实际上是创建多个线程同时处理多个job,
 * 并阻塞等待所有线程完成任务*/
static ncclResult_t asyncJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsMain,
                                   volatile bool* groupAbortFlag/**出参，用于记录是否有job被aborted */) {
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;

  /**仅队列不为空才处理 */
  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);/**获取队列头的任务 */
    if (job->next == nullptr) {
      /**如果只有一个任务，直接在主线程中跑 */
      job->isThreadMain = true;
      ncclAsyncJobMain(job);/**在当前线程中直接调此函数，处理job */
      job->state = ncclGroupJobJoined;
      return job->result;
    }
    do {
      /**有多个job时，每个job创建一个线程跑job->func*/
      STDTHREADCREATE(job->thread, ncclAsyncJobMain/**在线程中调此函数，处理job */, job);
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);/**再获取队列头的任务（即上面创建的线程传入的一组job） */
      do {
        /*取job的最新状态*/
        ncclGroupJobState_t state = COMPILER_ATOMIC_LOAD(&job->state, std::memory_order_acquire);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;/**还在运行，未处理完成 */
        } else if (state == ncclGroupJobDone) {
          /**如果job->state为ncclGroupJobDone，说明任务已完成，等待job->thread退出 */
          int err;
          if ((err = ncclThreadJoin(job->thread)) != ncclSuccess) {
            WARN("asyncJobLaunch: failed to join thread for job");
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;/**标记线程已joined */
          if (job->result != ncclSuccess && ret == ncclSuccess) {
            ret = job->result;/**如果job->result不是ncclSuccess，说明任务失败，更新ret为job->result */
            errorJobAbortFlag = true;/**标记任务失败 */
          }
        } else {
          /* safety check */
          if (state != ncclGroupJobJoined) {/**其它情况只能是joined */
            WARN("Async job state is %d, expected %d", state, ncclGroupJobJoined);
            if (ret == ncclSuccess) ret = ncclInternalError;
            errorJobAbortFlag = true;
          }
        }

        if (!job->destroyFlag &&
            (COMPILER_ATOMIC_LOAD(groupAbortFlag, std::memory_order_acquire) || errorJobAbortFlag == true)) {
          COMPILER_ATOMIC_STORE(job->abortFlag, uint32_t(1), std::memory_order_release);/*标记此job被abort*/
          COMPILER_ATOMIC_STORE(job->abortFlagDev, uint32_t(1), std::memory_order_release);
          if (job->childAbortFlag) {
            COMPILER_ATOMIC_STORE(job->childAbortFlag, uint32_t(1), std::memory_order_release);
            COMPILER_ATOMIC_STORE(job->childAbortFlagDev, uint32_t(1), std::memory_order_release);
          }
        }

        job = job->next;/**处理下一个job */
      } while (job != nullptr);
      // Let preconnect threads progress.
      if (jobsDone == false) std::this_thread::sleep_for(std::chrono::microseconds(1));
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SingleProcMemRegEnable, "SINGLE_PROC_MEM_REG_ENABLE", 0);

static void ncclPrepareTasksAndCollPreconnectJobFree(void* _job) {
  struct ncclPrepareTasksAndCollPreconnectJob* job = (struct ncclPrepareTasksAndCollPreconnectJob*)_job;
  delete job;
}

static void ncclTaskPrepareJobFree(void* _job) {
  delete (struct ncclTaskPrepareJob*)_job;
}

static void ncclMgmtTaskJobFree(void* _job) {
  delete (struct ncclMgmtTaskJob*)_job;
}

static void ncclPreconnectJobFree(void* _job) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)_job;
  delete job;
}

static void ncclGroupSymmetricJobFree(void* _job) {
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)_job;
  delete job;
}

static ncclResult_t ncclPrepareTasksAndCollPreconnect(
  struct ncclComm* comm, ncclSimInfo_t* simInfo,
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncCollJobs) {
  if (ncclParamSingleProcMemRegEnable()) {
    struct ncclPrepareTasksAndCollPreconnectJob* job;
    NEW_NOTHROW(job, ncclPrepareTasksAndCollPreconnectJob);
    job->base.func = ncclPrepareTasksAndCollPreconnectFunc;/*指明job函数*/
    job->base.destructor = ncclPrepareTasksAndCollPreconnectJobFree;
    job->base.state = ncclGroupJobRunning;
    job->base.abortFlag = comm->abortFlag;
    job->base.abortFlagDev = comm->abortFlagDev;
    job->comm = comm;
    job->simInfo = simInfo;
    ncclIntruQueueEnqueue(asyncCollJobs, &job->base);
  } else {
    bool needConnect = false;
    bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
    memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

    CUDACHECK(cudaSetDevice(comm->cudaDev));
    NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo));

    if (comm->cuMemSupport && needConnect) {
      ncclResult_t ret;
      struct ncclPreconnectJob* job;
      NEW_NOTHROW(job, ncclPreconnectJob);
      job->base.func = ncclCollPreconnectFunc;
      job->base.destructor = ncclPreconnectJobFree;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      if ((ret = ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS))) {
        delete job;
        NCCLCHECK(ret);
      }
      memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
      ncclIntruQueueEnqueue(asyncCollJobs, &job->base);/**job入人 */
    }
  }
  return ncclSuccess;
}

/** 按阶段创建job,并发跑并回收 */
static ncclResult_t groupLaunchLegacy(struct ncclAsyncJob* job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob* gjob = (struct ncclGroupJob*)job_;/*此group对应的job*/
  /*取这三种待执行内容*/
  struct ncclComm** groupCommHeadMain = gjob->groupCommHead;
  struct ncclComm* groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsMain = &gjob->asyncJobs;
  bool* groupAbortFlag = &gjob->abortFlag;

  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    /**preConnect链表不空时，按顺序生成ncclPreconnectJob挂在asyncJobsMain中 */
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;/**preConnect任务的job */
      NEW_NOTHROW_GOTO(job, ncclPreconnectJob, ret, fail);
      job->base.func = ncclP2PPreconnectFunc;/**指明此类job的处理函数 */
      job->base.undo = nullptr;/*不回退*/
      job->base.destructor = ncclPreconnectJobFree;/*仅释放内存*/
      job->base.state = ncclGroupJobRunning;/*状态指为running*/
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      /**将此任务添加到异步任务队列中（多个job是通过job的next指针串起来的)*/
      ncclIntruQueueEnqueue(asyncJobsMain, (struct ncclAsyncJob*)job);

      struct ncclComm* next = comm->preconnectNext;/*取下一个*/
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);/*断开链，并标记*/
      comm = next;/*继续循环*/
    } while (comm != nullptr);
  }

  /**启动异步任务队列asyncJobsMain中的任务,完成跑preConnect（在加入preConnect之前其内部可能已有其它job)
  由于多个job是通过job的next指针串起来的，因此当有多个job时，
  asyncJobLanch函数实际上是创建多个线程同时处理多个job,并阻塞等待所有线程完成任务
   */
  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);

  // only loop through sym alloc and register tasks
  for (int type = ncclGroupTaskTypeSymRegister; type <= ncclGroupTaskTypeSymRegister; ++type) {
    if (groupCommHeadMain[type]) {
      /**此类型任务链表不空 */
      struct ncclComm* cliqueHead = groupCommHeadMain[type];/**取此类型任务链表头节点 */
      struct ncclComm* comm = NULL;
      struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncSymJobs;
      ncclIntruQueueConstruct(&asyncSymJobs);/**初始化异步任务队列asyncSymJobs */
      do {
        comm = cliqueHead;
        do {
          struct ncclGroupSymmetricJob* job;
          NEW_NOTHROW_GOTO(job, ncclGroupSymmetricJob, ret, fail);
          job->base.func = ncclCommGroupRegisterSymmetric;/**指明此类job的处理函数 */
          job->base.undo = nullptr;/*不回退*/
          job->base.destructor = ncclGroupSymmetricJobFree;/*仅释放内存*/
          job->base.state = ncclGroupJobRunning;/*指明running*/
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncSymJobs, (struct ncclAsyncJob*)job);/**将符号申请与注册任务添加到异步任务队列中 */
          comm = comm->groupNext[type];/**取同一类型的下一个communicator,进行入队*/
        } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0/*？？？*/);
        /**启动异步任务队列中的任务,跑完符号申请与注册任务（这一类型的将并行被执行，同一类型如果intraComm0不同也不并行执行） */
        NCCLCHECKGOTO(asyncJobLaunch(&asyncSymJobs, groupAbortFlag), ret, fail);
        /**在asyncJobLaunch中，我们已经完成了所有job的处理，
         * 但并没有自动列中移除这些job，这里遍历一次，并调用其destructor回调函数，释放内存 */
        while (!ncclIntruQueueEmpty(&asyncSymJobs)) {
          struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncSymJobs);/**从异步任务队列中取一个任务 */
          if (job->destructor) job->destructor((void*)job);/**释放此任务的内存 */
        }
        cliqueHead = comm;
      } while (cliqueHead != nullptr);
    }
  }

  /* Connect channels at runtime if cumem is supported */
  if (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) {
    /**collective任务链表不空 */
    struct ncclComm* cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    struct ncclComm* comm = NULL;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncDebugJobs;
    ncclIntruQueueConstruct(&asyncCollJobs);/**初始化异步任务队列asyncCollJobs */
    ncclIntruQueueConstruct(&asyncDebugJobs);
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      comm = cliqueHead;
      do {
    	  /*入队列asyncCollJobs中*/
        NCCLCHECKGOTO(ncclPrepareTasksAndCollPreconnect(comm, simInfo, &asyncCollJobs), ret, fail);
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0/*同一类型，也会阶段*/);
      /**启动异步任务队列中的任务,执行collective job*/
      // connect
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
        if (job->destructor) job->destructor((void*)job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);

    // done with all buffer allocation, start registration and enqueue
    comm = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
      NCCLCHECKGOTO(ncclTasksRegAndEnqueue(comm), ret, fail);
      comm = comm->groupNext[ncclGroupTaskTypeCollective];
    } while (comm);

    // debug check
    cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      comm = cliqueHead;
      do {
        if (comm->checkMode == ncclCheckModeDebugGlobal) {
          struct ncclGroupSymmetricJob* job;
          NCCLCHECK(ncclCalloc(&job, 1));
          job->base.func = ncclCommGroupArgsGlobalCheck;
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncDebugJobs, (struct ncclAsyncJob*)job);
        }
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      NCCLCHECKGOTO(asyncJobLaunch(&asyncDebugJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncDebugJobs)) {
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncDebugJobs);
        if (job->destructor) job->destructor((void*)job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);
  }

  if ((!simInfo) && (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr)) {
    /**simInfo为空，collective任务链表不空 */
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective], ncclGroupTaskTypeCollective), ret, fail);/** +++启动collective任务链表中的任务 */
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking &&
        groupCommHeadMain[ncclGroupTaskTypeCollective] == nullptr) {
      (void)ncclCommSetAsyncError(job->comm, ret);
    }
    if (job->destructor) job->destructor((void*)job);
  }

  /**对每一类任务链表做一遍遍历，目的 */
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    while (groupCommHeadMain[type] != nullptr) {
      struct ncclComm* comm = groupCommHeadMain[type];
      struct ncclComm* next = comm->groupNext[type];
      // Poll for callbacks sent to us from other threads. Typically these free
      // resources from to our memory pools and UB
      if (comm->reclaimSteps == GROUP_MAX_RECLAIM_STEPS) {
        NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, fail);
        comm->reclaimSteps = 0;
      } else {
        comm->reclaimSteps++;
      }
      (void)ncclGroupCommLeave(comm, type);
      if (!comm->config.blocking) {
        (void)ncclCommSetAsyncError(comm, ret);
      }
      groupCommHeadMain[type] = next;
    }
  }

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHead, &gjob->asyncJobs, ret);
  goto exit;
}

// Preparation-job errors abort communicators. Validate launch-completion-event
// usage first so invalid usage is returned without aborting the communicator.
static ncclResult_t groupValidateLaunchCompletionEvents(struct ncclComm* comm) {
  while (comm != nullptr) {
    NCCLCHECK(ncclValidateCollConfigLaunchCompletionEvents(comm));
    comm = comm->groupNext[ncclGroupTaskTypeRawTask];
  }
  return ncclSuccess;
}

static ncclResult_t groupLaunchEnqueueRearch(struct ncclAsyncJob* job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob* gjob = (struct ncclGroupJob*)job_;
  struct ncclComm** groupCommHeadMain = gjob->groupCommHead;
  bool* groupAbortFlag = &gjob->abortFlag;

  struct ncclComm* comm = nullptr;
  struct ncclComm* cliqueHead = nullptr;
  bool destroyFlag = false;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncMgmtTaskJobs;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncPrepareJobs;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncScheduleJobs;
  ncclIntruQueueConstruct(&asyncMgmtTaskJobs);
  ncclIntruQueueConstruct(&asyncPrepareJobs);
  ncclIntruQueueConstruct(&asyncScheduleJobs);

  NCCLCHECKGOTO(groupValidateLaunchCompletionEvents(groupCommHeadMain[ncclGroupTaskTypeRawTask]), ret, fail);

  // launch management tasks
  cliqueHead = groupCommHeadMain[ncclGroupTaskTypeMgmtTask];
  while (cliqueHead) {
    comm = cliqueHead;
    do {
      struct ncclMgmtTaskJob* job;
      NEW_NOTHROW_GOTO(job, ncclMgmtTaskJob, ret, fail);
      job->base.func = ncclMgmtTaskJobFunc;
      job->base.destructor = ncclMgmtTaskJobFree;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->base.childAbortFlag = comm->childAbortFlag;
      job->base.childAbortFlagDev = comm->childAbortFlagDev;
      job->base.destroyFlag = comm->destroyFlag;
      destroyFlag = comm->destroyFlag;
      job->comm = comm;
      ncclIntruQueueEnqueue(&asyncMgmtTaskJobs, &job->base);
      comm = comm->groupNext[ncclGroupTaskTypeMgmtTask];
    } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
    NCCLCHECKGOTO(ncclGroupJobLaunch(&asyncMgmtTaskJobs, groupAbortFlag), ret, fail);
    while (!ncclIntruQueueEmpty(&asyncMgmtTaskJobs)) {
      struct ncclAsyncJob* mgmtTaskJob = ncclIntruQueueDequeue(&asyncMgmtTaskJobs);
      if (mgmtTaskJob->destructor) mgmtTaskJob->destructor((void*)mgmtTaskJob);
    }
    cliqueHead = comm;
  }
  // prepare tasks clique by clique
  cliqueHead = groupCommHeadMain[ncclGroupTaskTypeRawTask];
  while (cliqueHead) {
    comm = cliqueHead;
    do {
      struct ncclTaskPrepareJob* job;
      NEW_NOTHROW_GOTO(job, ncclTaskPrepareJob, ret, fail);
      job->base.func = ncclTaskPrepareJobFunc;
      job->base.destructor = ncclTaskPrepareJobFree;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      job->simInfo = simInfo;
      ncclIntruQueueEnqueue(&asyncPrepareJobs, &job->base);
      comm = comm->groupNext[ncclGroupTaskTypeRawTask];
    } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
    NCCLCHECKGOTO(ncclGroupJobLaunch(&asyncPrepareJobs, groupAbortFlag), ret, fail);
    while (!ncclIntruQueueEmpty(&asyncPrepareJobs)) {
      struct ncclAsyncJob* prepareJob = ncclIntruQueueDequeue(&asyncPrepareJobs);
      if (prepareJob->destructor) prepareJob->destructor((void*)prepareJob);
    }
    cliqueHead = comm;
  }
  // Schedule and launch tasks. Scheduler and launcher module of the enqueue framework
  // is not yet implemented and falls back to the legacy launcher: a single phased
  // doLaunches over the clique, run here on the user's thread.
  if (!simInfo && groupCommHeadMain[ncclGroupTaskTypeRawTask] != nullptr) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeRawTask], ncclGroupTaskTypeRawTask), ret, fail);
  }

  if (destroyFlag) {
    for (int type = ncclGroupTaskTypeRawTask; type <= ncclGroupTaskTypeMgmtTask; ++type) {
      groupCommHeadMain[type] = nullptr;
    }
  } else {
    for (int type = ncclGroupTaskTypeRawTask; type <= ncclGroupTaskTypeMgmtTask; ++type) {
      while (groupCommHeadMain[type] != nullptr) {
        struct ncclComm* comm = groupCommHeadMain[type];
        struct ncclComm* next = comm->groupNext[type];
        // Poll for callbacks sent to us from other threads. Typically these free
        // resources from our memory pools and UB
        if (comm->reclaimSteps == GROUP_MAX_RECLAIM_STEPS) {
          NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, fail);
          comm->reclaimSteps = 0;
        } else {
          comm->reclaimSteps++;
        }
        (void)ncclGroupCommLeave(comm, type);
        if (!comm->config.blocking) {
          (void)ncclCommSetAsyncError(comm, ret);
        }
        groupCommHeadMain[type] = next;
      }
    }
  }

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHead, &gjob->asyncJobs, ret);
  goto exit;
}

static ncclResult_t groupLaunch(struct ncclAsyncJob* job_, ncclSimInfo_t* simInfo = NULL) {
  return ncclParamEnqueueRearchEnable() ? groupLaunchEnqueueRearch(job_, simInfo) : groupLaunchLegacy(job_, simInfo);
}

/*非阻塞型按线程处理的groupLaunch*/
static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob* job_) {
  return groupLaunch(job_ /* estimatedTime = NULL */);
}

ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  ncclSimInfo_t internalSimInfo = NCCL_SIM_INFO_INITIALIZER;
  ncclSimInfo_t* internalSimInfoPtr = NULL;
  size_t realSize = 0;
  bool hasCommHead = false;
  ncclGroupJob* groupJob = NULL;
  internalSimInfo.magic = 0;

  if (ncclGroupDepth == 0) {
    /*调用end时，start未调用，返回错误*/
    WARN("ncclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth--;
  }
  if (ncclProfilerApiState.profilerGroupDepth == 0) {
    NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart));
  }

  if ((--ncclGroupDepth) > 0) goto exit;/**减少depth时未减到0，退出 */

  /**仅在groupDepth为0时，才执行以下流程 */
  if ((ret = ncclGroupError) != ncclSuccess) goto fail;/*减为0了，但有错误发生*/

  if (simInfo) {
    memcpy((void*)&realSize, (void*)&simInfo->size, sizeof(size_t));
    realSize = realSize > sizeof(ncclSimInfo_t) ? sizeof(ncclSimInfo_t) : realSize;
    memcpy((void*)&internalSimInfo, (void*)simInfo, realSize);
    if (internalSimInfo.magic != 0x74685283) {
      WARN("ncclSimInfo_t argument not initialized via NCCL_SIM_INFO_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }
    internalSimInfoPtr = &internalSimInfo;
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    if (ncclGroupCommHead[type]) {
      hasCommHead = true;/**groupCommHead存在任务 */
      break;
    }
  }

  /** 申请内存，初始化groupJob */
  NEW_NOTHROW_GOTO(groupJob, ncclGroupJob, ret, fail);
  ncclIntruQueueConstruct(&groupJob->asyncJobs);
  groupJob->groupRefCount = 0;
  groupJob->nonBlockingInit = false;
  /*复制commHead到groupJob中*/
  memcpy(groupJob->groupCommHead, ncclGroupCommHead, sizeof(ncclGroupCommHead));
  /*设置commpreconnectHead到groupJob*/
  groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead;
  groupJob->groupError = ncclSuccess;
  groupJob->abortFlag = false;
  groupJob->joined = false;
  /*转移ncclAsyncJobs中的内容到groupJob*/
  ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs);

  if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
	  /*以上三者有任意一个有内容，则进入*/
    /* make sure ncclGroupBlocking has been set. */
    if (ncclGroupBlocking != 0 && ncclGroupBlocking != 1) {
	    /*阻塞方式/非阻塞方式必须指明*/
      WARN("Invalid group blocking state %d", ncclGroupBlocking);
      ret = ncclInternalError;
      goto fail;
    }
    if (ncclGroupBlocking == 0) {
      /**要求以非阻塞模式执行 */
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&groupJob->asyncJobs)) {
    	  /*asyncJobs中有job需要执行，取首个job*/
        ncclAsyncJob* job = ncclIntruQueueHead(&groupJob->asyncJobs);
        /*将此组job指定为InProgress状态，为其job->comm关联groupJob*/
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          if (job->comm->groupJob == NULL) {
            job->comm->groupJob = groupJob;
            groupJob->groupRefCount++;/*增加此groupJob计数*/
          }
          job = job->next;
        } while (job);
      }

      /*将此组ncclGroupCommHead指定为InProgress状态，为其comm关联groupJob*/
      for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
        if (ncclGroupCommHead[type]) {
          ncclComm_t comm = ncclGroupCommHead[type];
          do {
            NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
            /* link group job to communicators. */
            if (comm->groupJob == NULL) {
              comm->groupJob = groupJob;
              groupJob->groupRefCount++;/*增加引用计数*/
            }
            comm = comm->groupNext[type];
          } while (comm);
        }
      }

      /**要求非阻塞，创建线程,将groupLaunch放在线程中执行 */
      groupJob->base.func = groupLaunchNonBlocking;/*指明job的处理函数*/
      /*传参（异步时无simInfo参数）*/
      STDTHREADCREATE_GOTO(groupJob->base.thread, ncclAsyncJobMain, ret, fail, &groupJob->base);
      groupJob->nonBlockingInit = true;/*指明为非阻塞初始化*/
      ret = ncclInProgress;/*指明处理中*/
    } else {
      /* blocking group */
      /**要求以阻塞方式执行，直接执行groupLaunch */
      int savedDev;
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);/*保存当前cuda设备*/
      /**直接调用groupLaunch函数,完成所有任务 */
      NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr/*同步时有此参数*/), ret, fail);
      CUDACHECKGOTO(cudaSetDevice(savedDev), ret, fail);/*回复保存的cuda设备*/
      if (simInfo) memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize);/*填充simInfo*/
      delete groupJob;
    }
  } else {
    // Free when not needed (single rank case)
    delete groupJob;
  }
  /* Reset the job state for the next group call. */
  groupLocalResetJobState();/*重置以便支持next group*/

exit:
  // Profiler group API start is called inside taskAppend to get graph capture information for the event
  NCCLCHECK(ncclProfilerStopGroupApiEvent());
  return ret;
fail:
  if (groupJob) {
    groupCleanup(groupJob->groupCommHead, &groupJob->asyncJobs, ret);
    delete groupJob;
  } else {
	  /*将groupCommHead清零（使其在每个group范围内有效）*/
    groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ret);
  }
  groupLocalResetJobState();
  goto exit;
}

ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;
  if (groupJob && groupJob->nonBlockingInit) {
    if (!COMPILER_ATOMIC_EXCHANGE(&groupJob->joined, true, std::memory_order_acq_rel)) {
      ret = ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      delete groupJob;
    }
  }
  return ret;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  if (groupJob && groupJob->nonBlockingInit) {
    if (!COMPILER_ATOMIC_EXCHANGE(&groupJob->joined, true, std::memory_order_acq_rel)) {
      COMPILER_ATOMIC_STORE(&groupJob->abortFlag, true, std::memory_order_relaxed);
      ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      delete groupJob;
    }
  }
  return ncclSuccess;
}
