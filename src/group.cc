/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"
#include "channel.h"
#include <assert.h>
#include "bootstrap.h"
#include "ce_coll.h"
#include "profiler.h"
#include "nvtx.h"

#define GROUP_MAX_RECLAIM_STEPS 10

__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
__thread ncclResult_t ncclGroupError = ncclSuccess;
__thread struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
__thread int ncclGroupBlocking = -1; /* default mode */
void* ncclAsyncJobMain(void* arg);

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*), ncclComm_t comm
  ) {
  ncclResult_t ret = ncclSuccess;

  job->destroyFlag = comm->destroyFlag;
  if (ncclGroupDepth == 0) {
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->abortFlagDev = comm->abortFlagDev;
    job->childAbortFlag = comm->childAbortFlag;
    job->childAbortFlagDev = comm->childAbortFlagDev;
    job->state = ncclGroupJobRunning;
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (comm->destroyFlag) {
      ncclGroupBlocking = 1;
    } else if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->config.blocking;
    } else if (ncclGroupBlocking != comm->config.blocking) {
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;
    }
    if (ret == ncclSuccess) {
      ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
    } else {
      // no need to undo, the job hasn't run
      if (destructor) destructor(job);
    }
  }

  return ret;
}

/**直接调用job的func函数来处理此job */
void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);/*调用job函数来处理此job,并设返回值*/
  if (job->result != ncclSuccess) {
    /**如果不成功，显示错误信息 */
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);/*标记状态为done*/
  return arg;
}

ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;
  PTHREADCHECK(pthread_join(job->thread, NULL), "pthread_join");
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
  struct ncclComm* comm;
  bool* algoNeedConnect;
};

struct ncclPrepareTasksAndCollPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  ncclSimInfo_t* simInfo;
};

ncclResult_t ncclP2PPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));/**设置当前线程的cuda设备为comm->cudaDev */
  /*设置cpu亲和性*/
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

static ncclResult_t ncclCollPreconnect(struct ncclComm* comm, bool* algoNeedConnect) {
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (algoNeedConnect[i]) {
      switch (i) {
        case NCCL_ALGO_RING: {
          NCCLCHECK(ncclTransportRingConnect(comm));
          break;
        }
        case NCCL_ALGO_TREE: {
          NCCLCHECK(ncclTransportTreeConnect(comm));
          break;
        }
        case NCCL_ALGO_NVLS: {
          /* If we are using NVLS_TREE algo, we must mark NVLS algo to set up
           * NVLS intra-node buffer */
          NCCLCHECK(ncclNvlsBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_NVLS_TREE: {
          NCCLCHECK(ncclNvlsTreeConnect(comm));
          break;
        }
        case NCCL_ALGO_COLLNET_CHAIN: {
          NCCLCHECK(ncclCollNetChainBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_COLLNET_DIRECT: {
          NCCLCHECK(ncclCollNetDirectBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_PAT: {
          NCCLCHECK(ncclTransportPatConnect(comm));
          break;
        }
        // Yes, it's a dead code.  That's fine...
        // coverity[dead_error_begin]
        default: {
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
  memset(algoNeedConnect, 0, sizeof(bool)*NCCL_NUM_ALGORITHMS);
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, job->simInfo));
  if (comm->cuMemSupport && needConnect) NCCLCHECK(ncclCollPreconnect(comm, algoNeedConnect));
  return ncclSuccess;
}

ncclResult_t ncclCollPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (!job_->isThreadMain) CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECKGOTO(ncclCollPreconnect(comm, job->algoNeedConnect), ret, fail);

exit:
  free(job->algoNeedConnect);
  return ret;
fail:
  goto exit;
}

struct ncclGroupSymmetricJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};

ncclResult_t ncclCommGroupRegisterSymmetric(struct ncclAsyncJob* job_) {
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)job_;
  struct ncclComm* comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&comm->devrState.regTaskQueue);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(
      comm, task->userPtr, task->userSize, task->winFlags, task->outWinDev),
      ret, fail);
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) {
    struct ncclDevrCommCreateTask* task = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue);
    NCCLCHECKGOTO(ncclDevrCommCreateInternal(
      comm, (struct ncclDevCommRequirements const*)task->reqs, task->outDevComm),
      ret, fail);
    freeDevCommRequirements(task->reqs); // free additional task memory for reqs
    free(task);
  }

  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    NCCLCHECKGOTO(ncclCeInit(task->comm), ret, fail);
    free(task);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t doLaunches(struct ncclComm* head) {
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
      comm = comm->groupNext[ncclGroupTaskTypeCollective];/**获取下一个collective任务 */
    } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
    cliqueNextHead = comm;

    if (capturingYes && capturingNo) {
      // We have entered barriers but are aborting without leaving them. Thus
      // these comms are permanently trashed. We need a good mechanism for
      // tracking and reporting that.
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      result = ncclInvalidUsage;
      goto failure;
    }

    while (true) { // Iterate rounds of launches for clique.
      bool moreRounds = false;
      comm = cliqueHead;
      do { // Iterate clique members.
        struct ncclComm* next = comm->groupNext[ncclGroupTaskTypeCollective];
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
            } else {
              NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
            }
          }
          // Barrier reduction input indicates if we require further rounds.
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->planner.unlaunchedPlansHead != nullptr ? 1 : 0);
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        } else { // Final round.
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
        }
        comm = next;
      } while (comm != cliqueNextHead);
      if (!moreRounds) break;
    }
    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr);
failure:
  return result;
}

static inline void groupLocalResetJobState() {
  ncclGroupError = ncclSuccess;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) ncclGroupCommHead[type] = NULL;
  ncclGroupCommPreconnectHead = NULL;
  ncclGroupBlocking = -1;
  ncclIntruQueueConstruct(&ncclAsyncJobs);
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t error) {
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
      if (type == ncclGroupTaskTypeCollective) {
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
          memset(&comm->planner, 0, sizeof(comm->planner));
          comm->planner.peers = tmp;
          if (comm->planner.peers != NULL) memset(comm->planner.peers, 0, comm->nRanks * sizeof(comm->planner.peers[0]));
        }
      }

      if (!comm->config.blocking)
        (void)ncclCommSetAsyncError(comm, error);
      comm = next;
    }
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, error);
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  return;
}

/*由于多个job是通过job的next指针串起来的，因此当有多个job时，asyncJobLanch函数实际上是创建多个线程同时处理多个job,并阻塞等待所有线程完成任务*/
static ncclResult_t asyncJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain, volatile bool *groupAbortFlag/**出参，用于记录是否有job被aborted */) {
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
      /**有多个job，对job->next不为空的情况，创建线程，每个job一个线程 */
      PTHREADCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain/**在线程中调此函数，处理job */, job), "pthread_create", ret, fail);
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);/**再获取队列头的任务（即上面创建的线程传入的一组job） */
      do {
        /*取job的最新状态*/
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;/**还在运行，未处理完成 */
        } else if (state == ncclGroupJobDone) {
          /**如果job->state为ncclGroupJobDone，说明任务已完成，等待job->thread退出 */
          int err;
          if ((err = pthread_join(job->thread, nullptr)) != 0) {
            WARN("Error waiting for pthread_join: %s", strerror(err));
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;/**标记线程已joined */
          if (job->result != ncclSuccess && ret == ncclSuccess) {
            ret = job->result;/**如果job->result不是ncclSuccess，说明任务失败，更新ret为job->result */
            errorJobAbortFlag = true;/**标记任务失败 */
          }
        } else {
          /* safety check */
          assert(state == ncclGroupJobJoined);/**其它情况只能是joined */
        }

        if (!job->destroyFlag && (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) || errorJobAbortFlag == true)) {
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
          __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
          if (job->childAbortFlag) {
            __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
            __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
          }
        }

        job = job->next;/**处理下一个job */
      } while (job != nullptr);
      // Let preconnect threads progress.
      if (jobsDone == false) usleep(1);
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SingleProcMemRegEnable, "SINGLE_PROC_MEM_REG_ENABLE", 0);

static ncclResult_t ncclPrepareTasksAndCollPreconnect(struct ncclComm* comm, ncclSimInfo_t* simInfo, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncCollJobs) {
  if (ncclParamSingleProcMemRegEnable()) {
    struct ncclPrepareTasksAndCollPreconnectJob* job;
    NCCLCHECK(ncclCalloc(&job, 1));
    job->base.func = ncclPrepareTasksAndCollPreconnectFunc;
    job->base.undo = nullptr;
    job->base.destructor = free;
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
      NCCLCHECK(ncclCalloc(&job, 1));
      job->base.func = ncclCollPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      if ((ret = ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS))) {
        free(job);
        NCCLCHECK(ret);
      }
      memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
      ncclIntruQueueEnqueue(asyncCollJobs, &job->base);/**job入人 */
    }
  }
  return ncclSuccess;
}

/** 按阶段创建job,并发跑并回收 */
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm **groupCommHeadMain = gjob->groupCommHead;
  struct ncclComm *groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = &gjob->asyncJobs;
  bool *groupAbortFlag = &gjob->abortFlag;

  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    /**preConnect链表不空，生成job挂在asyncJobsMain中 */
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;/**preConnect任务的job */
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclP2PPreconnectFunc;/**指明此类job的处理函数 */
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain,  (struct ncclAsyncJob*)job);/**将此任务添加到异步任务队列中（多个job是通过job的next指针串起来的） */

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  /**启动异步任务队列asyncJobsMain中的任务,完成跑preConnect
  由于多个job是通过job的next指针串起来的，因此当有多个job时，asyncJobLanch函数实际上是创建多个线程同时处理多个job,并阻塞等待所有线程完成任务
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
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->base.func = ncclCommGroupRegisterSymmetric;/**指明此类job的处理函数 */
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncSymJobs, (struct ncclAsyncJob*)job);/**将符号申请与注册任务添加到异步任务队列中 */
          comm = comm->groupNext[type];/**取同一类型的下一个communicator */
        } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
        NCCLCHECKGOTO(asyncJobLaunch(&asyncSymJobs, groupAbortFlag), ret, fail);/**启动异步任务队列中的任务,跑完符号申请与注册任务 */
        /**在asyncJobLaunch中，我们已经完成了所有job的处理，但并没有自动列中移除这些job，这里遍历一次，并调用其destructor回调函数，释放内存 */
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
    ncclIntruQueueConstruct(&asyncCollJobs);/**初始化异步任务队列asyncCollJobs */
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      comm = cliqueHead;
      do {
        NCCLCHECKGOTO(ncclPrepareTasksAndCollPreconnect(comm, simInfo, &asyncCollJobs), ret, fail);
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      // connect
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);/**启动异步任务队列中的任务,跑完collective任务的preconnect */
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
  }

  if ((!simInfo) && (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr)) {
    /**simInfo为空，collective任务链表不空 */
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective]), ret, fail);/** +++启动collective任务链表中的任务 */
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking && groupCommHeadMain[ncclGroupTaskTypeCollective] == nullptr)
      (void) ncclCommSetAsyncError(job->comm, ret);
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

static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob *job_) {
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
      hasCommHead = true;/**存在任务 */
      break;
    }
  }

  /** 申请内存，初始化groupJob */
  NCCLCHECKGOTO(ncclCalloc(&groupJob, 1), ret, fail);
  ncclIntruQueueConstruct(&groupJob->asyncJobs);
  groupJob->groupRefCount = 0;
  groupJob->nonBlockingInit = false;
  memcpy(groupJob->groupCommHead, ncclGroupCommHead, sizeof(ncclGroupCommHead));
  groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead;
  groupJob->groupError = ncclSuccess;
  groupJob->abortFlag = false;
  groupJob->joined = false;
  ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs);

  if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    /* make sure ncclGroupBlocking has been set. */
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
    if (ncclGroupBlocking == 0) {
      /**要求以非阻塞模式执行 */
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&groupJob->asyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&groupJob->asyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          if (job->comm->groupJob == NULL) {
            job->comm->groupJob = groupJob;
            groupJob->groupRefCount++;
          }
          job = job->next;
        } while (job);
      }

      for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
        if (ncclGroupCommHead[type]) {
          ncclComm_t comm = ncclGroupCommHead[type];
          do {
            NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
            /* link group job to communicators. */
            if (comm->groupJob == NULL) {
              comm->groupJob = groupJob;
              groupJob->groupRefCount++;
            }
            comm = comm->groupNext[type];
          } while (comm);
        }
      }

      /**要求非阻塞，创建线程,将groupLaunch放在线程中执行 */
      groupJob->base.func = groupLaunchNonBlocking;
      PTHREADCHECKGOTO(pthread_create(&groupJob->base.thread, NULL, ncclAsyncJobMain, (void*)&groupJob->base), "pthread_create", ret, fail);
      groupJob->nonBlockingInit = true;
      ret = ncclInProgress;
    } else {
      /* blocking group */
      /**要求塞塞，直接执行groupLaunch */
      int savedDev;
      CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);
      NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr), ret, fail);/**直接调用groupLaunch函数,完成所有任务 */
      CUDACHECKGOTO(cudaSetDevice(savedDev), ret, fail);
      if (simInfo) memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize);
      free(groupJob);
    }
  }
  /* Reset the job state for the next group call. */
  groupLocalResetJobState();

exit:
  // Profiler group API start is called inside taskAppend to get graph capture information for the event
  NCCLCHECK(ncclProfilerStopGroupApiEvent());
  return ret;
fail:
  if (groupJob) {
    groupCleanup(groupJob->groupCommHead, &groupJob->asyncJobs, ret);
    free(groupJob);
  } else {
    groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ret);
  }
  groupLocalResetJobState();
  goto exit;
}

ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;
  if (groupJob && groupJob->nonBlockingInit) {
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
      ret = ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      free(groupJob);
    }
  }
  return ret;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  if (groupJob && groupJob->nonBlockingInit) {
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
      __atomic_store_n(&groupJob->abortFlag, true, __ATOMIC_RELAXED);
      ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      free(groupJob);
    }
  }
  return ncclSuccess;
}
