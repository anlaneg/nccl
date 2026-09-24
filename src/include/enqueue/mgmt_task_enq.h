/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MGMT_TASK_ENQ_H_
#define NCCL_MGMT_TASK_ENQ_H_

#include "nccl.h"
#include "utils.h"

#include <thread>

typedef enum ncclGroupJobState {
  ncclGroupJobRunning = 0,
  ncclGroupJobDone = 1,
  ncclGroupJobJoined = 2,
} ncclGroupJobState_t;

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  std::thread thread;
  ncclResult_t result;
  /*完成job必须函数*/
  ncclResult_t (*func)(struct ncclAsyncJob*);
  /*job执行失败后，用于回退，可选*/
  void (*undo)(struct ncclAsyncJob*);
  /*job未执行/job执行完成/undo执行完后后，用于清理，可选*/
  void (*destructor)(void*);
  ncclGroupJobState_t state;/*标记job的执行状态*/
  /*指向其所属的abort标记*/
  uint32_t* abortFlag; /* point to comm abortFlag */
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  ncclComm_t comm;
  int destroyFlag;
  bool isThreadMain;

  ~ncclAsyncJob() {
    if (thread.joinable()) {
      (void)ncclThreadJoin(thread);
    }
  }
};

ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* task, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t comm);

#endif // NCCL_MGMT_TASK_ENQ_H_
