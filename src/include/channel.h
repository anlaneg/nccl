/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"
#include "utils.h"

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks,
                         struct ncclComm* comm);

/*由p2pRound映射channel*/
inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound) {
  int base;
  if (comm->nNodes > 1) {
	  /*多节点环境*/
    int localSize = comm->p2pSchedGroupSize;
    int groupDelta = p2pRound / localSize;/*round所在组组号*/
    int localDelta = p2pRound % localSize;/*round所在组序号*/
    /*divUp,表示每个group需要配对几个batch;乘后表示，此组起如位置对应的batch数*/
    base = groupDelta * divUp(localSize, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    /*定位round在group内的batch数*/
    base += localDelta / NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  } else {
	/** 单节点时，每个round对应一个channel base */
    base = p2pRound;
  }
  /*使base的取值在p2pnChannels以内*/
  return reverseBits(base, log2Up(comm->p2pnChannels)/*p2pnChannels需要用多少bits*/);
}

#endif
