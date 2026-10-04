/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include "os.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "gdrwrap.h"

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile = NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p, int* isCrossClique = nullptr);
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret);
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,
  ncclTopoGdrModeDefault = 1,
  ncclTopoGdrModePci = 2,
  ncclTopoGdrModeNum = 3
};
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode);

enum ncclTopoFlushType {
  ncclTopoFlushNone = 0,   // no flush needed
  ncclTopoFlushAlways = 1, // flush always needed
  ncclTopoFlushC2c = 2     // PCIe NIC and C2C sync path are unordered, flush is needed.
};
static inline uint32_t ncclGdcPinFlag(enum ncclTopoFlushType flush) {
  if (flush == ncclTopoFlushC2c && ncclGdrPinV2Available()) return GDR_PIN_FLAG_FORCE_PCIE;
  return GDR_PIN_FLAG_DEFAULT;
}
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool* avail);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

ncclResult_t ncclGetUserP2pLevel(int* level);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity);

#define NCCL_TOPO_CPU_ARCH_UNDEF 0
#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN12 1
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN34 2
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN5 3
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount);
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw);

enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,
  NETDEVS_POLICY_ALL = 0x1,
  NETDEVS_POLICY_MAX = 0x2,
  NETDEVS_POLICY_UNDEF = 0xffffffff
};
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum);

// Allows for up to 576 GPUs (e.g., NVLD144) with headroom for internal operations
#define NCCL_TOPO_MAX_NODES 640
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes);

// Local (myself)
/*自已到自已，同一个设备*/
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
/*共同在同一个pcie switch下*/
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
/*在不同的pcie switch下，但在相同的pcie host bridge下*/
#define PATH_PXB 5

// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 6

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network
// send/recv operations.
#define PATH_PXN 7

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
/*共同在一个CPU下，即分属于一个或多个host bridge，此时互联需要经过host brige*/
#define PATH_PHB 8

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
/*共同节点是System(即分属于不同的numa node,此时互联需要过QPI)*/
#define PATH_SYS 9

// Connection through the network
#define PATH_NET 10

// New type of path which should precede PATH_PIX
/*同一个NIC的多个端口*/
#define PATH_PORT PATH_NVL

// Disconnected
/*认定为断开（未通到<system>节点，不可达）*/
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE \
  1   // Spread NIC traffic between two GPUs (Tree parent + one child on first
                                            // GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE \
  2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree
                                            // children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
struct ncclTopoGraph {
  // Input / output
  /*算法编号 */
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  /*拓扑模式 */
  int pattern;
  /*是否容许跨轨*/
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  /*规划出几条并行通道（这此字段均是集群对齐的）*/
  int nChannels;
  /*节点内估算带宽（同机GPU-GPU）每通道带宽*/
  float bwIntra;
  /*节点间估算带宽（跨机 GPU-NIC-NIC-GPU)每通道带宽*/
  float bwInter;
  /*节点间估算延迟*/
  float latencyInter;
  /*节点内链路类型*/
  float latencyInter;
  /*节点内链路类型*/
  int typeIntra;
  /*节点间链路类型*/
  int typeInter;
  int sameChannels;
  /*节点间跳数*/
  int nHops;
  /*指明各channel中本机各gpu rank的排列顺序*/
  int intra[MAXCHANNELS * NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS * 2];/*指明各channel中前一个net节点，后一个net节点*/
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int crossNicRing;
  /*用于描述整个本机片段用于连接其它机器的recv边界（即从其它机器收的rank)*/
  int ringRecv[MAXCHANNELS];
  /*用于描述整个本机片段用于连接其它机器的send边界(即向其它机器发的rank)*/
  int ringSend[MAXCHANNELS];
  /*用于描述当前rank在本机片段内部的recv边界（即从本机收的rank)*/
  int ringPrev[MAXCHANNELS];
  /*用于描述当前rank在本机片段内部的send边界（即向本机发的rank)*/
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];/*本机哪个rank连接父机器*/
  int treeToChild0[MAXCHANNELS];/*本机哪个rank连接第一个子机器（leftchild)*/
  int treeToChild1[MAXCHANNELS];/*本机哪个rank连接第二个子机器(rightchild)*/
  /*nvlink通信中，head负责额外的中转工作，即用于节点间通信（这也是为什么前面在算拓扑时head算了两份流量）*/
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;/*nvlsHeads数组的有效长度*/
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time);

#endif
