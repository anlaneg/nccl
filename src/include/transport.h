/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "device.h"
#include "graph.h"
#include "nvmlwrap.h"
#include "core.h"

#define NTRANSPORTS 4
#define TRANSPORT_UNDEFINED -1
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3
#define TRANSPORT_PROFILER 4

#include "proxy.h"
#include "comm.h"
#include "bootstrap.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;
extern struct ncclTransport profilerTransport;

extern struct ncclTransport* ncclTransports[];
// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

struct ncclPeerInfo {
  int rank;/*节点当前rank*/
  int cudaDev;
  int nvmlDev;
  int gdrSupport;/*是否支持gdr*/
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;/*share memory对应的dev_t*/
  int64_t busId;
  struct ncclComm* comm;
  int cudaCompCap;
  size_t totalGlobalMem;
  // MNNVL support
  nvmlGpuFabricInfoV_t fabricInfo;
  int cuMemSupport;/*cuda是否支持'虚拟内存管理'*/
  int version;
};

#define CONNECT_SIZE 256
#define NCCL_MAX_PAGE_SIZE (512L * 1024L * 1024L)
#define NCCL_REC_PAGE_SIZE (2L * 1024L * 1024L)
struct ncclConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010

#define NVLS_HANDLE_SIZE 64
struct ncclNvlsSharedRes {
  int refCount;
  bool inited;
  CUmulticastObjectProp bufProp;
  CUmulticastObjectProp signalProp;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t creditUCSize;
  size_t creditMCSize;
  size_t buffUCSize;
  size_t buffMCSize;
  CUmemGenericAllocationHandle mcBuffHandle; // Multicast handle for NVLS buffer
  CUmemGenericAllocationHandle mcCreditHandle; // Multicast handle for NVLS credit buffer
  char* mcBuff; // Multicast NVLS buffer address
  char* mcCredit; // Multicast NVLS credit address
  CUmemGenericAllocationHandle ucBuffHandle; // Unicast Handle for NVLS buffer
  CUmemGenericAllocationHandle ucCreditHandle; // Unicast Handle for NVLS credit buffer
  char* ucBuff; // Unicast NVLS buffer address
  char* ucCredit; // Unicast NVLS credit address
  int nChannels;
  int nHeads;
  struct ncclShmemCollBuff nvlsShmem;
  void *nvlsShmemHandle;
};

#endif /* CUDART_VERSION >= 12010 */

struct ncclCollNetSharedRes {
  int refCount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  void* resources;
  int nChannels;
  size_t buffSize;
};

struct ncclTransportComm {
  /*当transport可用于通信时，此函数将首先被调用，用于初始化。比如:本地资源分配 + 生成 handshake 材料*/
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*/*本端信息*/, struct ncclPeerInfo*/*对端信息*/, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  /**连接到远程peer */
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  /**拆连接,释放 connector 资源 */
  ncclResult_t (*free)(struct ncclConnector*);
  /**做什么 : 在 proxy 线程 里,一次性建立"跨 comm 共享的资源"。 特点是"每 comm 只调一次,不是每 connection 调一次" 。
  举例 : net_ib.cc 里,一个 comm 内多个 (self, peer) 连接可能 共享同一个 IB Protection Domain (PD) ;
  proxySharedInit 就是建这个 PD、共享的 CQ 等。这样后面每个 connection 只需要建自己的 QP,不用重复建 PD。 */
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  /*Proxy 线程侧的连接资源建立*/
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  /**Proxy 线程侧的握手完成 */
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  /*Proxy 侧连接销毁*/
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  /**Proxy 线程主循环的核心回调*/
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  /**用户 buffer 预注册*/
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  /**用户 buffer 取消预注册*/
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};

struct ncclTransport {
  const char name[8];
  /**判断两个peer是否可以使用该transport通信 */
  ncclResult_t (*canConnect)(int*/*出参，可连接时置为非零*/, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*/*源信息*/, struct ncclPeerInfo*/*对端信息*/);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex);
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode, bool* isAllCudaP2p);
bool ncclP2pUsesMemcpy();

ncclResult_t ncclNvlsInit(struct ncclComm* comm);
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm);
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm);
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv);
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize);
ncclResult_t ncclNvlsFree(struct ncclComm* comm);

enum { collNetRecv=0, collNetSend=1 };
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect);
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail);
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm);
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufUsed, void** outHandle);
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm);
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm);
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm);

ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]);
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm);
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm);

ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle);
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle);
ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);

ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels);

#if CUDART_VERSION >= 12010
ncclResult_t ncclNvlsGroupCreate(struct ncclComm *comm, CUmulticastObjectProp *prop, int rank, unsigned int nranks, CUmemGenericAllocationHandle *mcHandle, char *shareableHandle);
ncclResult_t ncclNvlsGroupConnect(struct ncclComm *comm, char *shareableHandle, int rank, CUmemGenericAllocationHandle *mcHandle);
#endif

ncclResult_t ncclIpcSymmetricInit(struct ncclComm* comm);
ncclResult_t ncclIpcMapSymmetric(struct ncclComm* comm, size_t offset, size_t size, CUmemGenericAllocationHandle memHandle, void** symPtr);
ncclResult_t ncclIpcFreeSymmetric(struct ncclComm* comm, size_t size, void* symPtr);
ncclResult_t ncclIpcSymmetricFinalize(struct ncclComm* comm);
ncclResult_t ncclNvlsSymmetricInit(struct ncclComm* comm);
ncclResult_t ncclNvlsMapSymmetric(struct ncclComm* comm, size_t offset, size_t ucsize, void* ucaddr);
ncclResult_t ncclNvlsFreeSymmetric(struct ncclComm* comm, size_t ucsize, void* ucaddr);
ncclResult_t ncclNvlsSymmetricFinalize(struct ncclComm* comm);

#endif
