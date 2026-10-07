/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "device.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include "nccl_net.h"
#include "shmutils.h"
#include "p2p.h"
#include "collectives.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif
#include "os.h"

#include <atomic>
#include <mutex>
#include <condition_variable>

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv,
} ncclPattern_t;

enum ncclProxyOpState {
  ncclProxyOpNone,/*无op*/
  ncclProxyOpReady,
  ncclProxyOpProgress
};

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
  struct {
    int sendSlices;
    int recvSlices;
    int stepSize;
  } bcast;
};

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;/*对应的读写目的rank*/
  int next;
  int nsteps;
  size_t chunkSize;
  size_t sliceSize;
  size_t loopSize;
  size_t loopOffset;
  size_t channelSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t /*ncclPattern_t*/ pattern;/*例如：发/收*/
  uint8_t protocol;
  uint8_t algorithm;
  uint8_t reg;
  // collnet/p2p/coll buffer reg handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  union ncclProxyOpSpecifics specifics;
  int nChannels;
  int nPeers;

  // Profiler plugin
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;
  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  ncclPid_t pid;
  void* profilerContext;
  uint64_t workCounter;

  struct ncclProxyOp* enqNext;
};

struct ncclProxySubArgs;

struct ncclProxyEventHandle {
  void* stepEventHandle;
  struct ncclProxySubArgs* subArgPtr;
};

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int reg;
  // collnet handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
  ssize_t loopSize;
  ssize_t loopOffset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  ssize_t chunkSize;
  int peer;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  int regBufferReady;
  void* requests[NCCL_STEPS];

  // Profiler plugin
  int eActivationMask;
  int rank;
  ncclPid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* kernelEventHandle;
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];
  size_t transSize;
  uint64_t workCounter;

  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;/*有效的subs数目*/
  int done;
  int onePPN;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t protocol;
  uint8_t algorithm;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];
  int nChannels;
  int nPeers;

  int idle;

  // Element linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;

  union ncclProxyOpSpecifics specifics;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements. Each
// p2p work contains a send and recv proxy op hence the 2x before it.
#define MAX_OPS_PER_PEER (2 * MAXCHANNELS * 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

/*此结构位于共享内存（与本机其它rank进程共享）*/
struct ncclProxyOpsPool {
	/*每个元素代表一个待执行的代理操作，每个local rank最多有MAX_OPS_PER_PEER个空间*/
  struct ncclProxyOp ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS];
  /*标记当前待处理操作链表的头部*/
  volatile int nextOps;
  /*标记当前待处理操作链表的尾部*/
  volatile int nextOpsEnd;
  /*每个local rank对应一个freeOps，消费者更新此位置表示本local rank哪个ops可以用了*/
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];
  int syncObjectsInitialized;/*指明是否已初始化*/
  std::mutex mutex;/*锁（进程间共享）*/
  std::condition_variable cond;/*条件变量（进程间共享）*/
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;/*共享内存poll,与其机其它rank共享*/
  /*共享内存handle,此结构的pool与其它local rank共享，此handle是本机的map handle*/
  ncclShmHandle_t handle;
  int count;
  int freeOp;/*缓存本rank可用的首个ops索引，其它的从op->next中取*/
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  ssize_t size;
  char* cudaBuff;
  char* hostBuff;
  // CUDA IPC
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];
  int activeAccept[MAXCHANNELS];
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;/*指向进程间共享的ops pool*/
  ncclShmHandle_t handle;/*opsPool共享内存的详细信息*/
  char opsPoolShmSuffix[16];/*opsPool对应的共享内存路径后缀（前缀大家都一样）*/

  std::thread thread;
  std::atomic<int> stop{0};/*停线程标记*/
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void* opId;
  int respSize;
  bool done;
  void* respBuff;
  ncclResult_t res;
  struct ncclExpectedProxyResponse* next;
};

struct ncclProxyAsyncOp {
  int type;/**操作类型 */
  struct ncclProxyConnection* connection;/*指针，从socket收取*/
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void* opId;
  ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;/*peer的收发socket*/
  int tpRank;/*本rank编号*/
  int tpLocalRank;/*本机rank对应的编号*/
  ncclProxyAsyncOp* asyncOps;/**用于挂异步操作的队列 */
  int asyncOpCounter;
  /*为每个peer连接分配的id号*/
  uint64_t id; // Unique id per accepted peer for connection identification
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv calls
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS support
struct ncclIpcHdr {
  int type;/*消息类型*/
  int rank;/*目的rank*/
  int reqSize;
  int respSize;
  void* opId;/*目的hash(与rank共同组成唯一目的地）*/
  uint64_t data[16]; // 128-bytes
};

struct ncclProxyState {
  int refCount;
  struct ncclComm* comm;
  int tpRank;/*自身rank编号*/
  int tpnRanks;/*系统共有多少rank*/
  int tpLocalnRanks;/*本节点上共有多少个rank编号*/
  int cudaDev;/*对应的gpu*/
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;/*是否支持dmabuf*/
  ncclNet_t* ncclNet;/*网络插件（比如ncclNetIb)*/
  ncclCollNet_t* ncclCollNet;/*collnet插件*/
  struct ncclGinState* ginState;
  uint32_t* abortFlag;
  bool directMode;
  struct ncclMemManager* memManager;  // Shared memory manager for proxy allocations
  // Service threads
  std::thread thread;
  std::thread threadUDS;
  struct ncclSocket* listenSock;/*记录proxy监听的地址*/
  struct ncclIpcSocket ipcSock;/*记录unix socket（监听了rank+uds[rank])地址,ncclProxyServiceUDS负责处理*/
  int stop;
  ncclResult_t asyncResult;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;/*各rank监听的地址*/
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;/*每个本机rank一个proxyOps结构（通过localRank索引）*/
  void** sharedDevMems;
  int peerArraySize;  // Size of peerSocks/proxyOps/sharedDevMems arrays (tpNRanks)
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  /*各rank uds的编号*/
  uint64_t* peerAddressesUDS; // cuMem API support (UDS)

  // Progress thread
  struct ncclProxyProgressState progressState;/*progress线程state*/

  // Network plugin
  void* netContext;
  ncclNetAttr_t netAttr;
  void* collNetContext;

  // Profiler plugin
  void* profilerContext;

  // Queue of expected responses from the proxy
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized = 0,
  connInitialized = 1,
  connSharedInitialized = 2,
  connSetupDone = 3,
  connConnected = 4,
  numConnStates = 5
};

struct proxyMemHandle {
  void* handle;
  struct proxyMemHandle* next;
};

struct ncclProxyConnection {
  int send/*是否为发送*/, transport/*连接对应的transport,比如TRANSPORT_NET,其相对指明了ncclTransports数组的下标*/, shared;
  int tpLocalRank/*本机此rank编号*/, sameProcess/*是否同进程*/;
  struct ncclSocket* sock;/*连接使用的socket*/
  /*连接从属的peer(通过此id可以找到peer)*/
  uint64_t peerId; // Initialized with peer->id, used for connection checks
  struct ncclTransportComm* tcomm;/*此连接对应的transport comm（比如同机的，不同机的）,见ncclTransports数组成员*/
  struct ncclProxyArgs* proxyAppend;
  struct ncclProxyArgs** proxyAppendPtr;
  void* transportResources;
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
  int needsProxyProgress;
  /*记录已注册的memory(这个是由net层面具体完成后直接入队的）*/
  struct ncclIntruQueue<struct proxyMemHandle, &proxyMemHandle::next> proxyMemHandleQueue;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool* justInquire);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses,
                           uint64_t* peerAddressesUDS);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank,
                              struct ncclProxyConnector* proxyConn);

// NB: ncclProxyMsgTypeStr[] in proxy.cc needs to match
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8,
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)
  ncclProxyMsgQueryFd = 10,
  ncclProxyMsgRegister = 11,
  ncclProxyMsgDeregister = 12
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result
// is received
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                   int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff,
                                   void* opId);

// UDS support
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void* handle, ncclIpcFd* convertedFd);
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn,
                                            ncclIpcFd localFd, ncclIpcFd* rmtFd);
ncclResult_t ncclProxyClientBatchQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn,
                                                 ncclIpcFd* localFds, ncclIpcFd* rmtFds, int numSegments);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
