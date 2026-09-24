/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "proxy.h"
#include "param.h"
#include "ras.h"
#include "crypt.h"
#include <condition_variable>
#include <mutex>
#include "os.h"
#include <thread>
#include <chrono>
#include <new>

#define BOOTSTRAP_N_CHECK_ABORT 10000
#define BOOTSTRAP_TAG_CONNECT (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)
#define BOOTSTRAP_TAG_GROW_BOUNDARY (0x1 << 27)

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND 1
#define BOOTSTRAP_INIT_TIME_RECV 2
#define BOOTSTRAP_INIT_TIME_RING 3
#define BOOTSTRAP_INIT_TIME_TOTAL 4
#define BOOTSTRAP_INIT_TIME_DELAY 5
#define BOOTSTRAP_INIT_TIME_N 6
#define BOOTSTRAP_INIT_ROOT_WAIT 0
#define BOOTSTRAP_INIT_ROOT_SEND 1
#define BOOTSTRAP_INIT_ROOT_RECV 2
#define BOOTSTRAP_INIT_ROOT_N 3
#define BOOTSTRAP_PROF_OPEN(time) \
  do { \
    time = clockNano(); \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do { \
    time = clockNano() - time; \
  } while (0)

#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))
// returns the first rank associated to the root. must have root >=0
// if root >= n_roots, it does NOT assume periodicity
static int firstRankFromRoot(int root, int n_ranks, int nRoots, int offset) {
  if (root == -1) return 0;
  // only distribute the n_ranks - offset on the roots
  n_ranks -= offset;
  return offset + root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}
// returns the root of a rank, must have rank >=0
// if rank >= n_ranks, it does NOT assume periodicity
static int rootIdFromRank(int rank, int nRanks, int nRoots, int offset) {
  // ranks < offset have no root (id = -1), ranks above the offset will get assigned to their respective root
  if (nRoots == 0 || rank < offset) return -1;
  nRanks -= offset;
  rank -= offset;
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  int D = rmr * (rpr + 1);
  if (rank < D) return rank / (rpr + 1);
  else return (rank - D) / rpr + rmr;
}
// return the number of child for a root, root will be periodized
static int nRankFromRoot(int root, int nRanks, int nRoots, int offset) {
  if (root == -1) return 0;
  nRanks -= offset;
  int ir = BOOTSTRAP_PID(root, nRoots);
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  return rpr + ((ir < rmr) ? 1 : 0);
}
// return the local id of a given rank for a given root
// root will be periodize, rank will not
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  // any rank for root -1 has a local id that is the rank id
  if (root == -1) return rank;
  int ir = BOOTSTRAP_PID(root, nRoots);
  return rank - firstRankFromRoot(ir, nRanks, nRoots, offset);
}
// Check if the given rank is the first rank from the root
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots, offset));
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;/*listen的socket*/
  uint64_t magic;/*ncclBootstrapHandle使用的magic*/
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];/*本机boot期间使用的netif名称*/
static union ncclSocketAddress bootstrapNetIfAddr;/*使用的netif地址*/
static int bootstrapNetInitDone = 0;/*标记是否以上地址已选出*/
static std::mutex bootstrapNetMutex;

NCCL_PARAM(BootstrapNetEnable, "OOB_NET_ENABLE", 0);

/*确定bootstrap接口名称及地址*/
ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {/*加锁再查*/
      const char* env = ncclGetEnv("NCCL_COMM_ID");/*指定一个ip地址+端口*/
      int nIfs = 0;
      if (env) {
    	  /*取此环境变量指定的地址及端口信息*/
        union ncclSocketAddress remoteAddr;
        if (ncclSocketGetAddrFromString(&remoteAddr/*解析COMM_ID指定的远端地址*/, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        /*在本机选与远端地址在同一网段的接口及地址（仅找一个）*/
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
        	/*没有找到*/
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
    	/*没有指定remote地址,找一个接口*/
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      /*显示找到的bootstrap 接口名及地址*/
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t {
  findSubnetIf = -1,
  dontCareIf = -2
};

// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && COMPILER_ATOMIC_LOAD(flag, std::memory_order_acquire)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// send/recv functions
static ncclResult_t netReg(ncclNet_t* net, void* comm/*关联的comm*/, void* data/*内存地址*/, int size/*内存长度*/, void** handle/*出参，注册得到的mr*/) {
	/*注册mr*/
  NCCLCHECK(net->regMr(comm, data/*内存*/, size/*大小*/, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag,
                             void** sendReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
	/*处理发*/
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
	/*检查请求是否完成*/
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag,
                             void** recvReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
	  /*收取*/
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle,
                                void* recvComm, void* recvData, int recvSize, void* recvDataHandle, int tag,
                                volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
    	/*收没有做完，处理收*/
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
    	/*发没有做完，处理发*/
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// Additional socket based functions, first send the size, then send the message
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0) NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data/*出参，收取的内容*/, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));/*收取size*/
  if (recvSize > size) {
	  /*收到的size比参数预期的要大*/
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0) NCCLCHECK(ncclSocketRecv(sock, data, actualSize));/*再收取实际的内容*/
  return ncclSuccess;
}
/**按要求发送和接收指定数据片 */
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock/**发送socket */, void* sendData/**发送数据片指针 */, int sendSize/**发送数据片大小 */,
                                   struct ncclSocket* recvSock/**接收socket */, void* recvData/**接收数据片指针 */, int recvSize/**接收数据片大小 */) {
  /**先交换两边size */                                  
  int senderRecvSize;
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize/**发送数据 */, sizeof(int)/**数据大小 */, recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  /**再交换两边数据 */
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

/**按要求执行双向收发操作 */
static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange size then asynchronously exchange data in send->recv->send->recv order
  int senderRecvSize1, senderRecvSize2;
  /**先交换两边size */
  NCCLCHECK(ncclSocketSendRecv(ops[0].sock, &ops[0].size, sizeof(int), ops[1].sock, &senderRecvSize1, sizeof(int)));
  NCCLCHECK(ncclSocketSendRecv(ops[2].sock, &ops[2].size, sizeof(int), ops[3].sock, &senderRecvSize2, sizeof(int)));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size,
         ops[3].size);
    return ncclInternalError;
  }
  /**更新接收数据片大小 */
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  /**再利用multiop交换数据 */
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

union ringConnectInfo {
  union ncclSocketAddress addr;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct extInfo {
  int rank;                                  // rank of the process reaching out
  int nranks;                                // total number of ranks
  int iroot;                                 // current root index
  int nroots;                                // total number of roots
  int offset;                                // offset for rank distribution
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root
  union ringConnectInfo connectInfo;
};
#define NET_HANDLE(h, rank) ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}
/*bootstrap线程入口*/
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0, offset = 0;
  struct extInfo info;
  union ringConnectInfo* rankInfo = NULL;
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // get zeros for comparison
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));
  ncclOsSetFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);/*初始化socket*/
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);/*指明listenSock,用于accept 新的 client socket*/
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);/*从client socket收取info*/
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);/*然后关闭socket*/

    if (c == 0) {
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      nranks = info.nranks;/*取得rank总数*/
      iroot = info.iroot;
      nroots = info.nroots;
      offset = info.offset;
      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      n2send = nRankFromRoot(iroot, nranks, nroots, offset);
      // offset>0 automatically means that we need to switch to the multiroot logic
      nrecv = n2send + ((offset > 0 || nroots > 1) ? 1 : 0);
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot || offset != info.offset) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d, offset %d "
           "vs %d",
           nranks, info.nranks, nroots, info.nroots, iroot, info.iroot, offset, info.offset);
      goto out;
    }

    localId = localIdFromRoot(info.rank, iroot, nranks, nroots, offset);
    if (localId < 0 || localId >= nrecv) {
      WARN("Bootstrap Root : localId %d is out of range", localId);
      goto out;
    }
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    // if the previous has already checked in, send the newly received handle, if not save the handle for later
    // if we have more than 1 root, I do not own the previous of local_id = 0
    // if we have prev > n2send, we do not send anything
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send &&
        memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }
    // if the next rank has checked in, send the newly received info, if not save the addr for later
    // for nroots >=1, I will always own the information of the next connection
    // if the local_id id must be [0 ; n2send[ otherwise we do not answer
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // here we need to send info only to my own local process
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one,
    // if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)",
        timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo) free(rankInfo);
  if (rankAddressesRoot) free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

/*负责监听listenSock，并启动线程处理bootstrapRoot*/
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv/*是否来自于env*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  std::thread thread;

  /*申请listenSock*/
  NCCLCHECK(ncclCalloc(&listenSock, 1));
  /*初始化socket(bootstrap时期socket)*/
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  /*监听此地址*/
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  /*更新监听地址（比如未指定listen port，被自动分配）*/
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;/*使用的magic*/
  /*创建bootstrap线程*/
  thread = std::thread(bootstrapRoot, args);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  /*使此线程不用join*/
  thread.detach();
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // If comm is provided (grow operation), NCCL_COMM_ID should not be set
    if (comm) {
      WARN("ncclCommGetUniqueId should not be called when NCCL_COMM_ID is set");
      return ncclInvalidUsage;
    }
    // Normal init: use NCCL_COMM_ID from environment
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    /*从环境变量转地址*/
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
  } else {
    if (comm) {
      // comm->childCount will be increment in ncclCommGrow for all existing ranks, use +1 here
      handle->magic = hashCombine(comm->magic, comm->childCount + 1);
    } else {
	  /*没有指定环境变量，magic用随机数*/
      NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    }
    handle->nRanks = comm ? comm->nRanks : 0;
    /*使用bootstrap选中的网络接口及地址(此时本机ip就是root)*/
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    /*创建root*/
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  if (!parent || !handle) {
    WARN("bcastGrowHandle: parent comm and handle must be provided");
    return ncclInvalidArgument;
  }

  // Single rank parent already has the handle, no need to broadcast
  if (parent->nRanks == 1) return ncclSuccess;
  if (isRoot) {
    NCCLCHECK(bootstrapSend(parent->bootstrap, 0, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
    NCCLCHECK(bootstrapSend(parent->bootstrap, parent->nRanks - 1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  } else {
    NCCLCHECK(bootstrapRecv(parent->bootstrap, -1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm/*发送对应的comm*/, *recvComm/*接收对应的comm*/;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;/**ncclNet环 */
    struct {
      struct ncclSocket recv;/**接收socket，用于接收client的连接 */
      struct ncclSocket send;/**发送socket，用于与client通信 */
    } socket;/**socket环 */
  };
};
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;/**ncclNet插件时使用*/
    struct ncclSocket socket; // socket to be used for the ring
  };
};

struct bootstrapState;
struct bootstrapAsyncSend {
  struct bootstrapState* state;
  int peer;
  int tag;
  int size;
  char* data;
  struct bootstrapAsyncSend* next;
};

struct bootstrapState {
  struct bootstrapRing_t ring;/**ring型环收发信息 */
  struct bootstrapListen_t listen;/**监听socket，用于接收client的连接 */
  ncclNet_t* net;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct unexConn* unexpectedConnections;/**挂接非预期的连接 */
  int cudaDev;
  int rank;
  int nranks;/*rank总数*/
  uint64_t magic;
  volatile uint32_t* abortFlag;
  std::mutex asyncSendLock;
  std::condition_variable asyncSendCond;
  struct ncclIntruQueue<struct bootstrapAsyncSend, &bootstrapAsyncSend::next> asyncSendQueue; // in caller order
  ncclResult_t asyncSendError;
  bool asyncSendClosing; // rejects sends once drain/teardown begins
  bool asyncSendSetAbort; // async failure set abortFlag; report it once before abort cleanup
};
#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// helper functions
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket,
                                       union ncclSocketAddress* addr/*出参，监听的地址*/, ncclSocketType type) {
  /*创建listen socket,取listen的地址*/
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}
#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev/*选中的结果*/) {
  static int devOOB = -1;
  if (devOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
    	/*取带外网络接口名称*/
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';/*匹配取反*/
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';/*精确匹配*/
        if (searchExact) userIfEnv++;
        /*由环境变量解析成netIf数组*/
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));/*取设备总数*/
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);/*取设备devId属性*/
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs/*userIfs数组大小*/, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            devOOB = devId;/*带外命中*/
            break;
          }
          devId++;/*尝试下一个*/
        }
        if (devOOB == -1) {
	  /*遍历完所有设备，均未命中*/
          if (!searchNot) {
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          } else {
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "",
                 userIfEnv);
          }
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        devOOB = 0;/*默认选0号设备*/
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      /*指明名称*/
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;/*选中的结果*/
  return ncclSuccess;
}

static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen,
                                   char peerHandle[NCCL_NET_HANDLE_MAXSIZE], void** sendComm/*发送comm*/,
                                   ncclNetDeviceHandle_t** sendDevHandle, void** recvComm/*接收comm*/,
                                   ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    	/*连接到发送comm*/
    if (!*sendComm) NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    	/*获得接收comm*/
    if (!*recvComm) NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}
// With TLS on the bootstrap sockets, a connector cannot finish its handshake (and
// therefore its send) until the acceptor's application accepts the connection and
// runs the server half. The ring connect (connect to next, then accept from prev)
// would deadlock once every rank sits in its connect, so the connect side runs on
// a helper thread while this rank keeps serving its accept side. Plaintext mode
// keeps the stock single-threaded order. One-shot sends get the same property from
// the asynchronous bootstrapSend below.
struct bootstrapThreadOp {
  ncclResult_t (*fn)(void*);
  void* args;
  ncclResult_t result;
};

static void bootstrapThreadRun(void* opaque) {
  struct bootstrapThreadOp* op = (struct bootstrapThreadOp*)opaque;
  op->result = op->fn(op->args);
}

static ncclResult_t bootstrapConcurrent(ncclResult_t (*sendFn)(void*), void* sendArgs, ncclResult_t (*recvFn)(void*),
                                        void* recvArgs) {
  bool encrypted;
  NCCLCHECK(ncclGetCryptConnectionMode(&encrypted));
  if (!encrypted) {
    NCCLCHECK(sendFn(sendArgs));
    NCCLCHECK(recvFn(recvArgs));
    return ncclSuccess;
  }
  // We want to be able to do the initial TLS handshake connecting to the next rank around the ring
  // and accepting from the previous rank around the ring. To keep this simple, we spawn a thread to
  // handle the outgoing side. Unencrypted doesn't need this because no data flows on initial
  // connect.
  struct bootstrapThreadOp op = {sendFn, sendArgs, ncclInternalError};
  std::thread thread;
  STDTHREADCREATE(thread, bootstrapThreadRun, &op);
  ncclResult_t recvRes = recvFn(recvArgs);
  NCCLCHECK(ncclThreadJoin(thread));
  NCCLCHECK(op.result);
  NCCLCHECK(recvRes);
  return ncclSuccess;
}

static ncclResult_t socketConnectOp(void* opaque) {
  NCCLCHECK(ncclSocketConnect((struct ncclSocket*)opaque));
  return ncclSuccess;
}

struct socketAcceptArgs {
  struct ncclSocket* sock;
  struct ncclSocket* listenSock;
};

static ncclResult_t socketAcceptOp(void* opaque) {
  struct socketAcceptArgs* op = (struct socketAcceptArgs*)opaque;
  NCCLCHECK(ncclSocketAccept(op->sock, op->listenSock));
  return ncclSuccess;
}

static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket/**出参，发送socket */,
                                      struct ncclSocket* listenSock/**入参，listen socket */, struct ncclSocket* recvSocket/*出参，发送socket*/, uint64_t magic/**为socket关联的Magic */,
                                      volatile uint32_t* abortFlag/*指针，指向abortFlag,如出错设置此flags，使用指针可与其它结构体共享 */) {
  ncclResult_t ret = ncclSuccess;
  struct socketAcceptArgs acceptArgs = {recvSocket, listenSock};
  NCCLCHECK(ncclSocketInit(recvSocket));/**初始化接收socket */
  NCCLCHECKGOTO(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag/**bootstrap 创建的socket */), ret, fail);
  NCCLCHECKGOTO(bootstrapConcurrent(socketConnectOp, sendSocket, socketAcceptOp, &acceptArgs), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sendSocket);
  (void)ncclSocketClose(recvSocket);
  return ret;
}
static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss, union ncclSocketAddress* peerProxy,
                                uint64_t* peerUDS, struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // pack
  if (peerAddresss) memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy) memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS) memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks) memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // allgather
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // unpack
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss) memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy) memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS) memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks) memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return ncclSuccess;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  /*发送info*/
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);
extern int64_t ncclParamRasEnable();

ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = {0};
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  NEW_NOTHROW(state, bootstrapState);
  ncclIntruQueueConstruct(&state->asyncSendQueue);
  state->rank = rank;/*自身rank*/
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;

  // Set magic: for grow existing ranks, receive from coordinator; otherwise use handle magic.
  // This is consistent with the magic created in ncclCommGetUniqueId.
  if (handles != NULL) {
    // state and comm magic set to the first magic ID
    comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;
  } else if (parent != NULL) {
    comm->magic = state->magic = hashCombine(parent->magic, parent->childCount);
  } else {
    WARN("bootstrapInit: handles and parent are NULL");
    return ncclSystemError;
  }

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (ncclParamBootstrapNetEnable()) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    /*执行listen*/
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                 &STATE_LISTEN(state, net.comm)));
    /*指明连接本端的地址信息*/
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr,
                                 ncclSocketTypeBootstrap));
  }
  // Create socket for root to contact me using the root's magic
  // For grow operations, offset is parent->nRanks - 1 (last existing rank joins the root)
  // For normal init, offset is 0
  int offset = 0;
  if (comm->isGrow) {
    if (parent != NULL) {
      offset = parent->nRanks - 1;
    } else {
      if (handles != NULL) {
        offset = BOOTSTRAP_HANDLE(handles, 0)->nRanks - 1;
      } else {
        WARN("bootstrapInit: handles and parent are NULL");
        return ncclSystemError;
      }
    }
  }
  int curr_root = rootIdFromRank(rank, nranks, nHandles, offset);
  if (curr_root >= 0) {
    NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot,
                                 &info.listenRootAddress, ncclSocketTypeBootstrap));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles, offset);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles, offset) / msg_rate;
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    std::this_thread::sleep_for(std::chrono::microseconds(musec));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  info.offset = offset;
  if (curr_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  if (parent && comm->isGrow && rank != 0) {
    // Grow: Ranks 1 to N-1 use the parent bootstrap to send connection information to the previous rank
    NCCLCHECK(bootstrapSend(parent->bootstrap, rank - 1, 0, &info.connectInfo, sizeof(info.connectInfo)));
  }
  // if needed, send the connection info to the previous root
  // commGrow with more than = 1 rank in the parent comm is a special case of multiroot
  if (((comm->isGrow && parent && (parent->nRanks > 1)) || nHandles > 1) &&
      isFirstFromRoot(rank, curr_root, nranks, nHandles, offset)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles, offset);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // only send if the root is valid, existing rank N-1 will use the bootstrapSend just above
    if (prev_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  if (curr_root >= 0) {
    NCCLCHECK(ncclSocketInit(&sock));
  /*接入新socket*/
    NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  /*读取nextPeer（即我们要发送的对端）*/
    NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
    NCCLCHECK(ncclSocketClose(&sock));
    NCCLCHECK(ncclSocketClose(&listenSockRoot));
  }
  if (parent && comm->isGrow && rank != parent->nRanks - 1) {
    // Grow: Ranks 0 to N-2 use the parent bootstrap to recv connection information to the next rank.
    // This is consistent with the bootstrapSend above.
    NCCLCHECK(bootstrapRecv(parent->bootstrap, rank + 1, 0, &nextPeer, sizeof(nextPeer)));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // accept and connect the ring network
  if (ncclParamBootstrapNetEnable()) {
	  /*开启时，走NCCL net插件，连接发送comm,接受接收comm*/
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle/*发送对应的对端地址*/,
                             &STATE_RING(state, net.sendComm)/*发送*/, &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm)/*接收*/, &STATE_RING(state, net.recvDevHandle),
:q
                             state->abortFlag));
  } else {
    /**走tcp socket，与发送端，建收端建立起连接 */
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                   ncclSocketTypeProxy),
                result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                                   ncclSocketTypeBootstrap),
                result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS thread will take ownership after ncclRasAddRanks succeeds.
    NCCLCHECKGOTO(ncclCalloc(&rasRanks, nranks), result, fail);
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = ncclOsGetPid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks + rank) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks + rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                            state->peerProxyAddressesUDS, rasRanks),
                result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result,
                fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    } else {
      rasRanks = nullptr;
    }
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)",
       timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9, timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9, timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9, timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  free(rasRanks);
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  union ringConnectInfo info;
  union ringConnectInfo nextPeer;
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NEW_NOTHROW_GOTO(state, bootstrapState, ret, fail);
  ncclIntruQueueConstruct(&state->asyncSendQueue);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // create a handle for the others to reach out to me
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                               ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    }
  }

  // Get addr from next rank using the parent's connections
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)),
                ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer,
                              sizeof(union ringConnectInfo)),
                ret, fail);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                                 state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->shareResources) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                     ncclSocketTypeProxy),
                  ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                              state->peerProxyAddressesUDS, NULL),
                  ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE",
        comm, parent, rank, nranks, color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;
  int tag;
};

static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){state->rank, tag};
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap,
                               state->abortFlag),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
static ncclResult_t bootstrapSendSync(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));/*连接到对端peer，并发送socketAckInfo*/
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);/*发送data */
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

// With TLS, sockets that are send-only at the data level still need to receive for TLS handshaking
// so we put them on their own threads with a copy of the payload and run them async.
// bootstrapClose waits for them to drain, and a send failure is stashed on the state and returned
// by the next bootstrap operation or close. It also sets the abort flag so an operation already
// blocked on this state can make progress and fail.
//
// The receiver matches connections by (peer, tag) in accept order, so two sends to the same (peer,
// tag) must reach it in call order (NVLS setup broadcasts to the same peers with the same tag
// several times during init). The in-flight queue keeps caller order; a send's thread waits until it
// is the oldest in-flight send for its (peer, tag) before connecting. Sends to distinct (peer, tag)
// stay concurrent.

// Called with asyncSendLock held, which serializes competing async-send writers.
// asyncSendError is still atomic because bootstrap operations read it without
// taking the lock. Publish asyncSendSetAbort before asyncSendError so acquire
// readers that see asyncSendError also see the marker.
static void bootstrapAsyncSendSetError(struct bootstrapState* state, ncclResult_t res) {
  if (res == ncclSuccess || COMPILER_ATOMIC_LOAD(&state->asyncSendError, std::memory_order_relaxed) != ncclSuccess)
    return;
  COMPILER_ATOMIC_STORE(&state->asyncSendSetAbort, true, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&state->asyncSendError, res, std::memory_order_release);
  COMPILER_ATOMIC_STORE(state->abortFlag, 1u, std::memory_order_release);
}

static bool bootstrapAsyncSendMatches(struct bootstrapAsyncSend* a, struct bootstrapAsyncSend* b) {
  return a == b;
}

static void bootstrapAsyncSendMain(void* opaque) {
  struct bootstrapAsyncSend* op = (struct bootstrapAsyncSend*)opaque;
  struct bootstrapState* state = op->state;
  std::unique_lock<std::mutex> lock(state->asyncSendLock);
  for (;;) {
    struct bootstrapAsyncSend* earlier = ncclIntruQueueHead(&state->asyncSendQueue);
    while (earlier != op && !(earlier->peer == op->peer && earlier->tag == op->tag)) earlier = earlier->next;
    if (earlier == op) break;
    // someone is queued ahead of us with the same (peer, tag), wait for it
    state->asyncSendCond.wait(lock);
  }
  // Once any send on this state has failed, stop sending: a dropped message
  // followed by a delivered one would be matched to the wrong bootstrapRecv on
  // the receiver. The comm is coming down anyway; keep the first error.
  bool skip = (COMPILER_ATOMIC_LOAD(&state->asyncSendError, std::memory_order_relaxed) != ncclSuccess);
  ncclResult_t res = ncclSuccess;
  if (!skip) {
    lock.unlock();
    res = bootstrapSendSync(state, op->peer, op->tag, op->data, op->size);
    lock.lock();
  }
  bootstrapAsyncSendSetError(state, res);
  ncclIntruQueueDelete(&state->asyncSendQueue, op, bootstrapAsyncSendMatches);
  // we're done, wake up anyone that might be waiting for us to finish
  state->asyncSendCond.notify_all();
  lock.unlock();
  free(op->data);
  free(op);
}

static ncclResult_t bootstrapAsyncSendDrain(struct bootstrapState* state) {
  std::unique_lock<std::mutex> lock(state->asyncSendLock);
  state->asyncSendClosing = true;
  while (!ncclIntruQueueEmpty(&state->asyncSendQueue)) state->asyncSendCond.wait(lock);
  return COMPILER_ATOMIC_LOAD(&state->asyncSendError, std::memory_order_acquire);
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  ncclResult_t ret = ncclSuccess;
  bool encrypted;
  NCCLCHECK(ncclGetCryptConnectionMode(&encrypted));
  if (!encrypted) {
    NCCLCHECK(bootstrapSendSync(commState, peer, tag, data, size));
    return ncclSuccess;
  }

  char* copy = nullptr;
  struct bootstrapAsyncSend* op = nullptr;
  ncclResult_t pending;
  bool closing;
  std::thread thread;
  if (size > 0) {
    NCCLCHECKGOTO(ncclCalloc(&copy, size), ret, fail);
    memcpy(copy, data, size);
  }
  NCCLCHECKGOTO(ncclCalloc(&op, 1), ret, fail);
  op->state = state;
  op->peer = peer;
  op->tag = tag;
  op->size = size;
  op->data = copy;

  {
    std::lock_guard<std::mutex> lock(state->asyncSendLock);
    pending = COMPILER_ATOMIC_LOAD(&state->asyncSendError, std::memory_order_acquire);
    if (pending != ncclSuccess) COMPILER_ATOMIC_STORE(&state->asyncSendSetAbort, false, std::memory_order_relaxed);
    closing = state->asyncSendClosing;
    if (pending == ncclSuccess && !closing) {
      STDTHREADCREATE_GOTO(thread, bootstrapAsyncSendMain, ret, fail, op);
      // Append in caller order; bootstrapAsyncSendMain relies on it for the
      // per-(peer, tag) ordering.
      ncclIntruQueueEnqueue(&state->asyncSendQueue, op);
    }
  }
  if (pending != ncclSuccess) {
    WARN("bootstrapSend: an earlier asynchronous send to peer %d failed", peer);
    ret = pending;
    goto fail;
  }
  if (closing) {
    WARN("bootstrapSend: bootstrap state is closing");
    ret = ncclInternalError;
    goto fail;
  }

  thread.detach();
  TRACE(NCCL_BOOTSTRAP, "Async send to peer=%d tag=%d size=%d", peer, tag, size);
  return ncclSuccess;
fail:
  free(copy);
  free(op);
  return ret;
}

// Bootstrap send/receive functions
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  ncclSocketMove(&unex->sock, sock);

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;/*第一个连接，直接挂接在链表头 */
    return ncclSuccess;
  }
  while (list->next) list = list->next;/*找到链表尾 */
  list->next = unex;/*挂接新连接 */
  return ncclSuccess;
}
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock,
                                      int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    // peer < 0 means wildcard (accept from any peer)
    if ((peer < 0 || elem->peer == peer) && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      ncclSocketMove(sock, &elem->sock);
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    (void)ncclSocketClose(&prev->sock);
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected connections first
  int found;
  /*在非预期队列中查找peer,tag对应的连接 */
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new connections
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    /*接收socketAckInfo*/
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    // Match: tag must match, and peer must match (peer < 0 means wildcard)
    if (ack.tag == tag && (peer < 0 || ack.rank == peer)) return ncclSuccess;
    // No match: queue for later and try next connection
    /*将ackInfo放在非预期队列 */
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // Asynchronous send failures are stashed rather than returned to their caller;
  // surface them at the next bootstrap operation so init fails on this rank
  // instead of only hanging the peer.
  ret = COMPILER_ATOMIC_LOAD(&state->asyncSendError, std::memory_order_acquire);
  if (ret != ncclSuccess) COMPILER_ATOMIC_STORE(&state->asyncSendSetAbort, false, std::memory_order_relaxed);
  if (ret != ncclSuccess) {
    WARN("bootstrapRecv: an earlier asynchronous bootstrap send failed");
    return ret;
  }
  /**等待指定peer,tag的连接（否则不返回）*/
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  /**接收数据 */
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/ true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm/*发送侧comm */, void* recvComm/*接收侧comm*/, int rank/*自身rank */, int nranks/*总rank数 */, char* data,
                                     int size/*每个rank片大小*/, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  /*为sendComm,recvComm注册data做为mr*/
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size/*注册大小*/, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size/*注册整块数据 */, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  /*遍历每个rank通信拿到所有rank的数据*/
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    /*对于当前rank来说，第i轮，其收rank-i-1号报文
     * 发rank-i号报文
     * 即首次循环发送自已的那一块，下次发送自已从上家收到的那一块...
     * 接收，先收上家那一块，再收上家的上家那一块
     * */
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;/*取recv指针*/
    void* send_data = data + sslice * size;/*取send指针*/
    /*处理收与发*/
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data/*要发送的数据*/, size, sendDataHandle/*发送mr*/, recvComm, recv_data/*要接收的数据块*/, size, recvDataHandle/*接收mr*/,
                              tag, abortFlag),
                  res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);/*移除注册的mr*/
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock/**本rank的next_rank的发送socket */, struct ncclSocket* prevSock/**本rank的pre_rank的接收socket */, int rank/**本rank编号 */, int nranks/**总rank数 */,
                                        char* data/**交换的信息 */, int size/**每个rank片大小 */) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);/**指明开启做socketRingAllGather */
  int totalSteps = nranks / 2;/**又向ring(之前单向ring只向前后，只向后发，现在前后都可收发) */
  TRACE(NCCL_BOOTSTRAP, "bidirectional bootstrap: totalSteps=%d", totalSteps);
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    // N ranks requires (N-1)/2 steps for the double ring  algorithm.
    // If N is even, the last step is requires a single send/recv
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0/**总rank数为偶数 */);
    // Ring0: ring from previous to next
    /**发送的这一片是向后一个邻居去，故每步从自已的那一片向前偏（随step增大） */
    int sendSliceRing0 = (rank - step + nranks) % nranks;      // Send this slice to next neighbor
    /**接收的这一片是从前一个邻居来，故每步从自已的前一片向前偏（随step增大） */
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;  // Receive this slice from prev neighbor
    // Ring1: ring from next to previous
    /**发送的这一片是向前一个邻居去，故每步从自已的那一片向后偏（随step增大） */
    int sendSliceRing1 = (rank + step) % nranks;               // Send this slice to prev neighbor
    /**接收的这一片是从后一个邻居来，故每步从自已的那一片向后偏（随step增大） */
    int recvSliceRing1 = (rank + step + 1) % nranks;           // Receive this slice from next neighbor
    if (isFinalUnidirectional) {
      /**如果是最后一步，则只向后传递 */
      // Final unidirectional step, only Ring0 is used
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size,
                                   size),
                    res, exit);
    } else {
      /**非最后一步，需要双向传递，按步把数据向前后发送并接收前后数据 */
      // Bidirectional step: Ring0 and Ring1 are used simultaneously
      // clang-format off
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock/*发给后一个邻居 */, data + sendSliceRing0 * size/**数据片 */, size/**数据片大小 */, 0},  // Ring0: send to next
        {NCCL_SOCKET_RECV, prevSock/*从前一个邻居收*/, data + recvSliceRing0 * size, size, 0},  // Ring0: recv from prev
        {NCCL_SOCKET_SEND, prevSock/*发给前一个邻居 */, data + sendSliceRing1 * size, size, 0},  // Ring1: send to prev
        {NCCL_SOCKET_RECV, nextSock/*从后一个邻居收*/, data + recvSliceRing1 * size, size, 0}   // Ring1: recv from next
      };
      // clang-format on
      /**执行双向发送 */
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}
ncclResult_t bootstrapAllGather(void* commState, void* allData/*交换数据的指针 */, int size/*每个rank片大小*/) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;/*自身对应的rank*/
  int nranks = state->nranks;/*总rank数*/

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  if (ncclParamBootstrapNetEnable()) {
    /*开启时，利用netRingAllGather交换数据 */
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm)/*发送*/, STATE_RING(state, net.recvComm)/*接收*/, rank/*所属的rank*/,
                                   nranks/*rank总数*/, (char*)allData, size, state->abortFlag),
                  res, exit);
  } else {
    /*关闭时（默认），利用socketRingAllGather交换数据 */
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send)/**本rank的发送socket */, &STATE_RING(state, socket.recv)/**本rank的接收socket */, rank, nranks,
                                      (char*)allData, size),
                  res, exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9,
        (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank/*自身rank */, int nranks/*总rank数 */, int tag) {
  if (nranks == 1) return ncclSuccess;/*单rank，无需同步*/
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};/**仅用于标记发收动作，数据本身无意义 */
  /*
  轮 0:mask = 1,和距离 1 的邻居互通;
  轮 1:mask = 2,和距离 2 的邻居互通;
  轮 2:mask = 4,和距离 4 的邻居互通;
  轮 k:mask = 2^k,和距离 2^k 的邻居互通;
  直到 mask >= nranks 退出,共 ⌈log2(N)⌉ 轮。
  轮 0 结束:我从 rank-1 收到消息 ⇒ 我知道 rank-1 已经到了 barrier;
  轮 1 结束:我从 rank-2 收到消息,而 rank-2 在轮 0 已经知道 rank-3 到了,所以我此时间接地知道 rank-1、rank-2、rank-3 都到了;
  轮 2 结束:我从 rank-4 收到消息,rank-4 在前两轮已经知道 rank-5,6,7 到了,所以我知道 rank-1..7 都到了;
  轮 k 结束:我掌握了 rank-1..(2^{k+1}-1) 到达的间接见证;
  当 2^k ≥ N 时,全网都被覆盖 ⇒ barrier 完成。
   */
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst]/**dst是索引，发送目标需要从ranks中获取 */ : dst/*发送目标 */, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src/*接收目标 */, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

/**节点内同步barrier（同一物理节点上的多进程仍然是分布式系统） */
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  /**自身为rank,在nranks中同步等待所有rank达到tag位置（这一版本要求算出的rank需要在ranks数组中再映射一次）*/
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  /**自身为rank,在nranks中同步等待所有rank达到tag位置*/
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  /**指明此barrier用时多久 */
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// Like the ring connect: connecting to next and accepting from prev must make
// progress together once the connect includes a TLS handshake.
struct bootstrapPeerSocketOp {
  void* commState;
  int peer;
  int tag;
  struct ncclSocket* sock;
};
static ncclResult_t bootstrapPeerConnectOp(void* opaque) {
  struct bootstrapPeerSocketOp* op = (struct bootstrapPeerSocketOp*)opaque;
  NCCLCHECK(socketConnect(op->commState, op->peer, op->tag, op->sock));
  return ncclSuccess;
}
static ncclResult_t bootstrapPeerAcceptOp(void* opaque) {
  struct bootstrapPeerSocketOp* op = (struct bootstrapPeerSocketOp*)opaque;
  NCCLCHECK(socketAccept(op->commState, op->peer, op->tag, op->sock));
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  ncclResult_t ret = ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];/*前一个rank */
  int nextRank = ranks[(rank + 1) % nranks];/*后一个rank */
  // intraNode bootstrap is done defacto using the socket-based implementation
  struct ncclSocket recvSocket, sendSocket;
  struct bootstrapPeerSocketOp connectOp = {commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket};
  struct bootstrapPeerSocketOp acceptOp = {commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket};
  NCCLCHECK(ncclSocketInit(&recvSocket));
  NCCLCHECKGOTO(ncclSocketInit(&sendSocket), ret, fail);
  NCCLCHECKGOTO(bootstrapConcurrent(bootstrapPeerConnectOp, &connectOp, bootstrapPeerAcceptOp, &acceptOp), ret, fail);

  NCCLCHECKGOTO(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size), ret, fail);

  NCCLCHECKGOTO(ncclSocketClose(&sendSocket), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&recvSocket), ret, fail);

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(&sendSocket);
  (void)ncclSocketClose(&recvSocket);
  return ret;
}

// [IntraNode] in-place Broadcast
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                          int size) {
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root) {
        NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
      }
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData,
                            size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size,
        time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // In-flight asynchronous sends reference this state; let them finish before
  // tearing it down. On abort they exit quickly via the abort flag.
  ncclResult_t asyncSendError = bootstrapAsyncSendDrain(state);
  if (asyncSendError != ncclSuccess && (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0 ||
                                        COMPILER_ATOMIC_LOAD(&state->asyncSendSetAbort, std::memory_order_relaxed))) {
    WARN("bootstrapClose: an asynchronous bootstrap send failed");
    COMPILER_ATOMIC_STORE(&state->asyncSendSetAbort, false, std::memory_order_relaxed);
    return asyncSendError;
  }
  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // proxy things are free'd elsewhere
  free(state->peerP2pAddresses);
  delete state;
  return ncclSuccess;
}
