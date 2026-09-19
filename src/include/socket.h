/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,/**socket已初始化 */
  ncclSocketStateAccepting = 2,/**等待accept调用 */
  ncclSocketStateAccepted = 3,/**已accept */
  ncclSocketStateConnecting = 4,/**连接中 */
  ncclSocketStateConnectPolling = 5,/**等待连接完成结果 */
  ncclSocketStateConnected = 6,/**已连接 */
  ncclSocketStateReady = 7,
  ncclSocketStateTerminating = 8,
  ncclSocketStateClosed = 9,
  ncclSocketStateError = 10,/**出错 */
  ncclSocketStateNum = 11
};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,
  ncclSocketTypeBootstrap = 1,/**bootstrap 创建的socket */
  ncclSocketTypeProxy = 2,
  ncclSocketTypeNetSocket = 3,
  ncclSocketTypeNetIb = 4,
  ncclSocketTypeRasNetwork = 5
};

struct ncclSocket {
  int fd;/*client fd*/
  int acceptFd;/*自此fd接入client*/
  int errorRetries;/*重试次数*/
  union ncclSocketAddress addr;
  volatile uint32_t* abortFlag;/**指针，指向abortFlag,如出错设置此flags，使用指针可与其它结构体共享此标记变量 */
  int asyncFlag;/*指明是否异步（非阻塞）操作（默认为0，同步操作 ）*/
  enum ncclSocketState state;/**用于指明当前socket状态 */
  int salen;/*地址长度*/
  uint64_t magic;/*为此socket关联的Magic信息*/
  enum ncclSocketType type;/*socket类型，比如在bootstrap阶段创建的socket */
  int customRetry;/*是否custom自已尝试重连*/
  /**异步操作时，用于记录已完成发送/接收的字节数，用于从此位置继续发送/接收 */
  int finalizeCounter; // Used to keep track of initial handshake for async sockets.
  char finalizeBuffer[sizeof(uint64_t)]; // Used to keep track of initial handshake for async sockets.
};
struct ncclSocketOp {
  int op;                    // NCCL_SOCKET_SEND or NCCL_SOCKET_RECV
  struct ncclSocket* sock;   // Socket to operate on
  /**数据片指针*/
  void* ptr;                 // Data pointer
  int size;                  // Size of data
  /**当前操作在数据片中的偏移量*/
  int offset;                // Current progress offset
};
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1);
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found);
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs);

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr = NULL, uint64_t magic = NCCL_SOCKET_MAGIC, enum ncclSocketType type = ncclSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0/**默认是同步操作 */, int customRetry = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);
// Connect to sock->addr. sock->fd is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock);
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd);
ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed = NULL);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize);
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking);
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how);
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait = false);
#endif
