/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "socket.h"
#include "utils.h"
#include "os.h"
#include "crypt.h"
#include <stdlib.h>
#include <cstdlib>

#include "param.h"
#include <time.h>
#include <mutex>

/**重试次数 */
NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
/**重试时间间隔 */
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);
NCCL_PARAM(PollTimeOut, "SOCKET_POLL_TIMEOUT_MSEC", 0);
NCCL_PARAM(SocketMaxRecvBuff, "SOCKET_RCVBUF", -1);
NCCL_PARAM(SocketMaxSendBuff, "SOCKET_SNDBUF", -1);

uint64_t ncclSocketDefaultMagic(void) {
  /* Default is the historical constant; env may override on first init. */
  static uint64_t cached = NCCL_SOCKET_MAGIC;
  static std::once_flag once;
  std::call_once(once, []() {
    const char* env = ncclGetEnv("NCCL_SOCKET_MAGIC");
    bool fromEnv = false;
    if (env != NULL && env[0] != '\0') {/*环境变量有设置magic*/
      char* endptr = NULL;
      unsigned long long v = std::strtoull(env, &endptr, 0);/*转数字*/
      if (endptr != env && endptr != NULL && *endptr == '\0') {
        cached = (uint64_t)v;/*缓存*/
        fromEnv = true;
      } else {
        INFO(NCCL_ENV, "NCCL_SOCKET_MAGIC invalid value \"%s\", using built-in default", env);
      }
    }
    INFO(NCCL_ENV, "Socket handshake magic 0x%016llx (%s)", (unsigned long long)cached,
         fromEnv ? "NCCL_SOCKET_MAGIC" : "built-in default");
  });
  return cached;/*返回缓存值*/
}

void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  memcpy(dst, src, sizeof(struct ncclSocket));
  ncclCryptRebindSocket(dst);
  src->socketDescriptor = NCCL_INVALID_SOCKET;
  src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
  src->crypto = nullptr;
}

static ncclResult_t socketProgressRaw(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                      bool* pclosed = nullptr) {
  int closed;
  NCCLCHECK(ncclOsSocketProgressOpt(op, sock, ptr, size, offset, 0, &closed));
  if (closed) {
	  /*需要关闭socket*/
    if (pclosed) {
      *pclosed = true;
      return ncclSuccess;
    } else {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("socketProgress: Connection closed by remote peer %s",
           ncclSocketToString(&sock->addr, line, /*numericHostForm*/ 0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

static ncclResult_t socketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                   bool* pclosed = nullptr) {
  if (sock->crypto) {
    if (size <= *offset) return ncclSuccess;
    if (op == NCCL_SOCKET_SEND) NCCLCHECK(ncclCryptSocketSend(sock, ptr, size, offset, pclosed));
    else NCCLCHECK(ncclCryptSocketRecv(sock, ptr, size, offset, pclosed));
  } else {
    NCCLCHECK(socketProgressRaw(op, sock, ptr, size, offset, pclosed));
  }
  return ncclSuccess;
}

static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  while (*offset < size) {
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
    // If we have more data to read or write, use the poll system call to wait
    // until the socket becomes readable or writable again.
    if ((*offset < size) && ncclParamPollTimeOut()) {
      ncclOsPollSocket(sock->socketDescriptor, op);
    }
  }
  return ncclSuccess;
}

/*由socket地址获取port*/
uint16_t ncclSocketToPort(union ncclSocketAddress* addr) {
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (ncclOsSocketIsValid(sock)) {
      SYSCHECK(shutdown(sock->socketDescriptor, how), "shutdown");
    }
    sock->state = ncclSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* socketDescriptor) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (socketDescriptor) *socketDescriptor = sock->socketDescriptor;
  return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(ncclSocketDescriptor socketDescriptor, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketSetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->socketDescriptor = socketDescriptor;
  return ncclSuccess;
}

/*绑定socket中设置的地址*/
ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclSocketListen: socket is invalid");
    return ncclInvalidArgument;
  }

  if (ncclSocketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    /*已知设置了port,设置port reuse*/
    int opt = 1;
#if defined(NCCL_OS_LINUX)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
#elif defined(NCCL_OS_WINDOWS)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, (char*)&opt, sizeof(opt)), "setsockopt");
#endif
#endif
  }

  // addr port should be 0 (Any port)
  /*绑定此port（可能为0）*/
  SYSCHECK(bind(sock->socketDescriptor, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  /*取设置的port地址（0时分配的port）*/
  SYSCHECK(getsockname(sock->socketDescriptor, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(NCCL_INIT | NCCL_NET, "Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  SYSCHECK(listen(sock->socketDescriptor, 16384), "listen");/*执行listen系统调用*/

  // Set acceptSocketDescriptor to the same value as socketDescriptor for listening sockets
  sock->acceptSocketDescriptor = sock->socketDescriptor;
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, const int numericHostForm /*= 1*/) {
  const struct sockaddr* saddr = &addr->sa;
  char host[NI_MAXHOST], service[NI_MAXSERV];
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL) goto fail;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) goto fail;
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  if (getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag)) goto fail;
  sprintf(buf, "%s<%s>", host, service);/*显示绑定地址*/
  return buf;
fail:
  if (buf) buf[0] = '\0';
  return buf;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
int ncclEnvSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");
  if (env == NULL) return family;/* 未指定socket family,使用默认值（-1） */

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0) family = AF_INET6; // IPv6
  return family;
}

ncclResult_t ncclFindInterfaces(char* ifNames/*出参，找到的接口*/, union ncclSocketAddress* ifAddrs/*出参，找到的接口地址*/, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  static int shownIfName = 0;
  // Allow user to force the INET socket family selection
  int sock_family = ncclEnvSocketFamily();/*要匹配的socket family*/
  // User specified interface
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  *nIfs = 0;
  if (env && strlen(env) > 1) {
    /*环境变量指定了接口名称，使用这个名称*/
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);/*打印用户通过环境变量指定的接口名称 */
    NCCLCHECK(ncclOsFindInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));/*结合系统实际接口，确定匹配的接口*/
  } else {
	  /*未指定名称*/
    // Try to automatically pick the right one
    // Start with IB
    NCCLCHECK(ncclOsFindInterfaces("ib"/*选择ib开头的接口*/, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // else see if we can get some hint from COMM ID
    if (*nIfs == 0) {
      const char* commId = ncclGetEnv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union ncclSocketAddress idAddr;
        NCCLCHECK(ncclSocketGetAddrFromString(&idAddr, commId));
        NCCLCHECK(ncclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, nIfs));
      }
    }
    // Then look for anything else (but not docker,lo, or virtual)
    /*仍没有找到，找这些接口之外的*/
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("^docker,lo,virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
    /*仍没有找到，在这三个中选*/
    // Finally look for docker, then lo.
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua/*出参，解析ip_port_pair得到的地址*/, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
	  /*不可为空*/
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';/*以'['来分辨ipv6地址*/
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1/*仅一个*/) != 1) {
    	/*只容许一个*/
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    /*按host/ip取地址*/
    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      freeaddrinfo(p);
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;/*%号出现的位置*/
      if (ip_port_pair[i] == ']') break;/*v6地址结束*/
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[16];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);/*取地址*/
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);/*取port*/
    int port = atoi(port_str);
    /*取通过%分隔指明的接口名称*/
    if (!global_scope) {
      // If not global scope, we need the intf name
      strncpy(if_name, ip_port_pair + j + 1, i - j - 1);
    }

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock == NULL) {
    WARN("ncclSocketGetAddr: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  memcpy(addr, &sock->addr, sizeof(union ncclSocketAddress));/*取socket地址*/
  return ncclSuccess;
}

static void socketResetAccept(struct ncclSocket* sock) {
  ncclOsSocketResetAccept(sock);
  ncclCryptFree(sock->crypto);
  sock->crypto = nullptr;
}

/**接入了client与server的magic,type字节，验证是否一致 */
static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  NCCLCHECK(ncclOsSocketSetFlags(sock));

  ncclResult_t ret;
  do {
    // Runs the TLS handshake and reads the hello in encrypted mode, or reads the
    // stock plaintext hello otherwise; all of the socket I/O lives in crypt.cc.
    enum ncclCryptHelloVerdict verdict;
    NCCLCHECK(ret = ncclCryptAcceptHello(sock, &verdict));
    if (verdict == NCCL_CRYPT_HELLO_VERDICT_RESET) {
      socketResetAccept(sock);
      return ncclSuccess;
    }
    if (verdict == NCCL_CRYPT_HELLO_VERDICT_READY) {
      sock->state = ncclSocketStateReady;
      return ncclSuccess;
    }
  } while (sock->asyncFlag == 0 && ret == ncclInProgress);
  return ncclSuccess;
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclOsSocketPollConnect(sock));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  ncclResult_t ret;
  do {
    NCCLCHECK(ret = ncclCryptConnectHello(sock));
  } while (sock->asyncFlag == 0 && ret == ncclInProgress);
  if (ret == ncclInProgress) return ncclSuccess;
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

static ncclResult_t socketProgressState(struct ncclSocket* sock) {
	/*按状态处理*/
  if (sock->state == ncclSocketStateAccepting) {
    /*对于等待accept的socket，尝试accept*/
    NCCLCHECK(ncclOsSocketTryAccept(sock));
  }
  if (sock->state == ncclSocketStateAccepted) {
    /**对于已accept到clent的调用（读取magic,type） */
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == ncclSocketStateConnecting) {
    /*对于正在连接状态，启动执行connect*/
    NCCLCHECK(ncclOsSocketStartConnect(sock));
  }
  if (sock->state == ncclSocketStateConnectPolling) {
    /**未拿到连接结果，轮询时调用 */
    NCCLCHECK(ncclOsSocketPollConnect(sock));
  }
  if (sock->state == ncclSocketStateConnected) {
    /*connect成功后调用（告知对端magic,type,以便达到state ready状态 */
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running/*出参，是否已达到ready状态 */) {
  if (sock == NULL) {
    *running = 0;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) {
    WARN("ncclSocketReady: unexpected socket state %d", sock->state);
    return ncclRemoteError;
  }
  *running = (sock->state == ncclSocketStateReady) ? 1/*未达到state ready，则置为0*/ : 0;
  if (*running == 0) {
	  /*未达到ready,继续运行*/
    NCCLCHECK(socketProgressState(sock));
    if (sock->state == ncclSocketStateBadHandshake) sock->state = ncclSocketStateAccepting;
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

/*执行连接*/
ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif

  if (sock == NULL) {
    /**不能为空 */
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    /**不能未初始化的socket */
    WARN("ncclSocketConnect: socket is invalid");
    return ncclInvalidArgument;
  }

  if (sock->state != ncclSocketStateInitialized) {
    /**必须已完成初始化 */
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT | NCCL_NET, "Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  NCCLCHECK(ncclCryptStartConnect(sock));

  sock->state = ncclSocketStateConnecting;/*指定为正在连接状态*/
  sock->finalizeCounter = 0;
  do {
    NCCLCHECK(socketProgressState(sock));/*执行连接，如出错直接返回*/
  } while (sock->asyncFlag == 0/**同步操作 */ &&
           (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0)/**没有abort */ &&
           (sock->state == ncclSocketStateConnecting || sock->state == ncclSocketStateConnectPolling ||
            sock->state == ncclSocketStateConnected))/**未达到ready ready状态时，继续轮询 */;

  /**已设置abortFlag，返回错误 */
  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  /*按状态跳转*/
  switch (sock->state) {
  case ncclSocketStateConnecting:
  case ncclSocketStateConnectPolling:
  case ncclSocketStateConnected:
  case ncclSocketStateReady:
    return ncclSuccess;
  case ncclSocketStateError:
    return ncclSystemError;
  default:
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    return ncclInternalError;
  }
}

/*接入新的client*/
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  ncclResult_t ret = ncclSuccess;

  if (listenSock == NULL || sock == NULL) {
    /**listen socket不得为0，recv socket不得为NULL*/
    WARN("ncclSocketAccept: pass NULL socket");
    ret = ncclInvalidArgument;
    goto exit;
  }
  if (listenSock->state != ncclSocketStateReady) {
    /**listen socket必须为ready状态 */
    WARN("ncclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == ncclSocketStateError) ret = ncclSystemError;
    else ret = ncclInternalError;
    goto exit;
  }

  if (!ncclOsSocketDescriptorIsValid(sock->acceptSocketDescriptor)) {
    // Clone the listener configuration; the accepted socket owns its own TLS state.
    /**此sock首次应用于accept调用时，置为accepting状态*/
    memcpy(sock, listenSock, sizeof(struct ncclSocket));/**这里的复制大有深意，复用listenSock的magic,type */
    sock->acceptSocketDescriptor = listenSock->acceptSocketDescriptor;
    sock->state = ncclSocketStateAccepting;
    sock->finalizeCounter = 0;
    sock->crypto = nullptr;
  }

  /*执行accept*/
  do {
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);
    if (sock->state == ncclSocketStateBadHandshake) {
      // Most likely some issue with magic.  We will retry from the beginning, unless the caller requested not to.
      sock->state = ncclSocketStateAccepting;
      if (!retry) break;
    }
  } while (sock->asyncFlag == 0 /**同步操作 */&&
           (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0) /**没有abort */&&
           (sock->state == ncclSocketStateAccepting || sock->state == ncclSocketStateAccepted))/**未接入client，继续轮询 */;

  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  switch (sock->state) {
  case ncclSocketStateAccepting:
  case ncclSocketStateAccepted:
  case ncclSocketStateReady:
    ret = ncclSuccess;
    break;
  case ncclSocketStateError:
    ret = ncclSystemError;
    break;
  default:
    WARN("ncclSocketAccept: wrong socket state %d", sock->state);
    ret = ncclInternalError;
    break;
  }

exit:
  return ret;
}

/*初始化sock*/
ncclResult_t ncclSocketInit(struct ncclSocket* sock/*出参，待初始化的socket*/, const union ncclSocketAddress* addr/*要设置的地址*/, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  sock->errorRetries = 0;/**重试次数置为0*/
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = ncclSocketStateInitialized;/*状态指定为初始*/
  sock->magic = magic;
  sock->type = type;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  sock->customRetry = customRetry;
  sock->finalizeCounter = 0;
  sock->crypto = nullptr;
#ifdef NCCL_OS_WINDOWS
  sock->socketBlockingMode = 1;
#endif

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));/*设置地址*/
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
    	/*只考虑v4,v6两种*/
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
           ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);/**设置地址长度 */
    // in case of error, we close the descriptor before returning as it's unclear if the caller has to
    // use ncclSocketClose for cleanup
    NCCLCHECKGOTO(ncclOsSocketResetFd(sock), ret, fail);/*初始化socket*/
  } else {
	  /*没有指定地址，地址置为0，且不初始化socket*/
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  (void)ncclSocketClose(sock);
  goto exit;
}

ncclResult_t ncclSocketProgress(int op/*执行收或者发操作*/, struct ncclSocket* sock, void* ptr, int size, int* offset/*出参，偏移量*/, bool* closed) {
  if (sock == NULL) {
    WARN("ncclSocketProgress: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketProgress(op, sock, ptr, size, offset, closed));
  return ncclSuccess;
}

ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  if (sock == NULL) {
    WARN("ncclSocketWait: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketSend: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) {
    WARN("ncclSocketSend: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  /*发送并等待,直到ptr指向的size字节发送完成*/
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady && sock->state != ncclSocketStateTerminating) {
    WARN("ncclSocketRecv: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return ncclSuccess;
}

/**按要求发送和接收指定数据片 */
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock/**发送socket */, void* sendPtr/**发送数据片指针 */, int sendSize/**发送数据片大小 */, struct ncclSocket* recvSock/**接收socket */,
                                void* recvPtr/**接收数据片指针 */, int recvSize/**接收数据片大小*/) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    /**不能为空 */
    WARN("ncclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return ncclInternalError;
  }
  if (sendSock->state != ncclSocketStateReady ||
      (recvSock->state != ncclSocketStateReady && recvSock->state != ncclSocketStateTerminating)) {
        /**状态有误 */
    WARN("ncclSocketSendRecv: socket state (%d/%d) is not ready", sendSock->state, recvSock->state);
    return ncclInternalError;
  }

  /**持续直到发送和接收完成，或者出错 */
  while (sendOffset < sendSize || recvOffset < recvSize) {
    /*未完成发送，则执行发送*/
    if (sendOffset < sendSize) NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset));
    /*未完成接收，则执行接收*/
    if (recvOffset < recvSize) NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset));
  }
  return ncclSuccess;
}

/**按要求执行多个操作 */
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  if (ops == NULL || numOps <= 0) {
    /**操作不能为空 */
    WARN("ncclSocketMultiOp: invalid arguments ops=%p numOps=%d", ops, numOps);
    return ncclInvalidArgument;
  }

  /**不能没有socket */
  for (int i = 0; i < numOps; i++) {
    if (ops[i].sock == NULL) {
      WARN("ncclSocketMultiOp: invalid socket at index %d", i);
      return ncclInvalidArgument;
    }
    ops[i].offset = 0;/**起始偏移必须为0 */
  }
  int completedOps = 0, i = 0;
  while (completedOps < numOps) {
    /**持续直到所有操作完成 */
    if (ops[i].offset < ops[i].size) {
      NCCLCHECK(socketProgress(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset));
      if (ops[i].offset >= ops[i].size) completedOps++;/*记录完成操作的数目*/
    }
    i = (i + 1) % numOps;
  }
  return ncclSuccess;
}
// Receive or detect connection closed
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = false;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, ptr, size, &offset, closed));
  if (*closed) return ncclSuccess;
  if (offset == 0 && !blocking) return ncclInProgress;
  while (offset < size) {
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, ptr, size, &offset, closed));
    if (*closed) return ncclSuccess;
  }
  return ncclSuccess;
}
