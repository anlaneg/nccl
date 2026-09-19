/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket.h"
#include "utils.h"
#include <stdlib.h>

#include <unistd.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "param.h"
#include <time.h>

/**重试次数 */
NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
/**重试时间间隔 */
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);
static void msleep(unsigned int time_msec) {
  const long c_1e6 = 1e6;
  struct timespec tv = (struct timespec){
      .tv_sec = time_msec / 1000,
      .tv_nsec = (time_msec % 1000) * c_1e6,
  };
  nanosleep(&tv, NULL);
}

/*socket收发处理*/
static ncclResult_t socketProgressOpt(int op, struct ncclSocket* sock, void* ptr/*buffer*/, int size/*buffer大小*/, int* offset/*入出参，偏移量*/, int block/*是否阻塞socket*/, int* closed/*出参，是否关闭*/) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN+1];
  do {
	  /*检查是否recv,则接收buffer*/
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT/*本次调用指明非阻塞*/);
    /*检查是否send,则发送buffer*/
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->fd, data+(*offset), size-(*offset), block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL/*禁止断开的时候向进程发 SIGPIPE 信号*/);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
    	/*收到的长度为0，连接关闭*/
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == -1) {
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;/*发送失败，关闭*/
        return ncclSuccess;
      }
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
    	  /*遇到其它错误，返回错误*/
        WARN("socketProgressOpt: Call to %s %s failed : %s", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
             ncclSocketToString(&sock->addr, line), strerror(errno));
        return ncclRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;/*增加偏移*/
    if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) {
      INFO(NCCL_NET, "socketProgressOpt: abort called");
      return ncclInternalError;
    }
  } while (sock->asyncFlag == 0/*无异步标记*/ && bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

/*按op操作socket,执行收/发*/
static ncclResult_t socketProgress(int op/*收或者发*/, struct ncclSocket* sock, void* ptr, int size/*内容长度*/, int* offset/*出参，偏移量*/, int* pclosed = NULL/*出参，连接是否关闭*/) {
  int closed;
  /*按op操作*/
  NCCLCHECK(socketProgressOpt(op, sock, ptr, size, offset, 0 /*block*/, &closed/*出参，连接是否关闭*/));
  if (closed) {
	  /*需要关闭socket*/
    if (pclosed) {
      *pclosed = closed;
      return ncclSuccess;
    } else {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("socketProgress: Connection closed by remote peer %s",
           ncclSocketToString(&sock->addr, line, /*numericHostForm*/0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset/**入出参，偏移量,比如发送偏移量*/) {
  /*循环直到op操作处理完成*/
  while (*offset < size)
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
  return ncclSuccess;
}

/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm /*= 1*/) {
  const struct sockaddr *saddr;
  char host[NI_MAXHOST], service[NI_MAXSERV];
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL) goto fail;
  saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) goto fail;
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  if (getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag)) goto fail;
  sprintf(buf, "%s<%s>", host, service);
  return buf;
fail:
  if (buf)
    buf[0] = '\0';
  return buf;
}

/*取地址中指明的port*/
static uint16_t socketToPort(union ncclSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;/* 未指定socket family,使用默认值（-1） */

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static ncclResult_t findInterfaces(const char* prefixList/*匹配指令*/, char* names/*出参，以固定大小保存接口名称*/, union ncclSocketAddress *addrs/*出参，以固定大小保存接口对应的地址*/, int sock_family/*要匹配的family*/,
                                   int maxIfNameSize, int maxIfs/*查找的最大数目*/, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif
  struct netIf userIfs[MAX_IFS];
  /*支持对结果取反*/
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  /*支持相等匹配*/
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs/**出参，接口解析列表 */, MAX_IFS);

  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");/**获取系统接口列表 */
  for (interface = interfaces; interface && *found < maxIfs/*查找到的数目必须小于要求的最大数*/; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;/*跳过无地址的*/

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;/** 只支持IPv4和IPv6（其它地址不看） */

    /* Only consider running interfaces, i.e. UP and physically attached. */
    if (!(interface->ifa_flags & IFF_RUNNING)) continue;/** 只考虑运行中的接口 */

    /** 指出发现了哪些接口，接口地址是什么 */
    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, ncclSocketToString((union ncclSocketAddress *) interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;/** 只考虑指定的socket family */

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;/** ipv6不考虑loopback地址 */
    }

    // check against user specified interfaces
    /*发现的接口与用户配置进行匹配*/
    if (!(matchIfList(interface->ifa_name/**接口名称*/, -1/*不匹配端口*/, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;/*忽略失配*/
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < *found; i++) {
    	/*检查是否重复*/
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + (*found)*maxIfNameSize, interface->ifa_name, maxIfNameSize);/*固定数组形式排列*/
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memset(addrs + *found, '\0', sizeof(*addrs));
      memcpy(addrs + *found, interface->ifa_addr, salen);/*记录接口地址*/
      (*found)++;
    }
  }

  freeifaddrs(interfaces);
  return ncclSuccess;
}

static bool matchSubnet(struct ifaddrs local_if, union ncclSocketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;/*两者family不同*/
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;/*两者是否在同网段*/
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    INFO(NCCL_NET, "Net : Unsupported address family type");
    return false;
  }
}

/*读取本机接口信息，然后遍历并检查是否与remoteAddr有在同一网段的，返回选中的接口及地址（仅找一个）*/
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName/*出参，选中的接口*/, union ncclSocketAddress* localAddr/*出参，选中的本端地址*/,
                                          union ncclSocketAddress* remoteAddr/*远端地址*/, int ifNameMaxSize, int* found/*出参，是否找到*/) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  char line_a[SOCKET_NAME_MAXLEN+1];
#endif
  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");/*取本机接口信息*/
  for (interface = interfaces; interface && !*found/*没找到继续循环*/; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;/*仅ipv4/ipv6*/

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;/*忽略与远端地址非同网段的情况*/
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddr, interface->ifa_addr, salen);/*记录选中的本端地址*/

    // Store the interface name
    strncpy(ifName, interface->ifa_name, ifNameMaxSize);/*记录选中的地址*/

    TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
    *found = 1;/*找到*/
  }

  freeifaddrs(interfaces);
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
    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
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

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);/*取地址*/
    strncpy(port_str, ip_port_pair+i+2, len-i-1);/*取port*/
    int port = atoi(port_str);
    /*取通过%分隔指明的接口名称*/
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

ncclResult_t ncclFindInterfaces(char* ifNames/*出参，找到的接口*/, union ncclSocketAddress *ifAddrs/*出参，找到的接口地址*/, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  static int shownIfName = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();/*要匹配的socket family*/
  // User specified interface
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  *nIfs = 0;
  if (env && strlen(env) > 1) {
    /*环境变量指定了接口名称，使用这个名称*/
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);/*打印用户通过环境变量指定的接口名称 */
    NCCLCHECK(findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));/*结合系统实际接口，确定匹配的接口*/
  } else {
	  /*未指定名称*/
    // Try to automatically pick the right one
    // Start with IB
    NCCLCHECK(findInterfaces("ib"/*选择ib开头的接口*/, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
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
    if (*nIfs == 0) NCCLCHECK(findInterfaces("^docker,lo,virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    /*仍没有找到，在这三个中选*/
    // Finally look for docker, then lo.
    if (*nIfs == 0) NCCLCHECK(findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(findInterfaces("virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketListen: file descriptor is -1");
    return ncclInvalidArgument;
  }

  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    /*设置port reuse*/
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");/*绑定此port（可能为0）*/

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");/*取设置的port地址（0时分配的port）*/

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sock->fd, 16384), "listen");/*执行listen系统调用*/
  sock->state = ncclSocketStateReady;
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

static ncclResult_t socketTryAccept(struct ncclSocket* sock) {
  socklen_t socklen = sizeof(union ncclSocketAddress);
  /*接收client*/
  sock->fd = accept(sock->acceptFd, (struct sockaddr*)&sock->addr, &socklen);
  if (sock->fd != -1) {
	  /*接入成功，置accepted状态*/
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN ||
             errno == ENONET || errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH ||
             errno == EINTR) {
    /* per accept's man page, for linux sockets, the following errors might be already pending errors
     * and should be considered as EAGAIN. To avoid infinite loop in case of errors, we use the retry count*/
    if (++sock->errorRetries == ncclParamRetryCnt()) {
    	/*达到最大重试次数*/
      WARN("socketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries, strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET|NCCL_INIT, "Call to accept returned %s, retrying", strerror(errno));/**后续重试 */
  } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
	  /*accept出错*/
    WARN("socketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

NCCL_PARAM(SocketMaxRecvBuff, "SOCKET_RCVBUF", -1);
NCCL_PARAM(SocketMaxSendBuff, "SOCKET_SNDBUF", -1);

/**设置socket */
static ncclResult_t socketSetFlags(struct ncclSocket* sock) {
  const int one = 1;
  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((sock->asyncFlag || sock->abortFlag) && sock->fd >= 0) {
    int flags;
    /**设置非阻塞标志 */
    SYSCHECK(flags = fcntl(sock->fd, F_GETFL), "fcntl");
    SYSCHECK(fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }
  /**设置tcp no delay */
  SYSCHECK(setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt TCP NODELAY");
  // setsockopt should not fail even if the sizes are too large, do not change the default if unset by the user (=-1)
  int rcvBuf = ncclParamSocketMaxRecvBuff(), sndBuf = ncclParamSocketMaxSendBuff();
  /**设置发送/接收缓冲区大小 */
  if (sndBuf > 0) SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_SNDBUF, (char*)&sndBuf, sizeof(int)), "setsockopt SO_SNDBUF");
  if (rcvBuf > 0) SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBuf, sizeof(int)), "setsockopt SO_RCVBUF");
  return ncclSuccess;
}

static void socketResetAccept(struct ncclSocket* sock) {
  char line[SOCKET_NAME_MAXLEN+1];
  INFO(NCCL_NET|NCCL_INIT, "socketFinalizeAccept: didn't receive a valid magic from %s",
       ncclSocketToString(&sock->addr, line));
  // Ignore spurious connection and accept again
  (void)close(sock->fd);
  sock->fd = -1;
  sock->state = ncclSocketStateAccepting;
  sock->finalizeCounter = 0;
}

/**接入了client与server的magic,type字节，验证是否一致 */
static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  uint64_t magic;
  enum ncclSocketType type;
  int received;
  char line[SOCKET_NAME_MAXLEN+1];
  // once accepted, linux sockets do NOT inherit file status flags such as O_NONBLOCK (BSD ones do)
  NCCLCHECK(socketSetFlags(sock));

  if (sock->asyncFlag == 0 /*同步操作*/|| sock->finalizeCounter < sizeof(magic)/**异步操作未读取magic字节 */) {
    if (sock->asyncFlag == 0) {
      received = 0;
      /**读取magic字节 */
      if (socketWait(NCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received) != ncclSuccess) {
        socketResetAccept(sock);
        return ncclSuccess;
      }
    } else {
      /**异步读取magic字节 */
      int closed = 0;
      received = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(magic), &received, &closed));
      sock->finalizeCounter = received;
      if (received < sizeof(magic)) {
        if (closed) {
          socketResetAccept(sock);
        }
        return ncclSuccess;
      }
      memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
    }
    /**读取的magic与自身magic不一致的，断开连接 */
    if (magic != sock->magic) {
      socketResetAccept(sock);
      return ncclSuccess;
    }
  }
  /**读取type字节 */
  if (sock->asyncFlag == 0) {
    received = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
  } else {
    received = sock->finalizeCounter - sizeof(magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(type), &received));
    sock->finalizeCounter = received + sizeof(magic);
    if (received < sizeof(type)) return ncclSuccess;
    memcpy(&type, sock->finalizeBuffer, sizeof(type));
  }
  /**读取的type与自身type不一致的，断开连接 */
  if (type != sock->type) {
    WARN("socketFinalizeAccept from %s: wrong type %d != %d", ncclSocketToString(&sock->addr, line), type, sock->type);
    sock->state = ncclSocketStateError;
    close(sock->fd);
    sock->fd = -1;
    return ncclInternalError;
  } else {
    sock->state = ncclSocketStateReady;
  }
  return ncclSuccess;
}

/*重置socket对应的fd*/
static ncclResult_t socketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  int fd = -1;
  /*创建socket（tcp socket)*/
  SYSCHECKGOTO(fd = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  // if sock->fd is valid, close it and reuse its number
  if (sock->fd != -1) {
	  /*创建socket失败*/
    SYSCHECKGOTO(dup2(fd, sock->fd), "dup2", ret, cleanup);
    SYSCHECKGOTO(close(fd), "close", ret, cleanup);
  } else {
    sock->fd = fd;/*设置socket对应fd*/
  }
  NCCLCHECKGOTO(socketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup fd, leave sock->fd untouched
  if (fd != -1) {
    (void)close(fd);
  }
  goto exit;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode/**错误码 */, const char funcName[]/**调用方函数名称 */) {
  char line[SOCKET_NAME_MAXLEN+1];
  if (errCode == 0) {
	  /*指明为连接成功*/
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
	  /*处理中，置为connect polling状态*/
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == EINTR || errCode == EWOULDBLOCK || errCode == EAGAIN || errCode == ETIMEDOUT ||
             errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    if (sock->customRetry == 0) {
    	/*customRetry为0，nccl负责重试及避让*/
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
    	  /*重试次数超限*/
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts",
             funcName, ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      /**按重试次数增加避让时间间隔 */
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET|NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec",
           funcName, ncclSocketToString(&sock->addr, line), strerror(errCode),
           sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      msleep(sleepTime);/*避让重试*/
    }
    /*重置socket fd，置为连接中状态*/
    NCCLCHECK(socketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;/*指明为connting状态（一会会执行再连）*/
  } else {
	  /*连接失败*/
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

/**启动连接到远端 */
static ncclResult_t socketStartConnect(struct ncclSocket* sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->fd, &sock->addr.sa, sock->salen);/*连接到远端*/
  /*执行connect成功与否检查*/
  return socketConnectCheck(sock, (ret == -1) ? errno : 0, __func__);
}

static ncclResult_t socketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);
  char line[SOCKET_NAME_MAXLEN+1];

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->fd;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    WARN("socketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line), strerror(errno));
    return ncclSystemError;
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketPollConnect(sock));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  int sent;
  if (sock->asyncFlag == 0/**同步操作 */) {
    sent = 0;
    /**向对端发送此socket magic */
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
    sent = 0;
    /**向对端发送此socket type */
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
  } else {
    if (sock->finalizeCounter < sizeof(sock->magic)) {
      /*未完成Magic发送（finalizeCounter中记录的是已完成发送的字节数）*/
      sent = sock->finalizeCounter;
      /*发送magic*/
      NCCLCHECK(socketProgress(NCCL_SOCKET_SEND/*指明send*/, sock, &sock->magic, sizeof(sock->magic), &sent));
      sock->finalizeCounter = sent;/**更新已完成发送的字节数 */
      if (sent < sizeof(sock->magic)) return ncclSuccess;/**当前是异步实现，未完成magic发送，没法继续发送，先返回，等下一次再发送 */
    }
    /*发送type*/
    sent = sock->finalizeCounter - sizeof(sock->magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
    sock->finalizeCounter = sent + sizeof(sock->magic);
    if (sent < sizeof(sock->type)) return ncclSuccess;/**等待下次再发送 */
  }
  sock->state = ncclSocketStateReady;/*变更为state ready状态*/
  return ncclSuccess;
}

static ncclResult_t socketProgressState(struct ncclSocket* sock) {
	/*按状态处理*/
  if (sock->state == ncclSocketStateAccepting) {
	  /*对于等待accept的socket，尝试accept*/
    NCCLCHECK(socketTryAccept(sock));
  }
  if (sock->state == ncclSocketStateAccepted) {
    /**对于已accept到clent的调用（读取magic,type） */
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == ncclSocketStateConnecting) {
	  /*对于正在连接状态，启动执行connect*/
    NCCLCHECK(socketStartConnect(sock));
  }
  if (sock->state == ncclSocketStateConnectPolling) {
    /**未拿到连接结果，轮询时调用 */
    NCCLCHECK(socketPollConnect(sock));
  }
  if (sock->state == ncclSocketStateConnected) {
	  /*connect成功后调用（告知对端magic,type,以便达到state ready状态 */
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running/*出参，是否已达到ready状态 */) {
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
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif

  if (sock == NULL) {
    /**不能为空 */
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    /**不能未初始化的socket */
    WARN("ncclSocketConnect: file descriptor is -1");
    return ncclInvalidArgument;
  }

  if (sock->state != ncclSocketStateInitialized) {
    /**必须已完成初始化 */
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  sock->state = ncclSocketStateConnecting;/*指定为正在连接状态*/
  sock->finalizeCounter = 0;
  do {
    NCCLCHECK(socketProgressState(sock));/*执行连接，如出错直接返回*/
  } while (sock->asyncFlag == 0/**同步操作 */ &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0)/**没有abort */ &&
      (sock->state == ncclSocketStateConnecting ||
       sock->state == ncclSocketStateConnectPolling ||
       sock->state == ncclSocketStateConnected))/**未达到ready ready状态时，继续轮询 */;

  /**已设置abortFlag，返回错误 */
  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

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
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock) {
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
    if (listenSock->state == ncclSocketStateError)
      ret = ncclSystemError;
    else
      ret = ncclInternalError;
    goto exit;
  }

  if (sock->acceptFd == -1) {
	/**此sock首次应用于accept调用时，置为accepting状态*/
    memcpy(sock, listenSock, sizeof(struct ncclSocket));/**这里的复制大有深意，复用listenSock的magic,type */
    sock->acceptFd = listenSock->fd;
    sock->state = ncclSocketStateAccepting;
    sock->finalizeCounter = 0;
  }

  /*执行accept*/
  do {
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);
  } while (sock->asyncFlag == 0/**同步操作 */ &&
      (sock->abortFlag == NULL || __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE) == 0)/**没有abort */ &&
      (sock->state == ncclSocketStateAccepting ||
       sock->state == ncclSocketStateAccepted))/**未接入client，继续轮询 */;

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;

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
ncclResult_t ncclSocketInit(struct ncclSocket* sock/*出参，待初始化的socket*/, const union ncclSocketAddress* addr/*要设置的地址*/, uint64_t magic, enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  sock->errorRetries = 0;/**重试次数置为0*/
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = ncclSocketStateInitialized;/*状态指定为初始*/
  sock->magic = magic;
  sock->type = type;
  sock->fd = -1;
  sock->acceptFd = -1;/*默认acceptFd置为-1*/
  sock->customRetry = customRetry;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));/*设置地址*/
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
    	/*只考虑v4,v6两种*/
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
          ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);/**设置地址长度 */
    // in case of error, we close the fd before returning as it's unclear if the caller has to use ncclSocketClose for cleanup
    NCCLCHECKGOTO(socketResetFd(sock), ret, fail);/*初始化socket*/
  } else {
	  /*没有指定地址，地址置为0，且不初始化socket*/
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  if (sock->fd != -1) {
    close(sock->fd);
    sock->fd = -1;
  }
  goto exit;
}

ncclResult_t ncclSocketProgress(int op/*执行收或者发操作*/, struct ncclSocket* sock, void* ptr, int size, int* offset/*出参，偏移量*/, int* closed) {
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
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock/**发送socket */, void* sendPtr/**发送数据片指针 */, int sendSize/**发送数据片大小 */, struct ncclSocket* recvSock/**接收socket */, void* recvPtr/**接收数据片指针 */, int recvSize/**接收数据片大小 */) {
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
  int completedOps=0, i=0;
  while(completedOps < numOps){
    /**持续直到所有操作完成 */
    if (ops[i].offset < ops[i].size){
      NCCLCHECK(socketProgress(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset));
      if(ops[i].offset >= ops[i].size) completedOps++;/*记录完成操作的数目*/
    }
    i=(i+1)%numOps;
  }
  return ncclSuccess;
}
// Receive or detect connection closed
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  if (blocking) {
    while (offset < size) {
      NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
      if (*closed) return ncclSuccess;
    }
  } else {
    NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
    if (*closed) return ncclSuccess;

    // If any bytes were received, block waiting for the rest
    if (offset > 0) {
      while (offset < size) {
        NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
        if (*closed) return ncclSuccess;
      }
    // No bytes were received, return ncclInProgress
    } else {
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}

// Make it possible to close just one part of a socket.
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (sock->fd >= 0) {
      SYSCHECK(shutdown(sock->fd, how), "shutdown");
    }
    sock->state = ncclSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock != NULL) {
    if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && sock->fd >= 0) {
      if (wait) {
        char data;
        int closed = 0;
        do {
          int offset = 0;
          if (ncclSocketProgress(NCCL_SOCKET_RECV, sock, &data, sizeof(char), &offset, &closed) != ncclSuccess) break;
        } while (closed == 0);
      }
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is not affected
       * by refcount of fd, but close() is. close() won't close a fd and send FIN packet if
       * the fd is duplicated (e.g. fork()). So shutdown() guarantees the correct and graceful
       * connection close here. */
      (void)shutdown(sock->fd, SHUT_RDWR);
      (void)close(sock->fd);
    }
    sock->state = ncclSocketStateClosed;
    sock->fd = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = sock->fd;
  return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->fd = fd;
  return ncclSuccess;
}
