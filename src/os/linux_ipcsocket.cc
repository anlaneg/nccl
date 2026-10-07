/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ipcsocket.h"
#include "utils.h"
#include "os.h"
#include "param.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <inttypes.h>

NCCL_PARAM(IpcUseAbstractSocket, "IPC_USE_ABSTRACT_SOCKET", 1);

#define NCCL_IPC_SOCKNAME_STR "/tmp/nccl-socket-%d-%" PRIx64

/*
 * Create a Unix Domain Socket
 */
ncclResult_t ncclIpcSocketInit(ncclIpcSocket* handle, int rank/*当前rank*/, uint64_t hash/*当前uds编号*/, volatile uint32_t* abortFlag) {
  int fd = NCCL_INVALID_SOCKET;
  struct sockaddr_un cliaddr;
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return ncclInternalError;
  }

  handle->fd = NCCL_INVALID_SOCKET;
  handle->socketName[0] = '\0';
  /*创建unix socket*/
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    WARN("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    return ncclSystemError;
  }

  memset(&cliaddr, 0, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  /*创建地址（合入rank与hash)，hash保证了进程唯一，rank增加后，保证了唯一性*/
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (int)(sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot bind provided name to socket. Name too large");
    close(fd);
    return ncclInternalError;
  }

  int useAbstractSocket = ncclParamIpcUseAbstractSocket();
  if (!useAbstractSocket) {
    // For regular Unix domain sockets, unlink any existing socket file
    (void)unlink(temp);/*针对这种，直接删除防之前有残存*/
  }

  TRACE(NCCL_INIT | NCCL_P2P, "UDS: Creating socket %s%s", temp, useAbstractSocket ? " (abstract)" : "");

  strcpy(cliaddr.sun_path, temp);/*填地址*/
  if (useAbstractSocket) {
	  /*不产生具体文件*/
    cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
  }
  /*绑定此地址*/
  if (bind(fd, (struct sockaddr*)&cliaddr, sizeof(cliaddr)) < 0) {
    WARN("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno), errno);
    close(fd);
    return ncclSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  if (handle->abortFlag) {
    int flags;
    SYSCHECK(flags = fcntl(fd, F_GETFL), "fcntl");
    /*指定非阻塞*/
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return ncclSuccess;
}

/*取ipcsocket对应的fd*/
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, ncclIpcFd* fd) {
  if (handle == NULL) {
    WARN("ncclIpcSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = handle->fd;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket* handle) {
  if (handle == NULL) {
    return ncclInternalError;
  }
  if (handle->fd <= 0) {
    return ncclSuccess;
  }
  int useAbstractSocket = ncclParamIpcUseAbstractSocket();
  if (!useAbstractSocket) {
    (void)unlink(handle->socketName);
  }
  close(handle->fd);

  return ncclSuccess;
}

/*读取handle获得消息及fd*/
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket* handle, void* hdr/*出参，读到的消息*/, int hdrLen, ncclIpcFd* recvFd/*出参，读到的fd*/) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr* cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = (void*)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;/*收取的长度*/
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  /*读取消息*/
  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && COMPILER_ATOMIC_LOAD(handle->abortFlag, std::memory_order_acquire))
      return ncclInternalError;
  }

  if (recvFd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        WARN("UDS: Receiving data over socket failed");
        return ncclSystemError;
      }

      /*取捎带的fd*/
      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      WARN("UDS: Receiving data over socket %s failed", handle->socketName);
      return ncclSystemError;
    }
    TRACE(NCCL_INIT | NCCL_P2P, "UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket* handle, ncclIpcFd* recvFd) {
  return ncclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

/*向rank/hash构成的目的地址发送消息*/
ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, ncclIpcFd sendFd, int rank,
                                  uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[NCCL_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr* cmptr;
  char dummy_buffer[1] = {'\0'};
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  memset(&cliaddr, 0, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank/*送哪个rank*/, hash/*对应的hash是什么*/);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return ncclInternalError;
  }
  strcpy(cliaddr.sun_path, temp);/*设置目的*/

  int useAbstractSocket = ncclParamIpcUseAbstractSocket();
  if (useAbstractSocket) {
    cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
  }

  TRACE(NCCL_INIT, "UDS: Sending hdr %p len %d fd %d to UDS socket %s", hdr, hdrLen, sendFd, temp);

  if (sendFd != -1) {
	  /*要捎带fd*/
    memset(&control_un, '\0', sizeof(control_un));
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = (void*)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
	  /*未指定内容时，随便加此dummy buffer*/
    iov[0].iov_base = (void*)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  /*执行消息发送*/
  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Sending data over socket %s failed : %s (%d)", temp, strerror(errno), errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && COMPILER_ATOMIC_LOAD(handle->abortFlag, std::memory_order_acquire))
      return ncclInternalError;
  }

  return ncclSuccess;
}

/*向rank发送fd(sendFd)*/
ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket* handle, ncclIpcFd sendFd, int rank, uint64_t hash) {
  return ncclIpcSocketSendMsg(handle, NULL/*发送的内容为空*/, 0/*发送的内容长度为0*/, sendFd/*要捎带的显存fd*/, rank, hash);
}

int ncclIpcFdClose(ncclIpcFd fd) {
  return close(fd);
}
