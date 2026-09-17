/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "shmutils.h"
#include "comm.h"
#include "checks.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <utils.h>

struct shmHandleInternal {
  int fd;
  char* shmPath;
  char* shmPtr;
  void* devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int* refcount;
};

static void shmHandleInit(int fd, char* shmPath, size_t shmSize, size_t realShmSize, char* hptr, void* dptr, bool create, struct shmHandleInternal* handle/*出参，初始化此结构*/) {
  handle->fd = fd;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char*)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr) memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
  return;
}

ncclResult_t ncclShmOpen(char* shmPath/*共享内存文件路径*/, size_t shmPathSize/*路径长度*/, size_t shmSize/*占用大小*/, void** shmPtr/*出参，共享内存起始位置*/, void** devShmPtr, int refcount, ncclShmHandle_t* handle) {
  int fd = -1;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int); /* extra sizeof(int) bytes for reference count */
  const size_t realShmSize = shmSize + refSize;/*增加引用计数大小*/

  *handle = *shmPtr = NULL; /* assume shmPtr and handle always set correctly by users. */
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal*)calloc(1, sizeof(struct shmHandleInternal)), NULL, ret, fail);
  if (create/*创建*/) {
    /* refcount > 0 means the caller tries to allocate a shared memory. This shared memory segment will have
     * refcount references; when the peer attaches, it should pass -1 to reduce one reference count. When it
     * goes down to 0, unlink should be called in order to delete shared memory file. */
    if (shmPath[0] == '\0') {
      snprintf(shmPath, shmPathSize, "/dev/shm/nccl-XXXXXX");
    retry_mkstemp:
      fd = mkstemp(shmPath);/*创建临时文件*/
      if (fd < 0) {
    	  /*重新尝试或者报错*/
        if (errno == EINTR) {
          INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
          goto retry_mkstemp;
        }
        WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail;
      }
    } else {
    	/*创建共享内存*/
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
    }

  retry_fallocate:
    /*设置占用大小*/
    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      if (errno == EINTR) {
        INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize, strerror(errno), errno);
        goto retry_fallocate;
      }
      WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  } else {
	  /*仅打开*/
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
  }

  /*映射此文件*/
  hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED) {
    WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
	  /*初始化引用计数*/
    *(int*)(hptr + shmSize) = refcount;
  } else {
	  /*增加引用计数*/
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
    	/*移除此文件*/
      /* the last peer has completed attachment, it should unlink the shm mem file. */
      if (unlink(shmPath) != 0) {
        INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
      }
    }
  }

  if (devShmPtr) {
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
    CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
  }

  /*初始化tmphandle*/
  shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = (ncclShmHandle_t)tmphandle;
  return ret;
fail:
	/*创建share memory失败*/
  WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
       shmPath, shmSize, strerror(errno), errno);
  if (tmphandle) {
    shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclShmClose((ncclShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->fd >= 0) {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
        if (unlink(tmphandle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr) {
      if (tmphandle->devShmPtr) CUDACHECK(cudaHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
    free(tmphandle);
  }
  return ret;
}

ncclResult_t ncclShmUnlink(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
      if (unlink(tmphandle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
        ret = ncclSystemError;
      }
      free(tmphandle->shmPath);
      tmphandle->shmPath = NULL;
    }
  }
  return ret;
}

ncclResult_t ncclShmemAllgather(struct ncclComm *comm, struct ncclShmemCollBuff *shmem, void *sendbuff, void *recvbuff, size_t typeSize) {
  ncclResult_t ret = ncclSuccess;
  int nextRound = shmem->round + 1;
  int curIndex = shmem->round % 2;
  bool done;
  int index = 0;
  size_t maxTypeSize = shmem->maxTypeSize;

  if (comm == NULL || shmem == NULL || sendbuff == NULL || recvbuff == NULL || maxTypeSize < typeSize) {
    ret = ncclInvalidArgument;
    goto exit;
  }

  memcpy((char*)shmem->ptr[curIndex] + comm->localRank * maxTypeSize, sendbuff, typeSize);
  /* reset the previous round and notify I arrive this round */
  __atomic_store_n((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * comm->localRank), nextRound, __ATOMIC_RELEASE);

  do {
    done = true;
    for (int i = index; i < comm->localRanks; ++i) {
      if (i != comm->localRank && __atomic_load_n((int*)((char*)shmem->cnt[curIndex] + CACHE_LINE_SIZE * i), __ATOMIC_ACQUIRE) < nextRound) {
        done = false;
        index = i;
        break;
      }
    }
  } while (!done);

  for (int i = 0; i < comm->localRanks; ++i) {
    memcpy((uint8_t*)recvbuff + i * typeSize, (uint8_t*)shmem->ptr[curIndex] + i * maxTypeSize, typeSize);
  }
  shmem->round = nextRound;

exit:
  return ret;
}
