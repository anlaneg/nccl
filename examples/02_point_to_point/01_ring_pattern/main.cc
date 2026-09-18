/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * NCCL Ring Pattern Example - Educational Version
 *
 * This example demonstrates the fundamental ring communication pattern using
 * NCCL's point-to-point operations. Understanding ring patterns is essential
 * for NCCL programming as they form the basis of many collective algorithms.
 *
 * Learning Objectives:
 * - Understand ring topology and neighbor communication
 * - Learn NCCL point-to-point send/recv operations
 * - See how data flows in a ring pattern
 * - Practice deadlock avoidance with ncclGroup operations
 * - Understand single-process multi-GPU patterns
 *
 */

// Enhanced error checking macro for NCCL operations
// Provides detailed error information including the failed operation
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  // ========================================================================
  // STEP 1: Initialize Environment and Detect GPUs
  // ========================================================================

  int num_gpus = 0;
  ncclComm_t *comms = NULL;
  cudaStream_t *streams = NULL;
  float **h_sendbuff = NULL;
  float **h_recvbuff = NULL;
  float **d_sendbuff = NULL;
  float **d_recvbuff = NULL;

  printf("Starting NCCL ring communication example\n");

  // Get number of available CUDA devices
  CUDACHECK(cudaGetDeviceCount(&num_gpus));/**获取可用GPU数量 */

  if (num_gpus == 0) {
	  /*没有可用的gpu*/
    fprintf(stderr, "No CUDA devices found\n");
    return 1;
  }

  if (num_gpus < 2) {
    /**必须至少要两个gpu才能不输出此日志 */
    printf("At least 2 GPU are necessary to create inter-GPU traffic\n");
    printf("Found only %d GPU(s) - pattern will be limited\n", num_gpus);
  }

  printf("Using %d GPUs for ring communication\n", num_gpus);/*做ring型通信*/

  // ========================================================================
  // STEP 2: Prepare Data Structures and Device List
  // ========================================================================

  printf("Preparing data structures\n");

  // Create device list (use all available devices)
  int *devices = (int *)malloc(num_gpus * sizeof(int));
  for (int i = 0; i < num_gpus; i++) {
	  /**创建gpu设备编号列表（0到num_gpus-1） */
    devices[i] = i;
  }

  // Allocate communicators, streams, and buffer pointers
  /*每个gpu对应一个ncclComm_t结构*/
  comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  /*每个gpu对应一个cudaStream_t结构*/
  streams = (cudaStream_t *)malloc(num_gpus * sizeof(cudaStream_t));
  /**一组指针数据，每个指针指向一个gpu的发送缓冲区（主机端） */
  h_sendbuff = (float **)malloc(num_gpus * sizeof(float *));/*主机侧发送buffer*/
  h_recvbuff = (float **)malloc(num_gpus * sizeof(float *));/*主机侧接收buffer*/
  d_sendbuff = (float **)malloc(num_gpus * sizeof(float *));/*设备侧发送buffer*/
  d_recvbuff = (float **)malloc(num_gpus * sizeof(float *));/*设备侧接收buffer*/

  // ========================================================================
  // STEP 3: Initialize NCCL Communicators
  // ========================================================================

  /*
   * ncclCommInitAll is the simplest way to initialize NCCL communicators
   * for single-process, multi-GPU scenarios. It automatically:
   * - Creates one communicator per GPU
   * - Assigns ranks sequentially (GPU 0 = rank 0, GPU 1 = rank 1, etc.)
   */
  printf("Initializing NCCL communicators\n");
  /*对ncclComm_t结构体进行初始化*/
  NCCLCHECK(ncclCommInitAll(comms, num_gpus, devices/**出参，gpu设备编号数组（无重复） */));
  printf("All communicators initialized successfully\n");

  // ========================================================================
  // STEP 4: Create Streams and Verify Communicator Setup
  // ========================================================================

  printf("Creating CUDA streams and verifying setup\n");

  // Create streams and verify communicator info
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamCreate(&streams[i]));

    // Query communicator information for verification
    int rank, size, device;
    NCCLCHECK(ncclCommUserRank(comms[i], &rank));
    NCCLCHECK(ncclCommCount(comms[i], &size));
    NCCLCHECK(ncclCommCuDevice(comms[i], &device));

    printf("  GPU %d -> NCCL rank %d/%d on CUDA device %d\n", i, rank, size,
           device);
  }

  // ========================================================================
  // STEP 5: Set Up Ring Topology and Allocate Buffers
  // ========================================================================

  printf("Setting up ring topology\n");
  printf("Data flow -> GPU 0 -> ... -> GPU %d -> GPU 0\n", num_gpus - 1);

  // Test with 1GB of data
  const size_t count = 256 * 1024 * 1024; // 256M floats = 1GB
  const size_t size_bytes = count * sizeof(float);

  printf("Ring transfer with %zu elements (%.2f GB per GPU)\n", count,
         size_bytes / (1024.0 * 1024.0 * 1024.0));

  // Allocate buffers for each GPU
  printf("Allocating and initializing buffers\n");
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));/**设置当前gpu设备为devices[i] */

    h_sendbuff[i] = (float *)malloc(size_bytes);/**主机gpu的发送缓冲区（主机端） */
    h_recvbuff[i] = (float *)malloc(size_bytes);/**主机gpu的接收缓冲区（主机端） */
    CUDACHECK(cudaMalloc((void **)&d_sendbuff[i], size_bytes));/**申请当前gpu的发送缓冲区（设备端） */
    CUDACHECK(cudaMalloc((void **)&d_recvbuff[i], size_bytes));/**申请当前gpu的接收缓冲区（设备端） */

    // Initialize data with GPU-specific pattern for verification
    for (size_t j = 0; j < count; j++) {
      h_sendbuff[i][j] = (float)(i * 1000 + j % 1000);/**初始化当前gpu对应的主机侧发送缓冲区，按顺序排列的数字 */
    }
    CUDACHECK(cudaMemcpy(d_sendbuff[i], h_sendbuff[i], size_bytes,
                         cudaMemcpyHostToDevice));/**将主机侧gpu对应的发送缓冲区的内容复制到设备侧gpu的发送缓冲区（设备端） */
  }

  // ========================================================================
  // STEP 6: Execute Ring Communication Pattern
  // ========================================================================

  /*
   * The ring communication uses ncclGroup operations to avoid deadlock.
   * Without grouping, if all GPUs tried to send first, they would deadlock
   * waiting for receivers. Grouping allows NCCL to execute operations
   * in the optimal order.
   */
  printf("Executing ring communication\n");

  // NOTE: ncclGroupStart and ncclGroupEnd are essential to avoid deadlock
  // when using ncclCommInitAll!
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;/**得到当前gpu(i)的下一个gpu编号（next) */
    int prev = (i - 1 + num_gpus) % num_gpus;/**得到当前gpu(i)的上一个gpu编写（prev） */
    printf("  GPU %d sends to GPU %d, receives from GPU %d\n", i, next, prev);

    // Each GPU simultaneously sends to next and receives from previous
    /**发送内容到next */
    NCCLCHECK(
        ncclSend(d_sendbuff[i]/**当前gpu对应的发送缓冲区（设备端） */, count/**发送数据数量 */, ncclFloat/**发送类型（float） */, next/**下一个gpu编号（next） */, comms[i], streams[i]));
        /**自prev收取内容 */
    NCCLCHECK(
        ncclRecv(d_recvbuff[i]/**当前gpu对应的接收缓冲区（设备端） */, count/**接收数据数量 */, ncclFloat/**接收类型（float） */, prev/**上一个gpu编号（prev） */, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());/**上面只是入队，这里才是真正的执行*/

  // Synchronize all streams to ensure communication completes
  /**同步等待所有gpu的stream执行完成 */
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  printf("Ring communication completed successfully\n");

  // ========================================================================
  // STEP 7: Verify Data Correctness and Report Results
  // ========================================================================

  printf("Verifying data correctness\n");
  bool all_correct = true;

  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaMemcpy(h_recvbuff[i], d_recvbuff[i], size_bytes,
                         cudaMemcpyDeviceToHost));/**将设备侧gpu对应的接收缓冲区（设备端）的内容，复制到主机侧gpu对应的接收缓冲区（主机端） */

    int prev = (i - 1 + num_gpus) % num_gpus;
    // Verify that GPU i received data from GPU prev
    float expected = (float)(prev * 1000);
    bool correct = (h_recvbuff[i][0] == expected);/**由于这里数据实际上只就近传递了一次（即prev仅将自已的数据传给next)因此每个gpu拿的都是其prev之前的数据。 */

    printf("  GPU %d received data from GPU %d: %s\n", i, prev,
           correct ? "CORRECT" : "ERROR");

    if (!correct) {
      all_correct = false;
      printf("  Expected %.0f, got %.0f\n", expected, h_recvbuff[i][0]);
    }
  }

  if (all_correct) {
    printf("SUCCESS - All GPUs received correct data\n");
  } else {
    printf("FAILURE - Data verification failed\n");
  }

  // ========================================================================
  // STEP 8: Cleanup Resources
  // ========================================================================

  printf("Cleaning up resources\n");

  // Free buffers
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(cudaSetDevice(devices[i]));
    free(h_sendbuff[i]);
    free(h_recvbuff[i]);
    CUDACHECK(cudaFree(d_sendbuff[i]));
    CUDACHECK(cudaFree(d_recvbuff[i]));
  }

  // Destroy communicators and streams
  for (int i = 0; i < num_gpus; i++) {
    NCCLCHECK(ncclCommDestroy(comms[i]));
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }

  // Free allocated memory
  free(devices);
  free(comms);
  free(streams);
  free(h_sendbuff);
  free(h_recvbuff);
  free(d_sendbuff);
  free(d_recvbuff);

  printf("Example completed successfully!\n");

  return 0;
}
