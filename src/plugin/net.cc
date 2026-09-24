/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_net.h"

#include <string.h>
#include <errno.h>
#include <mutex>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>

#if defined(NCCL_OS_LINUX)
typedef ncclNet_t* getNcclNet_t(void* netPluginLib);
typedef ncclCollNet_t* getNcclCollNet_t(void* netPluginLib);

extern getNcclNet_t getNcclNet_v6;
extern getNcclNet_t getNcclNet_v7;
extern getNcclNet_t getNcclNet_v8;
extern getNcclNet_t getNcclNet_v9;
extern getNcclNet_t getNcclNet_v10;
extern getNcclNet_t getNcclNet_v11;
extern getNcclNet_t getNcclNet_v12;
extern getNcclCollNet_t getNcclCollNet_v6;
extern getNcclCollNet_t getNcclCollNet_v7;
extern getNcclCollNet_t getNcclCollNet_v8;
extern getNcclCollNet_t getNcclCollNet_v9;
extern getNcclCollNet_t getNcclCollNet_v10;
extern getNcclCollNet_t getNcclCollNet_v11;
extern getNcclCollNet_t getNcclCollNet_v12;
NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 0);
#define NCCL_NET_VERSION_COUNT 7
/*有哪些版本（从最高版本开始排列，如果有一个可获得，则退出）*/
int ncclNetVersion[NCCL_NET_VERSION_COUNT] = {12, 11, 10, 9, 8, 7, 6};
/*列出不同版本的ncclnet获取函数*/
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v12, getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
                                                    getNcclNet_v8,  getNcclNet_v7,  getNcclNet_v6};
getNcclCollNet_t* getNcclCollNet[NCCL_NET_VERSION_COUNT] = {getNcclCollNet_v12, getNcclCollNet_v11, getNcclCollNet_v10,
                                                            getNcclCollNet_v9,  getNcclCollNet_v8,  getNcclCollNet_v7,
                                                            getNcclCollNet_v6};
#endif

// Count only built-in transports registered below. Windows does not build the
// real IB transport (only a failure stub is linked), so it registers
// NetworkDirect and Socket instead.
/*内置插件数目*/
#define NCCL_NET_NUM_INTERNAL_PLUGINS 2

typedef enum ncclNetPluginState {
  ncclNetPluginStateDisabled = -2,       // Plugin library failed to initialize
  ncclNetPluginStateLoadFailed = -1,       // Plugin library failed to load
  ncclNetPluginStateLoadReady = 0,        // Plugin library is ready to be loaded
  ncclNetPluginStateInitReady = 1,        // Plugin library is loaded and ready to be initialized
  ncclNetPluginStateEnabled = 2,        // Plugin library is loaded and initialized
} ncclNetPluginState_t;

#define MAX_STR_LEN 255
typedef struct netPluginLib {
  /**插件库的名称 */
  char name[MAX_STR_LEN];                       // Name of the plugin library
  /*插件库对应的so handle*/
  void* dlHandle;                               // Handle to the plugin library
  /*此版本对应的ncclNet结构体指针，用于指出api函数指针*/
  ncclNet_t* ncclNet;                           // Pointer to the ncclNet_t structure
  int ncclNetVer;                               // Version of the nccl net plugin
  /*此版本对应的collnet结构体指针，用于指明api函数*/
  ncclCollNet_t* ncclCollNet;                   // Pointer to the ncclCollNet_t structure
  /*netplugin的加载状态(load,init,enable按顺序三种状态）*/
  ncclNetPluginState_t ncclNetPluginState;      // State of the nccl net plugin
  /*collnet plugin的加载状态*/
  ncclNetPluginState_t ncclCollNetPluginState;  // State of the nccl coll net plugin
  int ncclNetPluginRefCount;                    // Reference count for the nccl net plugin
  /*指明网络设备数目*/
  int netPhysDevs;                              // ncclNet - number of physical devices
  /*指明虚拟网络设备数目*/
  int netVirtDevs;                              // ncclNet - number of virtual devices
  /*指明collnet物理设备数目*/
  int collNetPhysDevs;                          // ncclCollNet -  number of physical devices
  int collNetVirtDevs;                          // ncclCollNet -  number of virtual devices
} netPluginLib_t;

static int pluginCount = 0;
static netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS] = {0};
static std::mutex netPluginMutex;
static std::once_flag initPluginLibsOnceFlag;

#if defined(NCCL_OS_LINUX)
/**卸载插件 */
static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
  if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {
    INFO(NCCL_DESTROY | NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));

    // Reset fields but preserve name, to be reused when reloading
    pluginLib->dlHandle = NULL;
    pluginLib->ncclNet = NULL;
    pluginLib->ncclNetVer = 0;
    pluginLib->ncclCollNet = NULL;
    pluginLib->ncclNetPluginState = ncclNetPluginStateLoadReady;
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadReady;
    pluginLib->ncclNetPluginRefCount = 0;
    pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
    pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  }
  return ncclSuccess;
}

/**加载插件（net插件，collnet插件，gin插件） */
static ncclResult_t ncclNetPluginLoad(netPluginLib_t* pluginLib) {
  pluginLib->dlHandle = ncclOpenNetPluginLib(pluginLib->name);/**打开插件库 */

  if (pluginLib->dlHandle == nullptr) goto fail;/*打开插件失败*/
  // load ncclNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
	/*设置尝试的net版本号*/
    pluginLib->ncclNetVer = ncclNetVersion[i];
    /*i号版本的处理函数传入lib,用于获取此版本对应的函数api结构体*/
    pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclNet) break;/*已取得退出，尝试结束*/
  }

  // if we fail to find a net, exit
  if (pluginLib->ncclNet == nullptr) {
    /*加载失败，报错*/
    const char* netPlugin = ncclGetEnv("NCCL_NET_PLUGIN");
    if (netPlugin) {
      ATTN("External network plugin %s is unsupported",
           (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
    } else {
      INFO(NCCL_INIT | NCCL_NET, "External network plugin %s is unsupported",
           (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
    }
    goto fail;
  }

  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;

  // load ncclCollNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
	  /*按版本优先级加载nccl collnet*/
    pluginLib->ncclCollNet = getNcclCollNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclCollNet) break;/*此版本加载成功，跳出*/
  }

  /*加载失败*/
  if (pluginLib->ncclCollNet == nullptr) pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  else pluginLib->ncclCollNetPluginState = ncclNetPluginStateInitReady;

  INFO(NCCL_INIT | NCCL_NET, "Successfully loaded external network plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->ncclNetPluginState = ncclNetPluginStateLoadFailed;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  goto exit;
}
#endif

ncclResult_t ncclNetCheckDeviceVersion(struct ncclComm* comm, ncclNet_t* net, int dev) {
  ncclNetProperties_t props;

  NCCLCHECK(net->getProperties(dev, &props));
  ncclNetDeviceType type = props.netDeviceType;
  switch (type) {
  case NCCL_NET_DEVICE_HOST:
    break;
  case NCCL_NET_DEVICE_UNPACK:
    if (props.netDeviceVersion == NCCL_NET_DEVICE_UNPACK_VERSION) {
      INFO(NCCL_INIT, "Using NCCL_NET_DEVICE_UNPACK net plugin version %d", props.netDeviceVersion);
      return ncclSuccess;
    } else {
      WARN("NCCL_DEVICE_UNPACK plugin has incompatible version %d, this NCCL build is compatible with %d, not using it",
           props.netDeviceVersion, NCCL_NET_DEVICE_UNPACK_VERSION);
      return ncclInternalError;
    }
  default:
    WARN("Unknown device code index %d", type);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclProfilerCallback_t ncclNetGetProfilerCallback() {
#if defined(NCCL_OS_LINUX)
  return ncclProfilerCallback;
#elif defined(NCCL_OS_WINDOWS)
  return nullptr;
#endif
}

/**初始化网络插件 */
static ncclResult_t ncclNetPluginInit(struct ncclComm* comm, netPluginLib_t* pluginLib) {
  int ndev;
  bool initCompleted = false;
  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    ncclNetCommConfig_t commConfig = {};
    commConfig.trafficClass =
      comm->config.trafficClass == NCCL_CONFIG_UNDEF_INT ? NCCL_NET_TRAFFIC_CLASS_UNDEF : comm->config.trafficClass;
    /*执行网络初始化*/
    if (pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ncclDebugLog,
                                 ncclNetGetProfilerCallback()) != ncclSuccess) {
      goto fail;
    }
    initCompleted = true;
  }
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    /*取网络设备数目*/
	  if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0) goto fail;
    pluginLib->netPhysDevs = ndev;
    pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  }
  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;
  INFO(NCCL_INIT | NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);

  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclCollNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
	  /*网络插件有collnet的执行初始化（对ib而言没有）*/
    if (pluginLib->ncclCollNet->init(&comm->collNetContext, comm->commHash, ncclDebugLog) != ncclSuccess) {
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    }
  }
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclCollNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    } else {
      pluginLib->collNetPhysDevs = ndev;
      pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateEnabled;
    }
  }

exit:
  return ncclSuccess;
fail:
  ATTN("Failed to initialize NET plugin %s", pluginLib->ncclNet->name);
  if (initCompleted) pluginLib->ncclNet->finalize(comm->netContext);
  pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->ncclNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  goto exit;
}

/*为comm设置使能的ncclNet,ncclCollNet,ncclGin*/
static ncclResult_t ncclNetPluginAssignToComm(struct ncclComm* comm, int pluginIndex/**要使用的插件索引 */, bool* isAssigned) {
  *isAssigned = false;

  if (ncclSuccess != ncclNetCheckDeviceVersion(comm, netPluginLibs[pluginIndex].ncclNet, 0)) return ncclSuccess;

  if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateEnabled) {
    /*此网络插件已被使能，设置ncclNet（比如ib情况下对应的即为ncclNetIb）*/
    comm->ncclNet = netPluginLibs[pluginIndex].ncclNet;
    comm->ncclNetVer = netPluginLibs[pluginIndex].ncclNetVer;
    comm->netPluginIndex = pluginIndex;
    netPluginLibs[pluginIndex].ncclNetPluginRefCount++;/*引用计数增大*/
    *isAssigned = true;/*assigned成功*/
    INFO(NCCL_INIT | NCCL_NET, "Assigned NET plugin %s to comm", netPluginLibs[pluginIndex].ncclNet->name);
    if (netPluginLibs[pluginIndex].ncclCollNetPluginState >= ncclNetPluginStateEnabled) {
      comm->ncclCollNet = netPluginLibs[pluginIndex].ncclCollNet;/*有collnet的设置*/
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclNetPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN * (NCCL_NET_MAX_PLUGINS - NCCL_NET_NUM_INTERNAL_PLUGINS)] = {0};
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex) {
      // Append all disabled plugin names to a string
      snprintf(names + strlen(names), sizeof(names) - strlen(names), (strlen(names) == 0) ? "%s" : ", %s",
               netPluginLibs[i].name);
      netPluginLibs[i].ncclNetPluginState = ncclNetPluginStateDisabled;/*非pluginIndex的插件均禁用*/
    }
  }
  if (strlen(names) > 0) {
    INFO(NCCL_INIT | NCCL_NET, "Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void ncclNetPluginRegisterInternal(int* pluginCounter, ncclNet_t* net, int version) {
  netPluginLib_t* pluginLib = &netPluginLibs[*pluginCounter];
  pluginLib->ncclNet = net;
  pluginLib->ncclNetVer = version;
  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->netPhysDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetPhysDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  (*pluginCounter)++;
}

/*通过环境变量初始化网络插件数组(如没有指定环境变量，则使用默认插件名称）
 * ，记录在netPluginLibs数组中，并增加内置的ib网络插件，socket网络插件*/
static void initPluginLibsOnceFunc() {
  int pluginCounter = 0;
  memset(netPluginLibs, 0, NCCL_NET_MAX_PLUGINS * sizeof(netPluginLib_t));
#if defined(NCCL_OS_LINUX)
  char* netPluginName = nullptr;
  const char* defaultNetPlugin = "libnccl-net.so";/**默认网络插件名称 */
  const char* envNetPlugin = nullptr;
  char* envNetPluginList = nullptr;
  char* savePtr = nullptr;
  envNetPlugin = ncclGetEnv("NCCL_NET_PLUGIN");
  if (envNetPlugin) {
	  /*通过环境变量设置的net插件*/
    INFO(NCCL_ENV | NCCL_NET, "NCCL_NET_PLUGIN set by environment to %s", envNetPlugin);
    if (strcasecmp(envNetPlugin, "none") == 0) envNetPlugin = "";/*如果为none则认为未设置*/
    envNetPluginList = strdup(envNetPlugin);
    // Iterate over list until the list is empty
    netPluginName = strtok_r(envNetPluginList, ",", &savePtr);/**环境变量指定的网络插件是一组逗号分隔的列表 */
    while (netPluginName) {
      // We have 2 internal plugins (ib and socket)
      // So, we can have at most( NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS)) in the NCCL_NET_PLUGIN list
      if (pluginCounter >= (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS))) {
    	  /*插件数目指定的过多*/
        INFO(NCCL_NET | NCCL_ENV, "NCCL_NET_PLUGIN list contains more than %d plugins, ignoring the rest",
             (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS + 1)));
        break;
      }
      // need to leave space for the name + "\n"
      if ((strlen(netPluginName) + 1) <= MAX_STR_LEN) {
        netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;/*先loadready*/
        netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();/*通过环境取引用计数*/
        strcpy(netPluginLibs[pluginCounter].name, netPluginName);/*指定插件名称*/
        pluginCounter++;/**占用这个counter */
      } else {
        /**插件名称过长，忽略 */
        INFO(NCCL_NET | NCCL_ENV,
             "NCCL_NET_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", netPluginName,
             MAX_STR_LEN);
      }
      netPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envNetPluginList) free(envNetPluginList);
  } else {
    // Add default net plugin
    /**无环境变量插件，使用默认网络插件 */
    netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
    netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
    strcpy(netPluginLibs[pluginCounter++].name, defaultNetPlugin);/*指定默认名称*/
  }

  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetIb, 0);/**增加内置ib网络插件（这种没有指定name) */
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetSocket, 0);/**增加内置socket插件 */
#elif defined(NCCL_OS_WINDOWS)
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetNd, 12);
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetSocket, 12);
#endif
  pluginCount = pluginCounter;/*全局变量，指明网络插件lib总数*/
}

static ncclResult_t ncclNetPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  NCCLCHECK(netPluginLibs[pluginIndex].ncclNet->finalize(comm->netContext));
  if (netPluginLibs[pluginIndex].ncclCollNet &&
      netPluginLibs[pluginIndex].ncclCollNetPluginState == ncclNetPluginStateEnabled) {
    NCCLCHECK(netPluginLibs[pluginIndex].ncclCollNet->finalize(comm->collNetContext));
  }
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;
#if defined(NCCL_OS_LINUX)
  if (pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[pluginIndex]));
  }
#endif
  return ncclSuccess;
}

/*为comm设置网络插件信息*/
ncclResult_t ncclNetInit(struct ncclComm* comm) {
  bool ncclNetPluginInitialized = false;
  comm->ncclCollNet = nullptr;
  /*初始化网络插件数组（内置的插件排在最后位置，环境变量指定的插件排在最前面）*/
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  std::lock_guard<std::mutex> lock(netPluginMutex);
  /**遍历所有网络插件 */
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
#if defined(NCCL_OS_LINUX)
    if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) &&
        (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {
    	/*加载环境变量指定的load有效插件*/
      NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));
    }
#endif
    if ((netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateInitReady) &&
        (!comm->config.netName || (strcasecmp(comm->config.netName, netPluginLibs[pluginIndex].ncclNet->name) == 0))) {
    	/*此类型插件initready，且配置指定了此名称或者配置没有指定名称*/
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclNetPluginInit(comm, &netPluginLibs[pluginIndex]));/** 初始化此网络插件 */
      if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {
        bool isAssigned = false;
        /*为comm绑定网络插件*/
        NCCLCHECK(ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclNetPluginDisableOtherExternal(pluginIndex);/*禁用其它外部插件*/
          ncclNetPluginInitialized = true;/*网络插件初始化完成*/
          break;
        } else {
	  /*没有assigned成功，释放*/
          ncclNetPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (ncclNetPluginInitialized) return ncclSuccess;
  WARN("Failed to initialize any NET plugin");
  return ncclInvalidUsage;
}

ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t ret = ncclSuccess;
  comm->netContext = parent->netContext;
  comm->collNetContext = parent->collNetContext;
  comm->ncclNet = parent->ncclNet;
  comm->ncclCollNet = parent->ncclCollNet;
  comm->netPluginIndex = parent->netPluginIndex;
  comm->ncclNetVer = parent->ncclNetVer;
  if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR && strcasecmp(comm->config.netName, parent->config.netName)) {
    WARN("Comm config netName (%s) does not match the parent (%s)", comm->config.netName, parent->config.netName);
    ret = ncclInvalidUsage;
  }
  if (comm->config.trafficClass != NCCL_CONFIG_UNDEF_INT && comm->config.trafficClass != parent->config.trafficClass) {
    INFO(NCCL_INIT, "Comm config trafficClass (%d) does not match the parent (%d)", comm->config.trafficClass,
         parent->config.trafficClass);
  }
  return ret;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->netPluginIndex;
  std::lock_guard<std::mutex> lock(netPluginMutex);
  NCCLCHECK(ncclNetPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled ||
      netPluginLibs[netPluginIndex].netPhysDevs == NCCL_UNDEF_DEV_COUNT)
    goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].netPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].netVirtDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized netPlugin[%d]", netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled ||
      netPluginLibs[netPluginIndex].collNetPhysDevs == NCCL_UNDEF_DEV_COUNT)
    goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].collNetPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].collNetVirtDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized netPlugin[%d]", netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].netVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("failed to set the number of devices for netPlugin[%d] to %d", netPluginIndex, nVirtDevs);
  return ncclInternalError;
}

ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].collNetVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("failed to set the number of devices for netPlugin[%d] to %d", netPluginIndex, nVirtDevs);
  return ncclInternalError;
}

/*检查是否支持gdr*/
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport/*是否支持gdr*/) {
  constexpr int GPU_BUF_SIZE = 2 * 1024 * 1024;
#if CUDART_VERSION >= 11030
  // In CUDA 11.3 and later we can now query the cudaDevAttrGPUDirectRDMASupported attribute
  int driverVersion;
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (driverVersion >= 11030) {
    int cudaDev, attr = 0;
    CUDACHECK(cudaGetDevice(&cudaDev));
    CUDACHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMASupported, cudaDev));/*是否支持gdr*/
    *gdrSupport = attr;
    return ncclSuccess;
  }
#endif
  static int gdrSupportMatrix[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  if (gdrSupportMatrix[comm->cudaDev] == -1) {
    int netDevs;
    NCCLCHECK(comm->ncclNet->devices(&netDevs));
    gdrSupportMatrix[comm->cudaDev] = 0;
    /*遍历检查每一个设备*/
    for (int dev = 0; dev < netDevs; dev++) {
      // Find a net device which is GDR-capable
      ncclNetProperties_t props;
      NCCLCHECK(comm->ncclNet->getProperties(dev, &props));/*取设备属性*/
      if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;/*不支持cuda也不行*/

      // Allocate memory on the GPU and try to register it on the NIC.
      void *lComm = NULL, *sComm = NULL, *rComm = NULL;
      ncclNetHandle_t handle;
      char* gpuPtr = NULL;
      void* mHandle = NULL;
      ncclResult_t ret;
      /*执行监听*/
      NCCLCHECKGOTONOWARN(comm->ncclNet->listen(comm->netContext, dev, &handle, &lComm), ret, cleanup1, NCCL_NET);

      bool connected;
      connected = false;
      while (!connected) {
        // If we're aborting now, skip to cleanup
        if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
          goto cleanup2;
        }

        if (sComm == NULL) {
          NCCLCHECKGOTONOWARN(comm->ncclNet->connect(comm->netContext, dev, &handle, &sComm, NULL), ret, cleanup2,
                              NCCL_NET);
        }

        if (rComm == NULL) NCCLCHECKGOTONOWARN(comm->ncclNet->accept(lComm, &rComm, NULL), ret, cleanup2, NCCL_NET);

        connected = (rComm != NULL) && (sComm != NULL);
      }

    /*申请gpu内存*/
      NCCLCHECKGOTONOWARN(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE, comm->memManager), ret, cleanup2, NCCL_NET);
    /*sComm注册mr*/
      NOWARN(ret = comm->ncclNet->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
      if (ret == ncclSuccess) {
    	/*注册mr成功，执行deregmr*/
        NCCLCHECKNOWARN(comm->ncclNet->deregMr(sComm, mHandle), NCCL_NET);
      /*rComm注册mr*/
        NCCLCHECKNOWARN(comm->ncclNet->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
        NCCLCHECKNOWARN(comm->ncclNet->deregMr(rComm, mHandle), NCCL_NET);
        gdrSupportMatrix[comm->cudaDev] = 1;/*当前cuda设备支持*/
      }
      NCCLCHECK(ncclCudaFree(gpuPtr, comm->memManager));
    cleanup2:
      if (rComm != NULL) NCCLCHECK(comm->ncclNet->closeRecv(rComm));
      if (sComm != NULL) NCCLCHECK(comm->ncclNet->closeSend(sComm));
      NCCLCHECK(comm->ncclNet->closeListen(lComm));
    cleanup1:
      break;
    }
  }
  *gdrSupport = gdrSupportMatrix[comm->cudaDev];
  return ncclSuccess;
}
