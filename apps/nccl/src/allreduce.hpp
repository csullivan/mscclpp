// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLREDUCE_HPP_
#define ALLREDUCE_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"
#include "cuda_fp8.h"

__device__ mscclpp::DeviceSyncer deviceSyncer;

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}
  __host__ __device__ explicit half4(const __nv_fp8x4_e4m3& fp8x4) {
    __nv_fp8x2_e4m3 lo_part, hi_part;
    lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
    hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
    __half2 lo_half2 = static_cast<__half2>(lo_part);
    __half2 hi_half2 = static_cast<__half2>(hi_part);
    x = reinterpret_cast<__half*>(&lo_half2)[0];
    y = reinterpret_cast<__half*>(&lo_half2)[1];
    z = reinterpret_cast<__half*>(&hi_half2)[0];
    w = reinterpret_cast<__half*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e4m3() const {
    __nv_fp8x4_e4m3 result;
    __half2 lo_half2 = *reinterpret_cast<const __half2*>(&x);
    __half2 hi_half2 = *reinterpret_cast<const __half2*>(&z);
    __nv_fp8x2_e4m3 lo_part(lo_half2), hi_part(hi_half2);
    result.__x = (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4(const __nv_fp8x4_e5m2& fp8x4) {
    __nv_fp8x2_e5m2 lo_part, hi_part;
    lo_part.__x = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
    hi_part.__x = static_cast<__nv_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
    __half2 lo_half2 = static_cast<__half2>(lo_part);
    __half2 hi_half2 = static_cast<__half2>(hi_part);
    x = reinterpret_cast<__half*>(&lo_half2)[0];
    y = reinterpret_cast<__half*>(&lo_half2)[1];
    z = reinterpret_cast<__half*>(&hi_half2)[0];
    w = reinterpret_cast<__half*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __nv_fp8x4_e5m2() const {
    __nv_fp8x4_e5m2 result;
    __half2 lo_half2 = *reinterpret_cast<const __half2*>(&x);
    __half2 hi_half2 = *reinterpret_cast<const __half2*>(&z);
    __nv_fp8x2_e5m2 lo_part(lo_half2), hi_part(hi_half2);
    result.__x = (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __device__ __nv_fp8x2_e5m2 make_fp8x2_e5m2(__nv_fp8_storage_t x, __nv_fp8_storage_t y) {
    __nv_fp8x2_e5m2 result;
    result.__x = (x) | (y << 8);
    return result;
  }
  __device__ __nv_fp8x4_e5m2 make_fp8x4_e5m2(__nv_fp8_storage_t a, __nv_fp8_storage_t b, __nv_fp8_storage_t c,
                                             __nv_fp8_storage_t d) {
    __nv_fp8x4_e5m2 result;
    result.__x = (a) | (b << 8) | (c << 16) | (d << 24);
    return result;
  }
  __device__ __nv_fp8x2_e4m3 make_fp8x2_e4m3(__nv_fp8_storage_t x, __nv_fp8_storage_t y) {
    __nv_fp8x2_e4m3 result;
    result.__x = (x) | (y << 8);
    return result;
  }
  __device__ __nv_fp8x4_e4m3 make_fp8x4_e4m3(__nv_fp8_storage_t a, __nv_fp8_storage_t b, __nv_fp8_storage_t c,
                                             __nv_fp8_storage_t d) {
    __nv_fp8x4_e4m3 result;
    result.__x = (a) | (b << 8) | (c << 16) | (d << 24);
    return result;
  }

  __device__ __nv_fp8x4_e4m3 operator+(const half4& rhs) const {
    __half2 lhs_lo = *reinterpret_cast<const __half2*>(&x);
    __half2 rhs_lo = *reinterpret_cast<const __half2*>(&rhs.x);
    __half2 lhs_hi = *reinterpret_cast<const __half2*>(&z);
    __half2 rhs_hi = *reinterpret_cast<const __half2*>(&rhs.z);

    __half2 lo_result = __hadd2(lhs_lo, rhs_lo);
    __half2 hi_result = __hadd2(lhs_hi, rhs_hi);
    __nv_fp8x2_e4m3 lo_part(lo_result), hi_part(hi_result);

    __nv_fp8x4_e4m3 result;
    result.__x = (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
};

__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) { return half4(x, y, z, w); }

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <>
__forceinline__ __device__ __nv_fp8x4_e4m3 add_elements(__nv_fp8x4_e4m3 a, __nv_fp8x4_e4m3 b) {
  return half4(a) + half4(b);
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <>
__forceinline__ __device__ int4 add_vectors_helper<__nv_fp8x4_e4m3>(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, __nv_fp8x4_e4m3>(
      add_elements(bit_cast<__nv_fp8x4_e4m3, int>(a.w), bit_cast<__nv_fp8x4_e4m3, int>(b.w)));
  ret.x = bit_cast<int, __nv_fp8x4_e4m3>(
      add_elements(bit_cast<__nv_fp8x4_e4m3, int>(a.x), bit_cast<__nv_fp8x4_e4m3, int>(b.x)));
  ret.y = bit_cast<int, __nv_fp8x4_e4m3>(
      add_elements(bit_cast<__nv_fp8x4_e4m3, int>(a.y), bit_cast<__nv_fp8x4_e4m3, int>(b.y)));
  ret.z = bit_cast<int, __nv_fp8x4_e4m3>(
      add_elements(bit_cast<__nv_fp8x4_e4m3, int>(a.z), bit_cast<__nv_fp8x4_e4m3, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__nv_fp8_e4m3>(int4 a, int4 b) {
  return add_vectors_helper<__nv_fp8x4_e4m3>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
__forceinline__ __device__ uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (size_t i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<T>(dst4[i], src4[i]);
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*)dst) + nInt4 * 4;
    int* srcLast = ((int*)src) + nInt4 * 4;
    for (size_t i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
      dstLast[i] = add_vectors<T>(dstLast[i], srcLast[i]);
    }
  }
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

template <typename T>
__global__ void __launch_bounds__(32, 1)
    allreduceAllToAll(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  if (sizeof(T) == 2) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerNode - 1;
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = smChannels[lid];
  }
  __syncwarp();

  // step 1: write data to each peer's scratch buffer
  channels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
                                                   blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems; idx += blockDim.x * gridDim.x) {
    uint32_t data = 0;
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;
  }
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce7(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
               size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  nelems = nelems / (sizeof(int) / sizeof(T));
  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  size_t scratchOffset = channelScratchOffset + rank * nPktsPerRank * sizeof(mscclpp::LL8Packet);
  size_t scratchResultOffset = channelScratchOffset + 2 * nPkts * sizeof(mscclpp::LL8Packet);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int) + channelDataOffset;
  uint32_t* src = (uint32_t*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint32_t* dst = (uint32_t*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // Put channels into shared memory, read channel info from global memory is unexpectable slow.
  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = smChannels[lid];
  }
  __syncwarp();

  // step 1: write to scratch buffer
  channels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                                                   blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint32_t data = 0;
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nPktsPerRank;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LL8Packet packet;
    packet.data = data;
    packet.flag = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LL8Packet) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      channels[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint32_t* result = (uint32_t*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint32_t data = dstPkt[idx + dstOffset].read(flag, -1);
    result[idx] = data;
  }
}

template <typename T>
__global__ void __launch_bounds__(512, 1)
    allreduce8(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, size_t channelOutDataOffset,
               size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  const int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by (16 * worldSize)
  const size_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;
  auto smChans = smChannels + chanOffset;
  auto smOutChans = smOutChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch + channelScratchOffset);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
      (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * 256 / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t chunkSizePerRank = nNeededBlocks * nInt4PerChunk;
  const size_t blockOffset = nInt4PerChunk * blockIdx.x;
  const size_t scratchChunkRankOffset = chunkSizePerRank * rank;
  const size_t scratchBaseOffsetInt4 = channelScratchOffset / sizeof(int4);

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> outChannels[NRANKS_PER_NODE - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = smChans[lid];
    outChannels[lid] = smOutChans[lid];
  }
  __syncwarp();

  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchBaseOffsetInt4 + scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    /// Starts reduce-scatter
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock + channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
    offsetOfThisBlock += nInt4PerChunk;
  }
  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchBaseOffsetInt4 + scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock + channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
  }
}

template <typename T>
cudaError_t allreduce(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, size_t channelInOffset,
                      size_t channelOutOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, cudaStream_t stream) {
  static uint32_t flag = 1;

  if (sizeof(T) * nelems < worldSize * sizeof(int)) {
    int nBlocks = 7;
    int nThreadsPerBlock = 32;
    allreduceAllToAll<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, channelInOffset,
                                                                channelScratchOffset, rank, nRanksPerNode, worldSize,
                                                                nelems, flag++);
  } else if (sizeof(T) * nelems <= (1 << 20)) {
    int nBlocks = 28;
    int nThreadsPerBlock = 1024;
    if (nelems >= 8192) {
      nBlocks = 56;
      nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
    }
    allreduce7<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, channelInOffset,
                                                         channelScratchOffset, rank, nRanksPerNode, worldSize, nelems,
                                                         flag++);
  } else {
    int nBlocks = 35;
    int nThreadsPerBlock = 512;
    allreduce8<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, smOutChannels,
                                                         channelOutOffset, channelScratchOffset, rank, nRanksPerNode,
                                                         worldSize, nelems);
  }

  return cudaGetLastError();
}

#endif  // ALLREDUCE_KERNEL_H
