#include "connection.hpp"

#include <algorithm>

#include "checks.hpp"
#include "infiniband/verbs.h"
#include "npkit/npkit.h"
#include "registered_memory.hpp"
#include "utils.hpp"

namespace mscclpp {

void validateTransport(RegisteredMemory mem, Transport transport) {
  if (!mem.transports().has(transport)) {
    throw Error("RegisteredMemory does not support this transport", ErrorCode::InvalidUsage);
  }
}

// Connection

std::shared_ptr<RegisteredMemory::Impl> Connection::getRegisteredMemoryImpl(RegisteredMemory& mem) { return mem.pimpl; }

// ConnectionBase

ConnectionBase::ConnectionBase(int remoteRank, int tag) : remoteRank_(remoteRank), tag_(tag) {}

int ConnectionBase::remoteRank() { return remoteRank_; }

int ConnectionBase::tag() { return tag_; }

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(int remoteRank, int tag, cudaStream_t stream)
    : ConnectionBase(remoteRank, tag), stream_(stream) {}

CudaIpcConnection::~CudaIpcConnection() {}

Transport CudaIpcConnection::transport() { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() { return Transport::CudaIpc; }

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  CUDATHROW(cudaMemcpyAsync(dstPtr + dstOffset, srcPtr + srcOffset, size, cudaMemcpyDeviceToDevice, stream_));
  INFO(MSCCLPP_P2P, "CudaIpcConnection write: from %p to %p, size %lu", srcPtr + srcOffset, dstPtr + dstOffset, size);

  // npkitCollectEntryEvent(conn, NPKIT_EVENT_DMA_SEND_DATA_ENTRY, (uint32_t)size);
}

void CudaIpcConnection::flush() {
  CUDATHROW(cudaStreamSynchronize(stream_));
  // npkitCollectExitEvents(conn, NPKIT_EVENT_DMA_SEND_EXIT);
}

// IBConnection

IBConnection::IBConnection(int remoteRank, int tag, Transport transport, Communicator::Impl& commImpl)
    : ConnectionBase(remoteRank, tag),
      transport_(transport),
      remoteTransport_(Transport::Unknown),
      numSignaledSends(0) {
  qp = commImpl.getIbContext(transport)->createQp();
}

Transport IBConnection::transport() { return transport_; }

Transport IBConnection::remoteTransport() { return remoteTransport_; }

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
  validateTransport(dst, remoteTransport());
  validateTransport(src, transport());

  auto dstTransportInfo = getRegisteredMemoryImpl(dst)->getTransportInfo(remoteTransport());
  if (dstTransportInfo.ibLocal) {
    throw Error("dst is local, which is not supported", ErrorCode::InvalidUsage);
  }
  auto srcTransportInfo = getRegisteredMemoryImpl(src)->getTransportInfo(transport());
  if (!srcTransportInfo.ibLocal) {
    throw Error("src is remote, which is not supported", ErrorCode::InvalidUsage);
  }

  auto dstMrInfo = dstTransportInfo.ibMrInfo;
  auto srcMr = srcTransportInfo.ibMr;

  qp->stageSend(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset, /*dstOffset=*/dstOffset,
                /*signaled=*/true);
  numSignaledSends++;
  qp->postSend();
  INFO(MSCCLPP_NET, "IBConnection write: from %p to %p, size %lu", (uint8_t*)srcMr->getBuff() + srcOffset,
       (uint8_t*)dstMrInfo.addr + dstOffset, size);
  // npkitCollectEntryEvent(conn, NPKIT_EVENT_IB_SEND_DATA_ENTRY, (uint32_t)size);
}

void IBConnection::flush() {
  Timer timer;
  while (numSignaledSends) {
    int wcNum = qp->pollCq();
    if (wcNum < 0) {
      throw mscclpp::IbError("pollCq failed: error no " + std::to_string(errno), errno);
    }

    auto elapsed = timer.elapsed();
    if (elapsed > MSCCLPP_POLLING_WAIT) {
      throw Error("pollCq is stuck: waited for " + std::to_string(elapsed / 1e6) + " seconds. Expected " +
                      std::to_string(numSignaledSends) + " signals",
                  ErrorCode::InternalError);
    }
    for (int i = 0; i < wcNum; ++i) {
      const ibv_wc* wc = qp->getWc(i);
      if (wc->status != IBV_WC_SUCCESS) {
        throw mscclpp::IbError("pollCq failed: status " + std::to_string(wc->status), wc->status);
      }
      if (wc->opcode == IBV_WC_RDMA_WRITE) {
        numSignaledSends--;
      }
    }
  }
  // npkitCollectExitEvents(conn, NPKIT_EVENT_IB_SEND_EXIT);
}

void IBConnection::beginSetup(std::shared_ptr<BaseBootstrap> bootstrap) {
  std::vector<char> ibQpTransport;
  std::copy_n(reinterpret_cast<char*>(&qp->getInfo()), sizeof(qp->getInfo()), std::back_inserter(ibQpTransport));
  std::copy_n(reinterpret_cast<char*>(&transport_), sizeof(transport_), std::back_inserter(ibQpTransport));

  bootstrap->send(ibQpTransport.data(), ibQpTransport.size(), remoteRank(), tag());
}

void IBConnection::endSetup(std::shared_ptr<BaseBootstrap> bootstrap) {
  std::vector<char> ibQpTransport(sizeof(IbQpInfo) + sizeof(Transport));
  bootstrap->recv(ibQpTransport.data(), ibQpTransport.size(), remoteRank(), tag());

  IbQpInfo qpInfo;
  auto it = ibQpTransport.begin();
  std::copy_n(it, sizeof(qpInfo), reinterpret_cast<char*>(&qpInfo));
  it += sizeof(qpInfo);
  std::copy_n(it, sizeof(remoteTransport_), reinterpret_cast<char*>(&remoteTransport_));
  it += sizeof(qpInfo);

  qp->rtr(qpInfo);
  qp->rts();
}

}  // namespace mscclpp