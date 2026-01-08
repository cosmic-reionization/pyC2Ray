#include "memory.h"
#include "utils.cuh"

#include <format>

namespace asora {

    device_buffer::device_buffer(size_t nbytes) : _nbytes(nbytes) {
        std::byte *ptr;
        safe_cuda(cudaMalloc(&ptr, _nbytes));
        _ptr.reset(ptr, [](std::byte *ptr) {
            try {
                safe_cuda(cudaFree(ptr));
            } catch (const std::exception &) {
            }
        });
    }

    void swap(device_buffer &lhs, device_buffer &rhs) noexcept {
        std::swap(lhs._ptr, rhs._ptr);
        std::swap(lhs._nbytes, rhs._nbytes);
    }

    void device_buffer::copyFromHost(const void *src, size_t nbytes) {
        if (size() != nbytes)
            throw std::invalid_argument("this device buffer is not large enough");
        safe_cuda(cudaMemcpy(data(), src, nbytes, cudaMemcpyHostToDevice));
    }

    void device_buffer::copyToHost(void *dst, size_t nbytes) const {
        if (size() != nbytes)
            throw std::invalid_argument("the destination buffer is not large enough");
        safe_cuda(cudaMemcpy(dst, data(), nbytes, cudaMemcpyDeviceToHost));
    }

    device &device::initialize(unsigned int rank) {
        // TODO: add log
        auto &self = instance();
        if (is_initialized()) return self;

        safe_cuda(cudaGetDeviceCount(&self._gpu_id));
        self._gpu_id = rank % self._gpu_id;
        safe_cuda(cudaSetDevice(self._gpu_id));
        return self;
    }

    void device::close() {
        auto &self = instance();
        self._gpu_id = -1;
        self._memory_pool.clear();
    }

    device_buffer &device::get(buffer_tag tag) {
        check_initialized();
        return instance()._memory_pool.at(tag);
    }

    bool device::contains(buffer_tag tag) {
        return is_initialized() && instance()._memory_pool.contains(tag);
    }

    device &device::instance() noexcept {
        static device self;
        return self;
    }

    void device::check_initialized(const std::source_location &loc) {
        if (!is_initialized()) {
            auto msg = std::format(
                "device not initialized at At {} in {}:{}; call "
                "asora::device::initialize(...) before",
                loc.function_name(), loc.file_name(), loc.line()
            );
            throw std::runtime_error(msg);
        }
    }

    void device::allocate_or_copy(buffer_tag tag, size_t nbytes, const void *src) {
        check_initialized();

        auto &&[it, success] = _memory_pool.try_emplace(tag, nbytes);
        if (!success && !src) throw std::runtime_error("tag already in use");
        if (src) it->second.copyFromHost(src, nbytes);
    }

}  // namespace asora
