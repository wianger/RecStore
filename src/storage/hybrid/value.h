#pragma once

#include "../../memory/persist_malloc.h"
#include "pointer.h"
#include <string_view>
#include <mutex>

class ValueManager {
public:
    ValueManager(const std::string& shm_file_path, size_t shm_capacity,
            const std::string& ssd_file_path, size_t ssd_capacity) 
        : shm_manage(shm_file_path, shm_capacity),ssd_manage(ssd_file_path, ssd_capacity) 
    {
        fd_ssd = open(ssd_file_path.c_str(),O_RDWR);
        if(fd_ssd < 0){
            LOG(ERROR) << "ssd open error";
        }
    }

    UnifiedPointer WriteValue(const std::string_view& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint16_t data_len = value.size();
        // base::PetKVData shmkv_data;
        size_t total_size = data_len + sizeof(data_len);
        char* ptr = shm_manage.New(total_size);
        if (ptr != nullptr) {
            // 写入长度前缀和数据
            if(shm_manage.GetMallocOffset(ptr) == -1){
                LOG(ERROR) <<"write memory failed";
                return UnifiedPointer();
            }
            uint16_t len = static_cast<uint16_t>(value.size());
            memcpy(ptr, &len, sizeof(uint16_t));
            memcpy(ptr + sizeof(uint16_t), value.data(), value.size());
            // 返回数据部分的指针
            return UnifiedPointer::FromMemory(ptr);
        }
        ptr = ssd_manage.New(total_size);
        if (ptr != nullptr){//当内存分配失败时在ssd上分配空间
            off_t offset = static_cast<off_t>(ssd_manage.GetMallocOffset(ptr));
            ssize_t bytes_written = pwrite(fd_ssd, &data_len, sizeof(data_len), offset);
            if (bytes_written != sizeof(data_len)) {
                perror("pwrite length failed!");
                ssd_manage.Free(ptr);
                return UnifiedPointer();
            }
            // 写入实际数据
            bytes_written = pwrite(fd_ssd, value.data(), data_len, offset + sizeof(data_len));
            
            // 确保数据写入磁盘
            if (bytes_written != data_len) {
                perror("pwrite data failed");
                ssd_manage.Free(ptr);
                return UnifiedPointer();
            }
            
            if (fsync(fd_ssd) < 0) {
                perror("fsync failed");
                ssd_manage.Free(ptr);
                return UnifiedPointer();
            }
            return UnifiedPointer::FromDiskPageId(static_cast<uint64_t>(offset));
        }
        return UnifiedPointer();
    }

    std::string RetrieveValue(const UnifiedPointer& p) {
        // UnifiedPointer p = UnifiedPointer::FromRaw(raw_value);
        if (p.type() != UnifiedPointer::Type::Invalid) {
            std::cout << "Raw pointer info: " << p.toString() << std::endl;
        } else {
            std::cout << "Raw pointer info: [Invalid Pointer]" << std::endl;
        }
        std::string value_;
        switch (p.type()) { // TODO: retrieve value from different tier
            case UnifiedPointer::Type::Memory: {
                void* mem_ptr = p.asMemoryPointer();
                const uint8_t* bytes = static_cast<const uint8_t*>(mem_ptr);
                // 从首部两个字节解析长度（小端序）
                uint16_t len = static_cast<uint16_t>(bytes[0]) | 
                            (static_cast<uint16_t>(bytes[1]) << 8);
                const char* str_data = reinterpret_cast<const char*>(bytes + 2);
                value_.assign(str_data, len);
                break;
            }
            case UnifiedPointer::Type::Disk: {
                off_t pointer = p.asDiskPageId();
                uint16_t value_len;
                ssize_t bytes_read = pread(fd_ssd, &value_len, sizeof(value_len), pointer);
                // 读取实际数据
                value_.resize(value_len);
                bytes_read = pread(fd_ssd, value_.data(), value_len, pointer + sizeof(value_len));
                if (bytes_read != value_len) {
                    if (bytes_read < 0) {
                        perror("pread data failed");
                    } else {
                        perror("Incomplete data read at pointer ");
                    }
                }
                break;
            }
            case UnifiedPointer::Type::PMem: {
                throw std::runtime_error("PMem not implemented");   
            }
            case UnifiedPointer::Type::Invalid:
                throw std::runtime_error("Invalid pointer type");
            default:
                throw std::runtime_error("Unknown pointer type");
            }
        return value_;
    }

private:
    base::PersistLoopShmMalloc shm_manage;
    base::PersistLoopShmMalloc ssd_manage;
    int fd_ssd = -1;
    std::mutex mutex_;
};