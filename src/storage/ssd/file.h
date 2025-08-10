#pragma once

#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#define PAGE_SIZE 4096
typedef uint64_t PageID_t;
const PageID_t INVALID_PAGE = -1;
using boost::coroutines2::coroutine;

class FileManager {
public:
  FileManager(const std::string &file_path) : fd(-1), next_page_id(0) {
    bool exists = (access(file_path.c_str(), F_OK) != -1);
    fd =
        open(file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR);
    if (fd < 0)
      throw std::runtime_error("Failed to open file: " + file_path);
    if (exists) {
      struct stat file_stat;
      fstat(fd, &file_stat);
      next_page_id = file_stat.st_size / PAGE_SIZE;
    } else {
      next_page_id = 0;
    }
    // io_uring will be initialized per-thread to avoid contention
  }

  ~FileManager() {
    if (fd >= 0)
      close(fd);
  }

  PageID_t AllocatePage(coroutine<Value_t>::push_type &sink) {
    // Simple sequential allocation
    PageID_t new_page_id = next_page_id++;
    // Extend file size using io_uring
    WritePageAsync(sink, new_page_id, empty_page);
    return new_page_id;
  }
  PageID_t AllocatePage() {
    // Simple sequential allocation
    PageID_t new_page_id = next_page_id++;
    // Extend file size
    if (pwrite(fd, &empty_page, PAGE_SIZE, new_page_id * PAGE_SIZE) !=
        PAGE_SIZE)
      throw std::runtime_error("Failed to extend file for new page");
    return new_page_id;
  }

  void ReadPage(coroutine<Value_t>::push_type &sink, PageID_t page_id,
                char *buffer) {
    ReadPageAsync(sink, page_id, buffer);
  }
  void ReadPage(PageID_t page_id, char *buffer) {
    if (pread(fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE) != PAGE_SIZE)
      throw std::runtime_error("Failed to read page " +
                               std::to_string(page_id));
  }

  void WritePage(coroutine<Value_t>::push_type &sink, PageID_t page_id,
                 const char *buffer) {
    WritePageAsync(sink, page_id, buffer);
  }
  void WritePage(PageID_t page_id, const char *buffer) {
    if (pwrite(fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE) != PAGE_SIZE) {
      throw std::runtime_error("Failed to write page " +
                               std::to_string(page_id));
    }
  }

  template <typename T>
  T *GetPage(coroutine<Value_t>::push_type &sink, PageID_t page_id) {
    char *buffer = new char[PAGE_SIZE];
    ReadPage(sink, page_id, buffer);
    return reinterpret_cast<T *>(buffer);
  }
  template <typename T> T *GetPage(PageID_t page_id) {
    void *buffer;
    if (posix_memalign(&buffer, PAGE_SIZE, PAGE_SIZE) != 0)
      throw std::runtime_error("Failed to allocate aligned memory for page");
    ReadPage(page_id, reinterpret_cast<char *>(buffer));
    return reinterpret_cast<T *>(buffer);
  }

  // Unpin a page, if dirty, write it back
  void Unpin(coroutine<Value_t>::push_type &sink, PageID_t page_id,
             const void *page_data, bool is_dirty) {
    if (is_dirty)
      WritePage(sink, page_id, reinterpret_cast<const char *>(page_data));
    delete[] reinterpret_cast<const char *>(page_data);
  }
  void Unpin(PageID_t page_id, const void *page_data, bool is_dirty) {
    if (is_dirty)
      WritePage(page_id, reinterpret_cast<const char *>(page_data));
    free(const_cast<void *>(page_data));
  }

  int fd;

private:
  PageID_t next_page_id;
  char empty_page[PAGE_SIZE] = {0};
  // RAII wrapper for thread-local io_uring
  struct ThreadRing {
    struct io_uring ring;
    bool initialized = false;
    ThreadRing() = default;
    ~ThreadRing() {
      if (initialized)
        io_uring_queue_exit(&ring);
    }

    struct io_uring *get() {
      if (!initialized) {
        int ret = io_uring_queue_init(512, &ring, 0);
        if (ret < 0)
          throw std::runtime_error(
              "Failed to initialize thread-local io_uring: " +
              std::string(strerror(-ret)));
        initialized = true;
      }
      return &ring;
    }
  };

  // Get thread-local io_uring instance
  struct io_uring *get_thread_ring() {
    static thread_local ThreadRing thread_ring;
    return thread_ring.get();
  }

  void ReadPageAsync(coroutine<Value_t>::push_type &sink, PageID_t page_id,
                     char *buffer) {
    struct io_uring *ring = get_thread_ring();
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe)
      throw std::runtime_error("Failed to get SQE for read operation");
    // Prepare read operation
    io_uring_prep_read(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    // Submit and wait for completion
    int ret = io_uring_submit(ring);
    if (ret < 0)
      throw std::runtime_error("Failed to submit read operation: " +
                               std::string(strerror(-ret)));
    sink(NONE);
    struct io_uring_cqe *cqe;
    ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0)
      throw std::runtime_error("Failed to wait for read completion: " +
                               std::string(strerror(-ret)));
    if (cqe->res < 0) {
      io_uring_cqe_seen(ring, cqe);
      throw std::runtime_error("Read operation failed: " +
                               std::string(strerror(-cqe->res)));
    }
    if (cqe->res != PAGE_SIZE) {
      io_uring_cqe_seen(ring, cqe);
      throw std::runtime_error("Incomplete read: expected " +
                               std::to_string(PAGE_SIZE) + " bytes, got " +
                               std::to_string(cqe->res));
    }
    io_uring_cqe_seen(ring, cqe);
  }

  void WritePageAsync(coroutine<Value_t>::push_type &sink, PageID_t page_id,
                      const char *buffer) {
    struct io_uring *ring = get_thread_ring();
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe)
      throw std::runtime_error("Failed to get SQE for write operation");
    // Prepare write operation
    io_uring_prep_write(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    // Submit and wait for completion
    int ret = io_uring_submit(ring);
    if (ret < 0)
      throw std::runtime_error("Failed to submit write operation: " +
                               std::string(strerror(-ret)));
    sink(NONE);
    struct io_uring_cqe *cqe;
    ret = io_uring_wait_cqe(ring, &cqe);
    if (ret < 0)
      throw std::runtime_error("Failed to wait for write completion: " +
                               std::string(strerror(-ret)));
    if (cqe->res < 0) {
      io_uring_cqe_seen(ring, cqe);
      throw std::runtime_error("Write operation failed: " +
                               std::string(strerror(-cqe->res)));
    }
    if (cqe->res != PAGE_SIZE) {
      io_uring_cqe_seen(ring, cqe);
      throw std::runtime_error("Incomplete write: expected " +
                               std::to_string(PAGE_SIZE) + " bytes, wrote " +
                               std::to_string(cqe->res));
    }
    io_uring_cqe_seen(ring, cqe);
  }
};