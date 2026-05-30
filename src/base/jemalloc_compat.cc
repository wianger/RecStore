#include <cstddef>

extern "C" {

void* jemallocx(std::size_t size, int flags) noexcept;
void* jerallocx(void* ptr, std::size_t size, int flags) noexcept;
std::size_t
jexallocx(void* ptr, std::size_t size, std::size_t extra, int flags) noexcept;
std::size_t jesallocx(const void* ptr, int flags) noexcept;
void jedallocx(void* ptr, int flags) noexcept;
void jesdallocx(void* ptr, std::size_t size, int flags) noexcept;
std::size_t jenallocx(std::size_t size, int flags) noexcept;
int jemallctl(const char* name,
              void* oldp,
              std::size_t* oldlenp,
              void* newp,
              std::size_t newlen) noexcept;
int jemallctlnametomib(
    const char* name, std::size_t* mibp, std::size_t* miblenp) noexcept;
int jemallctlbymib(const std::size_t* mib,
                   std::size_t miblen,
                   void* oldp,
                   std::size_t* oldlenp,
                   void* newp,
                   std::size_t newlen) noexcept;

void* mallocx(std::size_t size, int flags) noexcept {
  return jemallocx(size, flags);
}

void* rallocx(void* ptr, std::size_t size, int flags) noexcept {
  return jerallocx(ptr, size, flags);
}

std::size_t
xallocx(void* ptr, std::size_t size, std::size_t extra, int flags) noexcept {
  return jexallocx(ptr, size, extra, flags);
}

std::size_t sallocx(const void* ptr, int flags) noexcept {
  return jesallocx(ptr, flags);
}

void dallocx(void* ptr, int flags) noexcept { jedallocx(ptr, flags); }

void sdallocx(void* ptr, std::size_t size, int flags) noexcept {
  jesdallocx(ptr, size, flags);
}

std::size_t nallocx(std::size_t size, int flags) noexcept {
  return jenallocx(size, flags);
}

int mallctl(const char* name,
            void* oldp,
            std::size_t* oldlenp,
            void* newp,
            std::size_t newlen) noexcept {
  return jemallctl(name, oldp, oldlenp, newp, newlen);
}

int mallctlnametomib(
    const char* name, std::size_t* mibp, std::size_t* miblenp) noexcept {
  return jemallctlnametomib(name, mibp, miblenp);
}

int mallctlbymib(const std::size_t* mib,
                 std::size_t miblen,
                 void* oldp,
                 std::size_t* oldlenp,
                 void* newp,
                 std::size_t newlen) noexcept {
  return jemallctlbymib(mib, miblen, oldp, oldlenp, newp, newlen);
}

} // extern "C"
