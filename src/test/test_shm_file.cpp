#include "memory/shm_file.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

std::string TempPath(const std::string& name) {
  return "/tmp/recstore_test_shm_file_" + name;
}

} // namespace

TEST(ShmFileTest, CreatesAnonymousDramFromFactory) {
  auto shm_file = base::ShmFile::New(
      {{"type", "ANONY_DRAM"}, {"filename", "anonymous"}, {"size", 4096}});

  ASSERT_NE(shm_file, nullptr);
  ASSERT_NE(shm_file->Data(), nullptr);
  EXPECT_EQ(shm_file->Size(), 4096);

  std::memcpy(shm_file->Data(), "ok", 3);
  EXPECT_STREQ(shm_file->Data(), "ok");
}

TEST(ShmFileTest, CreatesFsDaxFileFromFactory) {
  const std::string path = TempPath("fsdax");
  base::file_util::Delete(path, false);

  auto shm_file = base::ShmFile::New(
      {{"type", "FS_DAX"}, {"filename", path}, {"size", 4096}});

  ASSERT_NE(shm_file, nullptr);
  ASSERT_NE(shm_file->Data(), nullptr);
  EXPECT_EQ(shm_file->Size(), 4096);
  EXPECT_EQ(shm_file->filename(), path);

  std::memcpy(shm_file->Data(), "fs", 3);
  EXPECT_STREQ(shm_file->Data(), "fs");

  shm_file.reset();
  base::file_util::Delete(path, false);
}

TEST(ShmFileTest, FsDaxRejectsMismatchedExistingSize) {
  const std::string path = TempPath("fsdax_size");
  base::file_util::Delete(path, false);

  auto first = base::ShmFile::New(
      {{"type", "FS_DAX"}, {"filename", path}, {"size", 4096}});
  ASSERT_NE(first, nullptr);
  first.reset();

  auto second = base::ShmFile::New(
      {{"type", "FS_DAX"}, {"filename", path}, {"size", 8192}});
  EXPECT_EQ(second, nullptr);

  base::file_util::Delete(path, false);
}

TEST(ShmFileTest, ConfigForMemoryBackendUsesAnonymousDramByDefault) {
  auto config = base::ShmFile::ConfigForMedium("DRAM", "default", 1024);
  EXPECT_EQ(config.at("type").get<std::string>(), "ANONY_DRAM");
  EXPECT_EQ(config.at("filename").get<std::string>(), "default");
  EXPECT_EQ(config.at("size").get<int64>(), 1024);
}
