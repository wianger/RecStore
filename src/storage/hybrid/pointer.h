#pragma once

#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>

class UnifiedPointer {
public:
	enum class Type : uint8_t {
		Memory = 0b00,
		Disk   = 0b01,
		PMem   = 0b10,
		Invalid= 0b11
	};

	UnifiedPointer() : raw_(0) {}

	static UnifiedPointer FromRaw(uint64_t raw) {
		UnifiedPointer p;
		p.raw_ = raw;
		return p;
	}

	// 构造内存指针
	static UnifiedPointer FromMemory(void* ptr) {
		UnifiedPointer p;
		p.set(Type::Memory, reinterpret_cast<uint64_t>(ptr));
		return p;
	}

	// 构造磁盘页ID
	static UnifiedPointer FromDiskPageId(uint64_t page_id) {
		UnifiedPointer p;
		p.set(Type::Disk, page_id);
		return p;
	}

	// 构造 PMem 指针
	static UnifiedPointer FromPMem(uint64_t pmem_offset) {
		UnifiedPointer p;
		p.set(Type::PMem, pmem_offset);
		return p;
	}

	Type type() const {
		return static_cast<Type>((raw_ >> 62) & 0b11);
	}

	uint64_t value() const {
		return raw_ & VALUE_MASK;
	}

	void* asMemoryPointer() const {
		assert(type() == Type::Memory);
		return reinterpret_cast<void*>(value());
	}

	uint64_t asDiskPageId() const {
		assert(type() == Type::Disk);
		return value();
	}

	uint64_t asPMemOffset() const {
		assert(type() == Type::PMem);
		return value();
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "Type: ";
		switch (type()) {
			case Type::Memory:
				oss << "Memory, Addr: 0x" << std::hex << value();
				break;
			case Type::Disk:
				oss << "Disk, PageID: " << std::dec << value();
				break;
			case Type::PMem:
				oss << "PMem, Offset: " << std::dec << value();
				break;
			case Type::Invalid:
			default:
				oss << "Invalid";
				break;
		}
		return oss.str();
	}

private:
	static constexpr uint64_t VALUE_MASK = (1ULL << 62) - 1;

	void set(Type t, uint64_t value) {
		assert((value & ~VALUE_MASK) == 0 && "Value overflow in 62 bits");
		raw_ = (static_cast<uint64_t>(t) << 62) | value;
	}

	uint64_t raw_;
};