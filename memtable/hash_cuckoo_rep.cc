//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
#ifndef YUIL
#ifndef ROCKSDB_LITE
#include "memtable/hash_cuckoo_rep.h"
#include "memtable/concurrentqueue.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "db/memtable.h"
#include "memtable/skiplist.h"
#include "memtable/stl_wrappers.h"
#include "port/port.h"
#include "rocksdb/memtablerep.h"
#include "util/murmurhash.h"
// YUIL
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "memtable/persca_skiplist.h"



using namespace moodycamel;

namespace rocksdb {
	namespace {
		
		// the default maximum size of the cuckoo path searching queue
		static const int kCuckooPathMaxSearchSteps = 100;
		static const unsigned int kIndexJobBucket = 0;
		static const unsigned int kIndexJobBackup = 1;
		struct CuckooStep {
			static const int kNullStep = -1;
			// the bucket id in the cuckoo array.
			int bucket_id_;
			// index of cuckoo-step array that points to its previous step,
			// -1 if it the beginning step.
			int prev_step_id_;
			// the depth of the current step.
			unsigned int depth_;

			CuckooStep() : bucket_id_(-1), prev_step_id_(kNullStep), depth_(1) {}

			CuckooStep(CuckooStep&& o) = default;

			CuckooStep& operator=(CuckooStep&& rhs) {
				bucket_id_ = std::move(rhs.bucket_id_);
				prev_step_id_ = std::move(rhs.prev_step_id_);
				depth_ = std::move(rhs.depth_);
				return *this;
			}

			CuckooStep(const CuckooStep&) = delete;
			CuckooStep& operator=(const CuckooStep&) = delete;

			CuckooStep(int bucket_id, int prev_step_id, int depth)
				: bucket_id_(bucket_id), prev_step_id_(prev_step_id), depth_(depth) {}
		};

		class HashCuckooRep : public MemTableRep {
		public:
			explicit HashCuckooRep(const MemTableRep::KeyComparator& compare,
				Allocator* allocator, const size_t bucket_count,
				const unsigned int hash_func_count,
				const size_t approximate_entry_size)
				: MemTableRep(allocator),
				compare_(compare),
				allocator_(allocator),
				bucket_count_(bucket_count),
				approximate_entry_size_(approximate_entry_size),
				cuckoo_path_max_depth_(kDefaultCuckooPathMaxDepth),
				occupied_count_(0),
				hash_function_count_(hash_func_count),
				// YUIL
				indexlist_(new PerscaSkipList<const char*>(compare)),
				yul_background_worker_done(true) {
				char* mem = reinterpret_cast<char*>(
					allocator_->Allocate(sizeof(std::atomic<const char*>) * bucket_count_));
				cuckoo_array_ = new (mem) std::atomic<char*>[bucket_count_];
				for (unsigned int bid = 0; bid < bucket_count_; ++bid) {
					cuckoo_array_[bid].store(nullptr, std::memory_order_relaxed);
				}

				cuckoo_path_ = reinterpret_cast<int*>(
					allocator_->Allocate(sizeof(int) * (cuckoo_path_max_depth_ + 1)));
				is_nearly_full_ = false;

				//for (int i = 0; i < kDefaultMaxBackgroundWorker; ++i) {
				//	std::thread th(BackgroundWorker, this);
				//}
				//std::lock_guard<std::mutex> lk(yul_background_worker_mutex);
			}

			// return false, indicating HashCuckooRep does not support merge operator.
			virtual bool IsMergeOperatorSupported() const override { return false; }

			// return false, indicating HashCuckooRep does not support snapshot.
			virtual bool IsSnapshotSupported() const override { return false; }

			// Returns true iff an entry that compares equal to key is in the collection.
			virtual bool Contains(const char* internal_key) const override;

			virtual ~HashCuckooRep() override {}

			// Insert the specified key (internal_key) into the mem-table.  Assertion
			// fails if
			// the current mem-table already contains the specified key.
			virtual void Insert(KeyHandle handle) override;

			virtual void InsertConcurrently(KeyHandle handle) override;

			// This function returns bucket_count_ * approximate_entry_size_ when any
			// of the followings happen to disallow further write operations:
			// 1. when the fullness reaches kMaxFullnes.
			// 2. when the backup_table_ is used.
			//
			// otherwise, this function will always return 0.
			virtual size_t ApproximateMemoryUsage() override {
				if (is_nearly_full_) {
					return bucket_count_ * approximate_entry_size_;
				}
				return 0;
			}

			virtual void Get(const LookupKey& k, void* callback_args,
				bool(*callback_func)(void* arg,
					const char* entry)) override;

			class Iterator : public MemTableRep::Iterator {
				// YUIL
#ifndef YUIL
				PerscaSkipList<const char*>::Iterator* cit_;
				const KeyComparator& compare_;
				std::atomic<char*>* cuckoo_array_;
				unsigned int bucket_count_;
				std::string tmp_;  // For passing to EncodeKey
#endif

			public:
				//YUIL
				//explicit Iterator(std::shared_ptr<std::vector<const char*>> bucket,
				//                  const KeyComparator& compare);
				explicit Iterator(PerscaSkipList<const char*>::Iterator* list, const KeyComparator& compare, std::atomic<char*>* cuckoo_arr, const unsigned int& bucket_count_);

				// Initialize an iterator over the specified collection.
				// The returned iterator is not valid.
				// explicit Iterator(const MemTableRep* collection);
				virtual ~Iterator() override {};

				// Returns true iff the iterator is positioned at a valid node.
				virtual bool Valid() const override;

				// Returns the key at the current position.
				// REQUIRES: Valid()
				virtual const char* key() const override;

				// Advances to the next position.
				// REQUIRES: Valid()
				virtual void Next() override;

				// Advances to the previous position.
				// REQUIRES: Valid()
				virtual void Prev() override;

				// Advance to the first entry with a key >= target
				virtual void Seek(const Slice& user_key, const char* memtable_key) override;

				// Retreat to the last entry with a key <= target
				virtual void SeekForPrev(const Slice& user_key,
					const char* memtable_key) override;

				// Position at the first entry in collection.
				// Final state of iterator is Valid() iff collection is not empty.
				virtual void SeekToFirst() override;

				// Position at the last entry in collection.
				// Final state of iterator is Valid() iff collection is not empty.
				virtual void SeekToLast() override;
			};

			struct CuckooStepBuffer {
				CuckooStepBuffer() : write_index_(0), read_index_(0) {}
				~CuckooStepBuffer() {}

				int write_index_;
				int read_index_;
				CuckooStep steps_[kCuckooPathMaxSearchSteps];

				CuckooStep& NextWriteBuffer() { return steps_[write_index_++]; }

				inline const CuckooStep& ReadNext() { return steps_[read_index_++]; }

				inline bool HasNewWrite() { return write_index_ > read_index_; }

				inline void reset() {
					write_index_ = 0;
					read_index_ = 0;
				}

				inline bool IsFull() { return write_index_ >= kCuckooPathMaxSearchSteps; }

				// returns the number of steps that has been read
				inline int ReadCount() { return read_index_; }

				// returns the number of steps that has been written to the buffer.
				inline int WriteCount() { return write_index_; }
			};

			struct IndexJob {
				IndexJob(const char* key, const unsigned int& bucket, const int& type) :
					indexkey(key), bucket_id(bucket), Type(type) {}
				IndexJob() : indexkey(nullptr), bucket_id(0), Type(kIndexJobBucket) {}
				~IndexJob() {}

				const char* indexkey;
				unsigned int bucket_id;
				unsigned int Type;


				// returns index key(internal_key => Object)
				inline const char* IndexKey() const { return indexkey; }
				// returns bucket id about indexkey
				inline unsigned int BucketId() const { return bucket_id; }
			};


		private:
			const MemTableRep::KeyComparator& compare_;
			// the pointer to Allocator to allocate memory, immutable after construction.
			Allocator* const allocator_;
			// the number of hash bucket in the hash table.
			const size_t bucket_count_;
			// approximate size of each entry
			const size_t approximate_entry_size_;
			// the maxinum depth of the cuckoo path.
			const unsigned int cuckoo_path_max_depth_;
			// the current number of entries in cuckoo_array_ which has been occupied.
			std::atomic<size_t> occupied_count_;
			// the current number of hash functions used in the cuckoo hash.
			unsigned int hash_function_count_;
			// the backup MemTableRep to handle the case where cuckoo hash cannot find
			// a vacant bucket for inserting the key of a put request.
			//std::shared_ptr<MemTableRep> backup_table_;
			// the array to store pointers, pointing to the actual data.
			std::atomic<char*>* cuckoo_array_;
			// a buffer to store cuckoo path
			int* cuckoo_path_;
			// a boolean flag indicating whether the fullness of bucket array
			// reaches the point to make the current memtable immutable.
			bool is_nearly_full_;
			// YUIL
			// a locking mutex when collision happend for modifying cuckoo hash
			// used when make cuckoo path and modify the cuckoo bucket.
			std::mutex cuckoo_path_building_mutex_;

			public:
			// YUIL
			// for making index these workers have to do their jobs in work_queue
			// How many workers will be created
			static const unsigned int kDefaultMaxBackgroundWorker = 1;
			
			// YUIL
			// queue includes all cuckoo indicies.
			// all jobs in queue will be done by background thread
			ConcurrentQueue<IndexJob> yul_work_queue_;

			// YUIL
			// all jobs in queue will be done by background thread
			std::vector<std::thread*> yul_background_worker;
			//std::thread yul_background_worker;

			// YUIL
			// CV for background workers
			std::condition_variable yul_background_worker_cv;
			std::condition_variable yul_background_worker_done_cv;

			// YUIL
			// Mutex for background workers
			std::mutex yul_background_worker_mutex;
			std::mutex yul_background_worker_done_mutex;

			// YUIL
			// boolean for Wakeup Signal
			bool yul_background_worker_done;

			// the default maximum depth of the cuckoo path.
			static const unsigned int kDefaultCuckooPathMaxDepth = 10;

			CuckooStepBuffer step_buffer_;

			// YUIL
			std::shared_ptr<PerscaSkipList<const char*>> indexlist_;


			// returns the bucket id assogied to the input slice based on the
			unsigned int GetHash(const Slice& slice, const int hash_func_id) const {
				// the seeds used in the Murmur hash to produce different hash functions.
				static const int kMurmurHashSeeds[HashCuckooRepFactory::kMaxHashCount] = {
					545609244,  1769731426, 763324157,  13099088,   592422103,
					1899789565, 248369300,  1984183468, 1613664382, 1491157517 };
				return static_cast<unsigned int>(
					MurmurHash(slice.data(), static_cast<int>(slice.size()),
						kMurmurHashSeeds[hash_func_id]) %
					bucket_count_);
			}

			// A cuckoo path is a sequence of bucket ids, where each id points to a
			// location of cuckoo_array_.  This path describes the displacement sequence
			// of entries in order to store the desired data specified by the input user
			// key.  The path starts from one of the locations associated with the
			// specified user key and ends at a vacant space in the cuckoo array. This
			// function will update the cuckoo_path.
			//
			// @return true if it found a cuckoo path.
			bool FindCuckooPath(const char* internal_key, const Slice& user_key,
				int* cuckoo_path, size_t* cuckoo_path_length,
				int initial_hash_id = 0);

			bool FindCuckooPathConcurrently(const char* internal_key, const Slice& user_key,
				size_t* cuckoo_path_length, int bucket_ids[],
				int local_cuckoo_path[], int initial_hash_id = 0);

			// Perform quick insert by checking whether there is a vacant bucket in one
			// of the possible locations of the input key.  If so, then the function will
			// return true and the key will be stored in that vacant bucket.
			//
			// This function is a helper function of FindCuckooPath that discovers the
			// first possible steps of a cuckoo path.  It begins by first computing
			// the possible locations of the input keys (and stores them in bucket_ids.)
			// Then, if one of its possible locations is vacant, then the input key will
			// be stored in that vacant space and the function will return true.
			// Otherwise, the function will return false indicating a complete search
			// of cuckoo-path is needed.
			bool QuickInsert(const char* internal_key, const Slice& user_key,
				int bucket_ids[], const int initial_hash_id);
			bool QuickInsertConcurrently(const char* internal_key, const Slice& user_key,
				int bucket_ids[], const int initial_hash_id);

			// YUIL
			// Index List(Persca Skiplist)에 Bucket Key와 Index 삽입
			void InsertIndexData(const char* internal_key, const unsigned int& bucket_id);

			// BackupTable 대신에 Skiplist 사용해서 바로 넣어준다.
			void InsertBackupData(const char * internal_key, const unsigned int & bucket_id);

			// BackupTable(Skiplist) 에 키가 있는지 뒤져준다.
			const char* GetFromBackupTable(const char* internal_key);

			// Background Worker는 Queue를 보고 Indexlist를 asynchronous 하게 업데이트 한다.
			public:
			//void BackgroundWorker();
			inline void InsertJob(const IndexJob& job);

			inline bool GetJob(IndexJob& job) {
				return yul_work_queue_.try_dequeue(job);
			}

			inline size_t GetWorkQueueSize() {
				return yul_work_queue_.size_approx();
			}

			void Regist_BackgroundWorker(std::thread* th) {
				yul_background_worker.push_back(th);
			}


			// Returns the pointer to the internal iterator to the buckets where buckets
			// are sorted according to the user specified KeyComparator.  Note that
			// any insert after this function call may affect the sorted nature of
			// the returned iterator.
			virtual MemTableRep::Iterator* GetIterator(Arena* arena) override {
				std::unique_lock<std::mutex> lock(yul_background_worker_done_mutex);
				yul_background_worker_done_cv.wait(lock, [=] { return yul_background_worker_done; });
				auto list = indexlist_.get();
				auto it = new PerscaSkipList<const char*>::Iterator(list);
				if (arena == nullptr) {
					return new Iterator(it, compare_, cuckoo_array_,static_cast<unsigned int>(bucket_count_));
				}
				else {
					auto mem = arena->AllocateAligned(sizeof(Iterator));
					return new (mem) Iterator(it, compare_, cuckoo_array_, static_cast<unsigned int>(bucket_count_));
				}
			}

		};

		void HashCuckooRep::Get(const LookupKey& key, void* callback_args,
			bool(*callback_func)(void* arg, const char* entry)) {
			Slice user_key = key.user_key();
			for (unsigned int hid = 0; hid < hash_function_count_; ++hid) {
				const char* bucket =
					cuckoo_array_[GetHash(user_key, hid)].load(std::memory_order_acquire);
				if (bucket != nullptr) {
					Slice bucket_user_key = UserKey(bucket);
					if (user_key == bucket_user_key) {
						callback_func(callback_args, bucket);
						break;
					}
				}
				else {
					// as Put() always stores at the vacant bucket located by the
					// hash function with the smallest possible id, when we first
					// find a vacant bucket in Get(), that means a miss.
					break;
				}
			}
			
			const char* bucket = GetFromBackupTable(user_key.data());
			if (bucket != nullptr) {
				callback_func(callback_args, bucket);
			}
		}

		const char* HashCuckooRep::GetFromBackupTable(const char* internal_key) {
			auto Indexlist_ = indexlist_.get();
			return Indexlist_->GetWrappedValue(internal_key);
		}

		//void HashCuckooRep::BackgroundWorker()
		//{
		//	std::unique_lock<std::mutex> lock(yul_background_worker_mutex);
		//	IndexJob job;
		//		while (yul_work_queue_.size_approx() >= 1) {

		//			if (yul_work_queue_.size_approx() == 0) yul_background_worker_cv.wait(lock);
		//			if (yul_work_queue_.try_dequeue(job)) {
		//				// 만약 큐에서 Job 을 성공적으로 꺼냈으면 Insert
		//				InsertIndexData(job.IndexKey(), job.BucketId());
		//				//list->Insert(job.IndexKey(), job.BucketId());
		//			}
		//		}
		//}

		inline void HashCuckooRep::InsertJob(const IndexJob& job) {
			if (yul_work_queue_.try_enqueue(job) == false) {
				// Queue에 넣는거 실패한경우..
				// 그냥 바로 삽입하자
				InsertIndexData(job.IndexKey(), job.BucketId());
				return;
			}
			
			if (yul_background_worker_done) {
				// Background Worker가 자고있으면 일단 다 깨운다~~
				yul_background_worker_cv.notify_one();
			}
		}

		void HashCuckooRep::InsertConcurrently(KeyHandle handle) {
			static const float kMaxFullness = 0.90f;

		CUCKOOCOLLISIONMOD:
			auto* key = static_cast<char*>(handle);
			int initial_hash_id = 0;
			size_t cuckoo_path_length = 0;
			auto user_key = UserKey(key);
			int bucket_ids[HashCuckooRepFactory::kMaxHashCount];

			if (QuickInsertConcurrently(key, user_key, bucket_ids, initial_hash_id) == false) {
				// Hash 에 바로 넣기 실패했으면 Cuckoo Path 해줘야함.
				int local_cuckoo_path_[kCuckooPathMaxSearchSteps] = { 0, };
				cuckoo_path_building_mutex_.lock();
				if (FindCuckooPathConcurrently(key, user_key, &cuckoo_path_length, bucket_ids,
					local_cuckoo_path_, initial_hash_id) == false) {
					// 만약 Path 생성도 실패하고 빈 공간도 없으면
					// 그냥 Skiplist에 통째로 넣는다. (BackupTable)
					cuckoo_path_building_mutex_.unlock();
					is_nearly_full_ = true;
					InsertBackupData(key, static_cast<unsigned int>(bucket_count_));
					//InsertJob(IndexJob(key, static_cast<unsigned int>(bucket_count_),kIndexJobBackup));
					return;
				}
				else {
					// Path 생성에 성공했다면..
					if (cuckoo_path_length == 0) return;
					char* indexkey = key;
					// 마지막 Path 만 Compare And Swap 해보고 안되면 Path 다시 찾게 하면됨
					int kicked_out_bid = local_cuckoo_path_[0];
					int current_bid = local_cuckoo_path_[1];
					indexkey = cuckoo_array_[current_bid].load(std::memory_order_relaxed);
					char* st_key = cuckoo_array_[kicked_out_bid];
					if (st_key == nullptr && cuckoo_array_[kicked_out_bid].compare_exchange_weak(st_key, indexkey)) {
						for (size_t i = 2; i < cuckoo_path_length; ++i) {
							kicked_out_bid = local_cuckoo_path_[i - 1];
							current_bid = local_cuckoo_path_[i];
							// since we only allow one writer at a time, it is safe to do relaxed read.
							indexkey = cuckoo_array_[current_bid].load(std::memory_order_relaxed);
							cuckoo_array_[kicked_out_bid]
								.store(indexkey, std::memory_order_release);
							// 쫒겨난 키들에 대해 Index update
							//InsertIndexData(indexkey, kicked_out_bid);
							InsertJob(IndexJob(indexkey, static_cast<unsigned int>(kicked_out_bid),kIndexJobBucket));
						}
						int insert_key_bid = local_cuckoo_path_[cuckoo_path_length - 1];
						cuckoo_array_[insert_key_bid].store(key, std::memory_order_release);
						cuckoo_path_building_mutex_.unlock();
						//InsertIndexData(indexkey, insert_key_bid);
						InsertJob(IndexJob(indexkey, static_cast<unsigned int>(insert_key_bid), kIndexJobBucket));
					} // Swap 실패하면
					else {
						cuckoo_path_building_mutex_.unlock();
						goto CUCKOOCOLLISIONMOD;
					}
				}
			}
			// find cuckoo path

			// when reaching this point, means the insert can be done successfully.
			occupied_count_.fetch_add(1, std::memory_order_relaxed); // occupied_count_++;
			if (occupied_count_.load(std::memory_order_relaxed) >= bucket_count_ * kMaxFullness) {
				is_nearly_full_ = true;
			}
		}

		void HashCuckooRep::Insert(KeyHandle handle) {
			static const float kMaxFullness = 0.90f;
			auto* key = static_cast<char*>(handle);
			int initial_hash_id = 0;
			size_t cuckoo_path_length = 0;
			auto user_key = UserKey(key);
			// find cuckoo path
			if (FindCuckooPath(key, user_key, cuckoo_path_, &cuckoo_path_length,
				initial_hash_id) == false) {
				// if true, then we can't find a vacant bucket for this key even we
				// have used up all the hash functions.  Then use a backup memtable to
				// store such key, which will further make this mem-table become
				// immutable.
				// 만들기 실패하면 Skiplist에 통째로 넣자.

				InsertBackupData(key, static_cast<unsigned int>(bucket_count_));
				//InsertJob(IndexJob(key, static_cast<unsigned int>(bucket_count_), kIndexJobBackup));
				return;
			}
			// when reaching this point, means the insert can be done successfully.
			occupied_count_.fetch_add(1, std::memory_order_relaxed); // occupied_count_++;
			if (occupied_count_.load(std::memory_order_relaxed) >= bucket_count_ * kMaxFullness) {
				is_nearly_full_ = true;
			}

			// perform kickout process if the length of cuckoo path > 1.
			if (cuckoo_path_length == 0) return;

			// the cuckoo path stores the kickout path in reverse order.
			// so the kickout or displacement is actually performed
			// in reverse order, which avoids false-negatives on read
			// by moving each key involved in the cuckoo path to the new
			// location before replacing it.
			//PerscaSkipList<const char*>* isl = indexlist_.get();
			char* indexkey = key;
			for (size_t i = 1; i < cuckoo_path_length; ++i) {
				int kicked_out_bid = cuckoo_path_[i - 1];
				int current_bid = cuckoo_path_[i];
				// since we only allow one writer at a time, it is safe to do relaxed read.
				indexkey = cuckoo_array_[current_bid].load(std::memory_order_relaxed);
				cuckoo_array_[kicked_out_bid]
					.store(indexkey, std::memory_order_release);
				InsertIndexData(indexkey, kicked_out_bid);
				//InsertJob(IndexJob(indexkey, static_cast<unsigned int>(kicked_out_bid), kIndexJobBucket));
			}
			int insert_key_bid = cuckoo_path_[cuckoo_path_length - 1];
			cuckoo_array_[insert_key_bid].store(key, std::memory_order_release);
			InsertIndexData(indexkey, insert_key_bid);
			//InsertJob(IndexJob(indexkey, static_cast<unsigned int>(insert_key_bid), kIndexJobBucket));
		}

		bool HashCuckooRep::Contains(const char* internal_key) const {
			auto user_key = UserKey(internal_key);
			for (unsigned int hid = 0; hid < hash_function_count_; ++hid) {
				const char* stored_key =
					cuckoo_array_[GetHash(user_key, hid)].load(std::memory_order_acquire);
				if (stored_key != nullptr) {
					if (compare_(internal_key, stored_key) == 0) {
						return true;
					}
				}
			}
			return false;
		}

		inline void HashCuckooRep::InsertIndexData(const char* internal_key, const unsigned int& bucket_id) {
			PerscaSkipList<const char*>* Indexlist_ = indexlist_.get();
			const char* indexkey = internal_key;
			Indexlist_->Insert(indexkey, bucket_id);
		}

		inline void HashCuckooRep::InsertBackupData(const char* internal_key, const unsigned int& bucket_id) {
			PerscaSkipList<const char*>* Indexlist_ = indexlist_.get();
			const char* indexkey = internal_key;
			Indexlist_->InsertWrappedValue(indexkey, bucket_id);
		}

		bool HashCuckooRep::QuickInsertConcurrently(const char* internal_key, const Slice& user_key,
			int bucket_ids[], const int initial_hash_id) {
			const char* stored_key = nullptr;

			// 일단 Key값에 대해 Hash ID 구함
			for (unsigned int hid = initial_hash_id; hid < hash_function_count_; ++hid) {
				bucket_ids[hid] = GetHash(user_key, hid);
			}

			// 해당 Key가 Hash 에 있는지 검사
			// 만약 2개 이상의 Thread가 검사해서 없다고 판단했다.
			// 그러면 동시에 똑같은 Hash 에 넣으려고 하겠지
			// 이때 compare swap 을 해서 먼저 쓰면 한명은 실패 하겠지
			// 그럼 Retry 를 통해 다시 중복키 검사를 해서 업데이트만 해주면 되고..
			// 만약 서로 다른 키인데 Bucket ID만 하는거면..
			// nullptr 인걸 찾아서 쓰게 해줘야하니까..

			while (true) {
				int cuckoo_bucket_id = -1;
				bool is_key_update = false;
				for (unsigned int hid = initial_hash_id; hid < hash_function_count_; ++hid) {
					stored_key = cuckoo_array_[bucket_ids[hid]].load(std::memory_order_relaxed);
					if (stored_key == nullptr) {
						cuckoo_bucket_id = bucket_ids[hid];
						break;
					}
					else {
						const auto bucket_user_key = UserKey(stored_key);
						if (bucket_user_key.compare(user_key) == 0) {
							cuckoo_bucket_id = bucket_ids[hid];
							is_key_update = true;
							break;
						}
					}
				}

				// 빈공간이나 Key-Update가 아님.
				if (cuckoo_bucket_id == -1) return false;

				if (is_key_update) {
					// 중복키 업데이트 인경우에..
					stored_key = cuckoo_array_[cuckoo_bucket_id].load(std::memory_order_relaxed);
					char* st_key = const_cast<char*>(stored_key);

					// 업데이트 될때까지 반복
					while (cuckoo_array_[cuckoo_bucket_id].compare_exchange_weak
					(st_key, const_cast<char*>(internal_key)) != true);

					// 하고나면 끝
					return true;
				}
				else {
					// 새로 삽입되는 경우
					// nullptr 이여야만함.
					// 시도했는데 다른애가 써버리면..
					// Retry
					char* st_key = nullptr;
					if (cuckoo_array_[cuckoo_bucket_id].compare_exchange_weak(st_key, const_cast<char*>(internal_key))) {
						// 만약 빈공간에 넣는게 성공했으면
						//InsertIndexData(internal_key, cuckoo_bucket_id);
						InsertJob(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
						return true;
					}
					else {
						continue;
					}

				}
			}
		}
	
		bool HashCuckooRep::QuickInsert(const char* internal_key, const Slice& user_key,
			int bucket_ids[], const int initial_hash_id) {
			int cuckoo_bucket_id = -1;

			// Below does the followings:
			// 0. Calculate all possible locations of the input key.
			// 1. Check if there is a bucket having same user_key as the input does.
			// 2. If there exists such bucket, then replace this bucket by the newly
			//    insert data and return.  This step also performs duplication check.
			// 3. If no such bucket exists but exists a vacant bucket, then insert the
			//    input data into it.
			// 4. If step 1 to 3 all fail, then return false.
			for (unsigned int hid = initial_hash_id; hid < hash_function_count_; ++hid) {
				bucket_ids[hid] = GetHash(user_key, hid);
				// since only one PUT is allowed at a time, and this is part of the PUT
				// operation, so we can safely perform relaxed load.
				const char* stored_key =
					cuckoo_array_[bucket_ids[hid]].load(std::memory_order_relaxed);
				if (stored_key == nullptr) {
					if (cuckoo_bucket_id == -1) {
						cuckoo_bucket_id = bucket_ids[hid];
					}
				}
				else {
					const auto bucket_user_key = UserKey(stored_key);
					if (bucket_user_key.compare(user_key) == 0) {
						cuckoo_bucket_id = bucket_ids[hid];
						break;
					}
				}
			}

			if (cuckoo_bucket_id != -1) {
				cuckoo_array_[cuckoo_bucket_id].store(const_cast<char*>(internal_key),
					std::memory_order_release);
				//YUIL
				//InsertIndexData(internal_key, cuckoo_bucket_id);
				InsertJob(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
				return true;
			}

			return false;
		}

		// Perform pre-check and find the shortest cuckoo path.  A cuckoo path
		// is a displacement sequence for inserting the specified input key.
		//
		// @return true if it successfully found a vacant space or cuckoo-path.
		//     If the return value is true but the length of cuckoo_path is zero,
		//     then it indicates that a vacant bucket or an bucket with matched user
		//     key with the input is found, and a quick insertion is done.
		bool HashCuckooRep::FindCuckooPath(const char* internal_key,
			const Slice& user_key, int* cuckoo_path,
			size_t* cuckoo_path_length,
			const int initial_hash_id) {
			int bucket_ids[HashCuckooRepFactory::kMaxHashCount];
			*cuckoo_path_length = 0;

			if (QuickInsert(internal_key, user_key, bucket_ids, initial_hash_id)) {
				return true;
			}
			// If this step is reached, then it means:
			// 1. no vacant bucket in any of the possible locations of the input key.
			// 2. none of the possible locations of the input key has the same user
			//    key as the input `internal_key`.

			// the front and back indices for the step_queue_

			step_buffer_.reset();

			for (unsigned int hid = initial_hash_id; hid < hash_function_count_; ++hid) {
				/// CuckooStep& current_step = step_queue_[front_pos++];
				CuckooStep& current_step = step_buffer_.NextWriteBuffer();
				current_step.bucket_id_ = bucket_ids[hid];
				current_step.prev_step_id_ = CuckooStep::kNullStep;
				current_step.depth_ = 1;
			}

			while (step_buffer_.HasNewWrite()) {
				int step_id = step_buffer_.read_index_;
				const CuckooStep& step = step_buffer_.ReadNext();
				// Since it's a BFS process, then the first step with its depth deeper
				// than the maximum allowed depth indicates all the remaining steps
				// in the step buffer queue will all exceed the maximum depth.
				// Return false immediately indicating we can't find a vacant bucket
				// for the input key before the maximum allowed depth.
				if (step.depth_ >= cuckoo_path_max_depth_) {
					return false;
				}
				// again, we can perform no barrier load safely here as the current
				// thread is the only writer.
				Slice bucket_user_key =
					UserKey(cuckoo_array_[step.bucket_id_].load(std::memory_order_relaxed));
				if (step.prev_step_id_ != CuckooStep::kNullStep) {
					if (bucket_user_key == user_key) {
						// then there is a loop in the current path, stop discovering this path.
						continue;
					}
				}
				// if the current bucket stores at its nth location, then we only consider
				// its mth location where m > n.  This property makes sure that all reads
				// will not miss if we do have data associated to the query key.
				//
				// The n and m in the above statement is the start_hid and hid in the code.
				unsigned int start_hid = hash_function_count_;
				for (unsigned int hid = 0; hid < hash_function_count_; ++hid) {
					bucket_ids[hid] = GetHash(bucket_user_key, hid);
					if (step.bucket_id_ == bucket_ids[hid]) {
						start_hid = hid;
					}
				}
				// must found a bucket which is its current "home".
				assert(start_hid != hash_function_count_);

				// explore all possible next steps from the current step.
				for (unsigned int hid = start_hid + 1; hid < hash_function_count_; ++hid) {
					CuckooStep& next_step = step_buffer_.NextWriteBuffer();
					next_step.bucket_id_ = bucket_ids[hid];
					next_step.prev_step_id_ = step_id;
					next_step.depth_ = step.depth_ + 1;
					// once a vacant bucket is found, trace back all its previous steps
					// to generate a cuckoo path.
					if (cuckoo_array_[next_step.bucket_id_].load(std::memory_order_relaxed) ==
						nullptr) {
						// store the last step in the cuckoo path.  Note that cuckoo_path
						// stores steps in reverse order.  This allows us to move keys along
						// the cuckoo path by storing each key to the new place first before
						// removing it from the old place.  This property ensures reads will
						// not missed due to moving keys along the cuckoo path.
						cuckoo_path[(*cuckoo_path_length)++] = next_step.bucket_id_;
						int depth;
						for (depth = step.depth_; depth > 0 && step_id != CuckooStep::kNullStep;
							depth--) {
							const CuckooStep& prev_step = step_buffer_.steps_[step_id];
							cuckoo_path[(*cuckoo_path_length)++] = prev_step.bucket_id_;
							step_id = prev_step.prev_step_id_;
						}
						assert(depth == 0 && step_id == CuckooStep::kNullStep);
						return true;
					}
					if (step_buffer_.IsFull()) {
						// if true, then it reaches maxinum number of cuckoo search steps.
						return false;
					}
				}
			}

			// tried all possible paths but still not unable to find a cuckoo path
			// which path leads to a vacant bucket.
			return false;
		}

		bool HashCuckooRep::FindCuckooPathConcurrently(const char* internal_key,
			const Slice& user_key, size_t* cuckoo_path_length, 
			int bucket_ids[], int local_cuckoo_path[], const int initial_hash_id) {
			//int bucket_ids[HashCuckooRepFactory::kMaxHashCount];
			*cuckoo_path_length = 0;

			// If this step is reached, then it means:
			// 1. no vacant bucket in any of the possible locations of the input key.
			// 2. none of the possible locations of the input key has the same user
			//    key as the input `internal_key`.

			// the front and back indices for the step_queue_

			//std::this_thread::sleep_for(std::chrono::milliseconds(10));
			//std::lock_guard<std::mutex> guard(cuckoo_path_building_mutex_);

			step_buffer_.reset();

			for (unsigned int hid = initial_hash_id; hid < hash_function_count_; ++hid) {
				/// CuckooStep& current_step = step_queue_[front_pos++];
				CuckooStep& current_step = step_buffer_.NextWriteBuffer();
				current_step.bucket_id_ = bucket_ids[hid];
				current_step.prev_step_id_ = CuckooStep::kNullStep;
				current_step.depth_ = 1;
			}

			while (step_buffer_.HasNewWrite()) {
				int step_id = step_buffer_.read_index_;
				const CuckooStep& step = step_buffer_.ReadNext();
				// Since it's a BFS process, then the first step with its depth deeper
				// than the maximum allowed depth indicates all the remaining steps
				// in the step buffer queue will all exceed the maximum depth.
				// Return false immediately indicating we can't find a vacant bucket
				// for the input key before the maximum allowed depth.
				if (step.depth_ >= cuckoo_path_max_depth_) {
					return false;
				}
				// again, we can perform no barrier load safely here as the current
				// thread is the only writer.
				Slice bucket_user_key =
					UserKey(cuckoo_array_[step.bucket_id_].load(std::memory_order_seq_cst));
				if (step.prev_step_id_ != CuckooStep::kNullStep) {
					if (bucket_user_key == user_key) {
						// then there is a loop in the current path, stop discovering this path.
						continue;
					}
				}
				// if the current bucket stores at its nth location, then we only consider
				// its mth location where m > n.  This property makes sure that all reads
				// will not miss if we do have data associated to the query key.
				//
				// The n and m in the above statement is the start_hid and hid in the code.
				unsigned int start_hid = hash_function_count_;
				for (unsigned int hid = 0; hid < hash_function_count_; ++hid) {
					bucket_ids[hid] = GetHash(bucket_user_key, hid);
					if (step.bucket_id_ == bucket_ids[hid]) {
						start_hid = hid;
					}
				}
				// must found a bucket which is its current "home".
				assert(start_hid != hash_function_count_);

				// explore all possible next steps from the current step.
				for (unsigned int hid = start_hid + 1; hid < hash_function_count_; ++hid) {
					CuckooStep& next_step = step_buffer_.NextWriteBuffer();
					next_step.bucket_id_ = bucket_ids[hid];
					next_step.prev_step_id_ = step_id;
					next_step.depth_ = step.depth_ + 1;
					// once a vacant bucket is found, trace back all its previous steps
					// to generate a cuckoo path.
					if (cuckoo_array_[next_step.bucket_id_].load(std::memory_order_relaxed) ==
						nullptr) {
						// store the last step in the cuckoo path.  Note that cuckoo_path
						// stores steps in reverse order.  This allows us to move keys along
						// the cuckoo path by storing each key to the new place first before
						// removing it from the old place.  This property ensures reads will
						// not missed due to moving keys along the cuckoo path.
						local_cuckoo_path[(*cuckoo_path_length)++] = next_step.bucket_id_;
						int depth;
						for (depth = step.depth_; depth > 0 && step_id != CuckooStep::kNullStep;
							depth--) {
							const CuckooStep& prev_step = step_buffer_.steps_[step_id];
							local_cuckoo_path[(*cuckoo_path_length)++] = prev_step.bucket_id_;
							step_id = prev_step.prev_step_id_;
						}
						assert(depth == 0 && step_id == CuckooStep::kNullStep);
						return true;
					}
					if (step_buffer_.IsFull()) {
						// if true, then it reaches maxinum number of cuckoo search steps.
						return false;
					}
				}
			}

			// tried all possible paths but still not unable to find a cuckoo path
			// which path leads to a vacant bucket.
			return false;
		}

#ifndef YUIL
		HashCuckooRep::Iterator::Iterator(
			PerscaSkipList<const char*>::Iterator* it,
			const KeyComparator& compare, std::atomic<char*>* cuckoo_arr, const unsigned int& bucket_count_)
			: cit_(it),
			cuckoo_array_(cuckoo_arr),
			bucket_count_(bucket_count_),
			compare_(compare) {}
		// Returns true iff the iterator is positioned at a valid node.
		bool HashCuckooRep::Iterator::Valid() const {
			return cit_->Valid();
		}

		// Returns the key at the current position.
		// REQUIRES: Valid()
		const char* HashCuckooRep::Iterator::key() const {
			assert(Valid());
			auto cindex_ = cit_->Value();
			if (cindex_ == static_cast<unsigned int>(bucket_count_)) {
				return cit_->WrappedValue();
			}
			else {
				const char* key = cuckoo_array_[cit_->Value()].load(std::memory_order_relaxed);
				return key;
			}

		}

		// Advances to the next position.
		// REQUIRES: Valid()
		void HashCuckooRep::Iterator::Next() {
			assert(Valid());
			cit_->Next();
			//if (cit_->Valid()) {
			//	const char* key = cuckoo_array_[atoi(cit_->value())].load(std::memory_order_relaxed);
			//	tmpkey_ = key;
			//	std::cout << tmpkey_ << std::endl;
			//}
		}

		// Advances to the previous position.
		// REQUIRES: Valid()
		void HashCuckooRep::Iterator::Prev() {
			assert(Valid());
			cit_->Prev();
		}

		// Advance to the first entry with a key >= target
		void HashCuckooRep::Iterator::Seek(const Slice& user_key,
			const char* memtable_key) {
			cit_->Seek(user_key, memtable_key);
		}

		// Retreat to the last entry with a key <= target
		void HashCuckooRep::Iterator::SeekForPrev(const Slice& user_key,
			const char* memtable_key) {
			assert(false);

		}

		// Position at the first entry in collection.
		// Final state of iterator is Valid() iff collection is not empty.
		void HashCuckooRep::Iterator::SeekToFirst() {
			cit_->SeekToFirst();
		}

		// Position at the last entry in collection.
		// Final state of iterator is Valid() iff collection is not empty.
		void HashCuckooRep::Iterator::SeekToLast() {
			cit_->SeekToLast();
		}
#else
		HashCuckooRep::Iterator::Iterator(
			std::shared_ptr<std::vector<const char*>> bucket,
			const KeyComparator& compare)
			: bucket_(bucket),
			cit_(bucket_->end()),
			compare_(compare),
			sorted_(false) {}

		void HashCuckooRep::Iterator::DoSort() const {
			if (!sorted_) {
				std::sort(bucket_->begin(), bucket_->end(),
					stl_wrappers::Compare(compare_));
				cit_ = bucket_->begin();
				sorted_ = true;
			}
		}

		// Returns true iff the iterator is positioned at a valid node.
		bool HashCuckooRep::Iterator::Valid() const {
			DoSort();
			return cit_ != bucket_->end();
		}

		// Returns the key at the current position.
		// REQUIRES: Valid()
		const char* HashCuckooRep::Iterator::key() const {
			assert(Valid());
			return *cit_;
		}

		// Advances to the next position.
		// REQUIRES: Valid()
		void HashCuckooRep::Iterator::Next() {
			assert(Valid());
			if (cit_ == bucket_->end()) {
				return;
			}
			++cit_;
		}

		// Advances to the previous position.
		// REQUIRES: Valid()
		void HashCuckooRep::Iterator::Prev() {
			assert(Valid());
			if (cit_ == bucket_->begin()) {
				// If you try to go back from the first element, the iterator should be
				// invalidated. So we set it to past-the-end. This means that you can
				// treat the container circularly.
				cit_ = bucket_->end();
			}
			else {
				--cit_;
			}
		}

		// Advance to the first entry with a key >= target
		void HashCuckooRep::Iterator::Seek(const Slice& user_key,
			const char* memtable_key) {
			DoSort();
			// Do binary search to find first value not less than the target
			const char* encoded_key =
				(memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
			cit_ = std::equal_range(bucket_->begin(), bucket_->end(), encoded_key,
				[this](const char* a, const char* b) {
				return compare_(a, b) < 0;
			}).first;
		}

		// Retreat to the last entry with a key <= target
		void HashCuckooRep::Iterator::SeekForPrev(const Slice& user_key,
			const char* memtable_key) {
			assert(false);
		}

		// Position at the first entry in collection.
		// Final state of iterator is Valid() iff collection is not empty.
		void HashCuckooRep::Iterator::SeekToFirst() {
			DoSort();
			cit_ = bucket_->begin();
		}

		// Position at the last entry in collection.
		// Final state of iterator is Valid() iff collection is not empty.
		void HashCuckooRep::Iterator::SeekToLast() {
			DoSort();
			cit_ = bucket_->end();
			if (bucket_->size() != 0) {
				--cit_;
			}
		}
#endif
		
	}  // anom namespace
	
	void BackgroundWorker(HashCuckooRep* cuckoo)
	{
		std::unique_lock<std::mutex> lock(cuckoo->yul_background_worker_mutex);
		//cuckoo->yul_background_worker_cv.wait(lock, [=] {return cuckoo->yul_background_worker_done; });
		HashCuckooRep::IndexJob job;
		while (true) {
			if (cuckoo->GetWorkQueueSize() == 0) {
				//printf("%zu\n", cuckoo->GetWorkQueueSize());
				cuckoo->yul_background_worker_done = true;
				cuckoo->yul_background_worker_done_cv.notify_one();
				cuckoo->yul_background_worker_cv.wait(lock);
				cuckoo->yul_background_worker_done = false;
			}
			while (cuckoo->GetWorkQueueSize() != 0) {
				if (cuckoo->GetJob(job)) {
					// 만약 큐에서 Job 을 성공적으로 꺼냈으면 Insert
					if (job.Type == kIndexJobBucket) {
						cuckoo->InsertIndexData(job.IndexKey(), job.BucketId());
					}
					else if (job.Type == kIndexJobBackup) {
						cuckoo->InsertBackupData(job.IndexKey(), job.BucketId());
					}
				}
			}
		}
	}

	MemTableRep* HashCuckooRepFactory::CreateMemTableRep(
		const MemTableRep::KeyComparator& compare, Allocator* allocator,
		const SliceTransform* transform, Logger* logger) {
		// The estimated average fullness.  The write performance of any close hash
		// degrades as the fullness of the mem-table increases.  Setting kFullness
		// to a value around 0.7 can better avoid write performance degradation while
		// keeping efficient memory usage.
		static const float kFullness = 0.7f;
		size_t pointer_size = sizeof(std::atomic<const char*>);
		assert(write_buffer_size_ >= (average_data_size_ + pointer_size));
		size_t bucket_count =
			static_cast<size_t>(
			(write_buffer_size_ / (average_data_size_ + pointer_size)) / kFullness +
				1);
		unsigned int hash_function_count = hash_function_count_;
		if (hash_function_count < 2) {
			hash_function_count = 2;
		}
		if (hash_function_count > kMaxHashCount) {
			hash_function_count = kMaxHashCount;
		}
		auto c = new HashCuckooRep(compare, allocator, bucket_count,
			hash_function_count,
			static_cast<size_t>(
			(average_data_size_ + pointer_size) / kFullness)
		);
		for (int i = 0; i < c->kDefaultMaxBackgroundWorker; ++i) {
			std::thread *th = new std::thread(BackgroundWorker, c);
			c->Regist_BackgroundWorker(th);
		}
		
		return c;
	}

	MemTableRepFactory* NewHashCuckooRepFactory(size_t write_buffer_size,
		size_t average_data_size,
		unsigned int hash_function_count) {
	
		return new HashCuckooRepFactory(write_buffer_size, average_data_size,
			hash_function_count);
	}


}  // namespace rocksdb
#endif  // ROCKSDB_LITE
#endif

