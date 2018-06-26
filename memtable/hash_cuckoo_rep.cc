// This is Toss_Cinsp_ConcurrentSkip_Backpointer Version
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
#include "memtable/yul_inlineskiplist.h"
#include "util/yul_util.h"


using namespace moodycamel;

namespace rocksdb {
	namespace {

		// the default maximum size of the cuckoo path searching queue
		static const int kCuckooPathMaxSearchSteps = 100;
		static const unsigned int kIndexJobBucket = 0;
		static const unsigned int kIndexJobBackup = 1;
		// Collision mark
		static const unsigned int kIndexJobUpdate = 2;
		static const unsigned int kQueueThreshhold = 20;
		typedef YulInlineSkipList<const MemTableRep::KeyComparator&> KeyIndex;
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
				yul_background_worker_written_ops(0),
				yul_background_worker_todo_ops(0),
				backup_table_(nullptr),
				yul_background_worker_terminate(false),
				yul_snapshot_count(0),
				// YUIL
				KeyIndex_(compare, allocator),
				is_there_dupliacated_key(false),
				have_arena(false),
				yul_index_array_(nullptr),
				yul_index_skip_array_(nullptr),
				//yul_work_queue_ptok_(yul_work_queue_),
				//yul_work_queue_ctok_(yul_work_queue_),
				yul_background_worker_done(true) {
				char* mem = reinterpret_cast<char*>(
					allocator_->Allocate(sizeof(std::atomic<const char*>) * bucket_count_));
				cuckoo_array_ = new (mem) std::atomic<char*>[bucket_count_];
				/* Shortcut pointer init */
				char* indexmem = reinterpret_cast<char*>(
					yul_arena_.Allocate(sizeof(const char*) * bucket_count_));
				yul_index_array_ = new (indexmem) std::atomic<KeyIndex::Node*>[bucket_count_];
				/* skip pointer init */
				char* skipmem = reinterpret_cast<char*>(
					yul_arena_.Allocate(sizeof(const char*) * bucket_count_));
				yul_index_skip_array_ = new (skipmem) std::atomic<KeyIndex::Node*>[bucket_count_];

				for (unsigned int bid = 0; bid < bucket_count_; ++bid) {
					cuckoo_array_[bid].store(nullptr, std::memory_order_relaxed);
					yul_index_array_[bid].store(nullptr, std::memory_order_relaxed);
					yul_index_skip_array_[bid].store(nullptr, std::memory_order_relaxed);
				}

				cuckoo_path_ = reinterpret_cast<int*>(
					allocator_->Allocate(sizeof(int) * (cuckoo_path_max_depth_ + 1)));
				is_nearly_full_ = false;
				//for (int i = 0; i < kDefaultMaxBackgroundWorker; ++i) {
				//  std::thread th(BackgroundWorker, this);
				//}
				//std::lock_guard<std::mutex> lk(yul_background_worker_mutex);
			}

			// return false, indicating HashCuckooRep does not support merge operator.
			virtual bool IsMergeOperatorSupported() const override { return true; }

			// return false, indicating HashCuckooRep does not support snapshot.
			virtual bool IsSnapshotSupported() const override { return true; }

			// Returns true iff an entry that compares equal to key is in the collection.
			virtual bool Contains(const char* internal_key) const override;

			virtual KeyHandle Allocate(const size_t len, char** buf) override {
				*buf = KeyIndex_.AllocateKey(len);
				return static_cast<KeyHandle>(*buf);
			}

			virtual ~HashCuckooRep() override {
				yul_background_worker_terminate.store(true, std::memory_order_release);
				if (yul_background_worker_done) yul_background_worker_cv.notify_all();
				for (int i = 0; i < kDefaultMaxBackgroundWorker; ++i) {
					yul_background_worker[i]->join();
				}
			}

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
				KeyIndex::Iterator* cit_;
				const KeyComparator& compare_;
				std::atomic<char*>* cuckoo_array_;
				unsigned int bucket_count_;
				std::string tmp_;  // For passing to EncodeKey
				HashCuckooRep* list_;
#endif

			public:
				//YUIL
				//explicit Iterator(std::shared_ptr<std::vector<const char*>> bucket,
				//                  const KeyComparator& compare);
				explicit Iterator(KeyIndex::Iterator* it,
					const KeyComparator& compare,
					std::atomic<char*>* cuckoo_arr,
					const unsigned int& bucket_count_,
					HashCuckooRep* list);

				// Initialize an iterator over the specified collection.
				// The returned iterator is not valid.
				// explicit Iterator(const MemTableRep* collection);
				virtual ~Iterator() override {
					if (list_->yul_snapshot_count.load(std::memory_order_relaxed) >= 1) {
						list_->yul_snapshot_count.fetch_sub(1, std::memory_order_relaxed);
					}
					delete cit_;
				};

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
			public:
				IndexJob(const char* key, const unsigned int& bucket, const int& type) :
					indexkey(key), bucket_id(bucket), Type(type) {}
				IndexJob() {}
				~IndexJob() {}

				const char* indexkey;
				unsigned int bucket_id;
				unsigned int Type;


				// returns index key(internal_key => Object)
				inline const char* IndexKey() const { return indexkey; }
				// returns bucket id about indexkey
				inline unsigned int BucketId() const { return bucket_id; }
			};

		public:
			const MemTableRep::KeyComparator& compare_;
		private:

			// the pointer to Allocator to allocate memory, immutable after construction.
			Allocator* const allocator_;
			// the number of hash bucket in the hash table.
			const size_t bucket_count_;
			// approximate size of each entry
			const size_t approximate_entry_size_;
			// the maxinum depth of the cuckoo path.
			const unsigned int cuckoo_path_max_depth_;
			// the backup MemTableRep to handle the case where cuckoo hash cannot find
			// a vacant bucket for inserting the key of a put request.
			std::shared_ptr<MemTableRep> backup_table_;
			// the current number of hash functions used in the cuckoo hash.
			unsigned int hash_function_count_;
			// the backup MemTableRep to handle the case where cuckoo hash cannot find
			// a vacant bucket for inserting the key of a put request.
			//std::shared_ptr<MemTableRep> backup_table_;
			// the array to store pointers, pointing to the actual data.
			std::atomic<char*>* cuckoo_array_;
			// the array to stroe pointers, pointing to the index data in skiplist.
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
			// the current number of entries in cuckoo_array_ which has been occupied.
			std::atomic<size_t> occupied_count_;
			// YUIL
			// for making index these workers have to do their jobs in work_queue
			// How many workers will be created
			static const unsigned int kDefaultMaxBackgroundWorker = 1;

			// YUIL
			// queue includes all cuckoo indicies.
			// all jobs in queue will be done by background thread
			ConcurrentQueue<IndexJob> yul_work_queue_;
			// For shortcut pointer and back pointer
			Arena yul_arena_;
			//ProducerToken yul_work_queue_ptok_;
			//ConsumerToken yul_work_queue_ctok_;

			// YUIL
			// all jobs in queue will be done by background thread
			std::vector<std::thread*> yul_background_worker;
			//std::thread* yul_background_worker;
			std::atomic<bool> yul_background_worker_terminate;

			// YUIL
			// CV for background workers
			std::condition_variable yul_background_worker_cv;
			std::condition_variable yul_background_worker_done_cv;

			// YUIL
			// Mutex for background workers
			std::mutex yul_background_worker_mutex;
			std::mutex yul_background_worker_done_mutex;

			std::atomic<size_t> yul_background_worker_todo_ops;
			std::atomic<size_t> yul_background_worker_written_ops;

			// YUIL
			// boolean for Wakeup Signal
			bool yul_background_worker_done;
			bool is_there_dupliacated_key;
			bool have_arena;

			// YUIL
			// need to support Snapshot so we prepare some tricky flag for snapshot :)
			std::atomic<short> yul_snapshot_count;
			// for terminating thread
			//std::promise<void> exitSignal;
			//std::future<void> futureObj;

			// the default maximum depth of the cuckoo path.
			static const unsigned int kDefaultCuckooPathMaxDepth = 10;

			CuckooStepBuffer step_buffer_;

			// YUIL
			KeyIndex KeyIndex_;
			std::atomic<KeyIndex::Node*>* yul_index_array_;
			std::atomic<KeyIndex::Node*>* yul_index_skip_array_;

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
			KeyIndex::Node* InsertIndexData(const char* internal_key);
			// Index List 에 Key와 Index를 덮어쓴다. (Append 하지 않음)
			KeyIndex::Node* InsertIndexDataOverwrite(const char* internal_key);
			// BackupTable 대신에 Skiplist 사용해서 바로 넣어준다.
			KeyIndex::Node* InsertBackupData(const char * internal_key);


			// Background Worker는 Queue를 보고 Indexlist를 asynchronous 하게 업데이트 한다.
		public:
			//void BackgroundWorker();
			inline void InsertJob(const IndexJob& job);
			inline void InsertJobConcurrently(const IndexJob& job);
			// Get() 했을때 Bucket에서 miss 나면 Indexlist에서 뒤져줌
			const char* GetFromIndexTable(const LookupKey& key);

			inline bool GetJob(IndexJob& job) {
				return yul_work_queue_.try_dequeue(job);
				//return yul_work_queue_.try_dequeue_non_interleaved(job);
			}

			inline bool GetForegroundJob(IndexJob& job) {
				//return yul_work_queue_.try_dequeue(job);
				return yul_work_queue_.try_dequeue(job);
			}

			inline size_t GetWorkQueueSize() {
				return yul_work_queue_.size_approx();
			}

			struct IndexCompare {
				IndexCompare(const KeyComparator& compare) : compare_(compare) {}
				const  KeyComparator& compare_;
				bool operator()(const IndexJob& x, const IndexJob& y) {
					return compare_(x.indexkey, y.indexkey) < 0;
				}
			};

			void UpdateSkipPointer(KeyIndex::Node* p, const IndexJob& job) {
				while (true) {
					KeyIndex::Node* skipp = yul_index_skip_array_[job.bucket_id].load(std::memory_order_acquire);
					if (skipp == nullptr) {
						/* Skippointer 가 null이면 update */
						/* Skippointer를 갖고있음으로써 duplicate 가 많을때 memory compare 및 memory 연산을 줄일 수 있음.*/
						/* 동시에 skippointer에 대한 update가 일어나는 상황이라면? */
						/* compare 는 한번 더 일어날수 있지만 correctness 에는 지장 없음 */
						if (yul_index_skip_array_[job.bucket_id].compare_exchange_weak(skipp, p, std::memory_order_release))
							break;
						/* CAS failed then continue*/
					}
					else {
						/* 이미 있는값이면 Seq 가 더 낮은걸 업데이트 해야함. */
						auto org = GetSequenceNum(skipp->Key());
						auto upd = GetSequenceNum(p->Key());
						if (org > upd) {
							if (yul_index_skip_array_[job.bucket_id].compare_exchange_weak(skipp, p, std::memory_order_release))
								break;
						}
						else {
							/* 안바꿔도 됨*/
							break;
						}
					}
				}
			}

			void DoForegroundWork(const size_t& thresh) {
				HashCuckooRep::IndexJob job;
				bool done = false;
				while (yul_work_queue_.size_approx() != 0) {
					size_t ops_complete = 0;
					while (GetJob(job)) {
						auto key = job.IndexKey();
						unsigned int bid = job.BucketId();
						KeyIndex::Node* index = nullptr;
						ops_complete++;
						if (job.Type == kIndexJobBucket || kIndexJobBackup) {
							index = InsertIndexData(key);
						}
						//auto snap_count = yul_snapshot_count.load(std::memory_order_acquire);
						//if (snap_count >= 1) {
						//	// Snapshot 켜졌으면 Append 방식으로 작동
						//	if (job.Type == kIndexJobBucket || kIndexJobBackup) {
						//		index = InsertIndexData(key);
						//	}
						//}
						//else if (snap_count == 0) {
						//	// Snapshot 꺼지면 Overwrite모드로 작동
						//	if (job.Type == kIndexJobBucket || job.Type == kIndexJobBackup) {
						//		index = InsertIndexDataOverwrite(key);
						//	}
						//}
						// Index array update
						if (index != nullptr) {
							UpdateSkipPointer(index, job);
							while (true) {
								// Hint는 무조건 최신버전 이여야함
								KeyIndex::Node* hint = yul_index_array_[bid].load(std::memory_order_acquire);
								if (hint != nullptr) {
									uint64_t hkey = GetSequenceNum(hint->Key());
									uint64_t ikey = GetSequenceNum(key);
									if (hkey < ikey) {
										if (yul_index_array_[bid].compare_exchange_weak(hint, index, std::memory_order_release)) {
											break;
										}
										else {
											continue;
										}
									}
									else {
										break;
									}
								}
								else {
									if (yul_index_array_[bid].compare_exchange_weak(hint, index, std::memory_order_release)) {
										break;
									}
									else {
										continue;
									}
								}
							}
							//printf("Make Shortcut ! BucketID : %zd",bid);PrintKey(key);
							//cuckoo->yul_index_array_[bid].store(index,std::memory_order_release);
						}
						if (yul_background_worker_written_ops.load(std::memory_order_relaxed) + ops_complete >= thresh) {
							done = true;
							break;
						}
					}
					yul_background_worker_written_ops.fetch_add(ops_complete, std::memory_order_relaxed);
					if (done) {
						break;
					}
				}
			}


			static inline uint64_t GetSequenceNum(const char* internal_key) {
				Slice akey = GetLengthPrefixedSlice(internal_key);
				const uint64_t anum = DecodeFixed64(akey.data() + akey.size() - 8) >> 8;
				return anum;

			}

			// Returns the pointer to the internal iterator to the buckets where buckets
			// are sorted according to the user specified KeyComparator.  Note that
			// any insert after this function call may affect the sorted nature of
			// the returned iterator.
			virtual MemTableRep::Iterator* GetIterator(Arena* arena) override {
				//if (yul_snapshot_count.load(std::memory_order_relaxed) == 0) {
				//      //매번 Add하게 하면 느려짐
				//      yul_snapshot_count.fetch_add(1, std::memory_order_relaxed);
				//}

				auto todo = yul_background_worker_todo_ops.load(std::memory_order_relaxed);
				auto ops = yul_background_worker_written_ops.load(std::memory_order_relaxed);
				auto queuesize = todo - ops;
				//bool done = false;

				//if (queuesize != 0) {
				//      if (yul_background_worker_done) {
				//              yul_background_worker_cv.notify_all();
				//              // Foreground 가 더느림..
				//              //DoForegroundWork();
				//              //done = true;
				//      }
				//      // 1. Busy loop using "pause" for 1 micro sec
				//      // 2. Else SOMETIMES busy loop using "yield" for 100 micro sec (default)
				//      // 3. Else blocking wait

				//      // On a modern Xeon each loop takes about 7 nanoseconds (most of which
				//      // is the effect of the pause instruction), so 200 iterations is a bit
				//      // more than a microsecond.  This is long enough that waits longer than
				//      // this can amortize the cost of accessing the clock and yielding.
				//      if (yul_background_worker_written_ops.load(std::memory_order_relaxed) < todo) {
				//              yul_background_worker_cv.notify_all();
				//              for (uint32_t tries = 0; tries < 200; ++tries) {
				//                      if (yul_background_worker_written_ops.load(std::memory_order_relaxed) >= todo) {
				//                              done = true;
				//                              break;
				//                      }
				//                      // More slower
				//                      //if (yul_background_worker_done) {
				//                      //      yul_background_worker_cv.notify_all();
				//                      //}
				//                      port::AsmVolatilePause();
				//              }
				//              if (!done) {
				//                      // 여기 까지 왔는데 혹시 background worker 가 자고있으면 한번 더깨운다.
				//                      if (yul_background_worker_done) {
				//                              yul_background_worker_cv.notify_all();
				//                      }
				//                      std::unique_lock<std::mutex> lock(yul_background_worker_done_mutex);
				//                      yul_background_worker_done_cv.wait(lock, [=] { return yul_background_worker_done ||
				//                              (yul_background_worker_written_ops.load(std::memory_order_relaxed) >= todo); });
				//              }
				//      }
				//}
				if (queuesize != 0) {
					if (yul_background_worker_done) {
						yul_background_worker_cv.notify_all();
					}
					DoForegroundWork(todo);
				}

				//KeyIndex::Iterator it(&KeyIndex_);
				auto it = new KeyIndex::Iterator(&KeyIndex_);

				if (arena == nullptr) {
					return new Iterator(it, compare_, cuckoo_array_, static_cast<unsigned int>(bucket_count_), this);
				}
				else {
					auto mem = arena->AllocateAligned(sizeof(Iterator));
					have_arena = true;
					return new (mem) Iterator(it, compare_, cuckoo_array_, static_cast<unsigned int>(bucket_count_), this);
				}
			}
		};
		const char* HashCuckooRep::GetFromIndexTable(const LookupKey& k) {
			auto iter = GetIterator(nullptr);
			iter->Seek(Slice(), k.memtable_key().data());
			if (iter->Valid()) {
				auto key = iter->key();
				delete iter;
				return key;
			}
			return nullptr;
		}

		void HashCuckooRep::Get(const LookupKey& key, void* callback_args,
			bool(*callback_func)(void* arg, const char* entry)) {
			Slice user_key = key.user_key();
			for (unsigned int hid = 0; hid < hash_function_count_; ++hid) {
				const char* bucket =
					cuckoo_array_[GetHash(user_key, hid)].load(std::memory_order_acquire);
				if (bucket != nullptr) {
					Slice bucket_user_key = UserKey(bucket);
					if (user_key == bucket_user_key) {
						Slice mem_key = key.internal_key(); // 이걸사용해야 Seq + meta까지 다 가져옴.
															// For snapshot support we should compare seq_num with foundkey and inputkey
						if (yul_snapshot_count.load(std::memory_order_acquire) == 0 || compare_(bucket, mem_key) >= 0) {
							// mem_key 의 Seq와 비교했을때 Seq가 같거나 크면 됨.
							callback_func(callback_args, bucket);
							//printf("[GETTTT] Hit Cuckoo : "); PrintKey(bucket);
							return;
						}
						else if (yul_snapshot_count.load(std::memory_order_acquire) > 0) {
							// 만약 overwrite 되었으면 IndexSkiplist에서 찾아줘야함.
							bucket = GetFromIndexTable(key);
							if (bucket != nullptr) {
								callback_func(callback_args, bucket);
								//printf("[GETTTT] Hit Skiplist : "); PrintKey(bucket);
								return;
							}
							break;
						}
					}
				}
				else {
					// as Put() always stores at the vacant bucket located by the
					// hash function with the smallest possible id, when we first
					// find a vacant bucket in Get(), that means a miss.
					break;
				}
			}


			MemTableRep* backup_table = backup_table_.get();
			if (backup_table != nullptr) {
				// 백업테이블은 애초에 Append 방식이라서 유지 잘되어있음.
				// Get에서는 Cuckoo 해시에서만 분기시켜주자.
				backup_table->Get(key, callback_args, callback_func);
			}
		}


		inline void HashCuckooRep::InsertJob(const IndexJob& job) {
			static const float kBackgroundworkerThreshhold = 0.15f;
			yul_work_queue_.enqueue(job);
			//yul_background_worker_todo_ops.fetch_add(1, std::memory_order_relaxed);
			yul_background_worker_todo_ops.store(yul_background_worker_todo_ops.load(std::memory_order_relaxed) + 1,
				std::memory_order_relaxed);
			auto queuesize = yul_background_worker_todo_ops.load(std::memory_order_relaxed) - yul_background_worker_written_ops.load(std::memory_order_relaxed);
			if (yul_background_worker_done &&
				queuesize >= kQueueThreshhold) {
				// Background Worker가 자고있으면 일단 다 깨운다~~
				yul_background_worker_cv.notify_all();
			}
		}

		inline void HashCuckooRep::InsertJobConcurrently(const IndexJob& job) {
			yul_work_queue_.enqueue(job);
			yul_background_worker_todo_ops.fetch_add(1, std::memory_order_relaxed);

			auto queuesize = yul_background_worker_todo_ops.load(std::memory_order_relaxed) - yul_background_worker_written_ops.load(std::memory_order_relaxed);
			if (yul_background_worker_done &&
				queuesize >= kQueueThreshhold) {
				// Background Worker가 자고있으면 일단 다 깨운다~~
				yul_background_worker_cv.notify_all();
			}
		}

		void HashCuckooRep::InsertConcurrently(KeyHandle handle) {
			static const float kMaxFullness = 0.90f;

		CUCKOOCOLLISIONMOD:
			auto* key = static_cast<char*>(handle);
			//printf("[CONCURRENT INSERT KEY] "); PrintKey(key);
			int initial_hash_id = 0;
			size_t cuckoo_path_length = 0;
			auto user_key = UserKey(key);
			//if (user_key == Slice("YUL_UNIQUE_GETSNAPSHOT")) {
			//	yul_snapshot_count.fetch_add(1, std::memory_order_relaxed);
			//	return;
			//}
			//else if (user_key == Slice("YUL_UNIQUE_RELSNAPSHOT")) {
			//	auto snapshot_count = yul_snapshot_count.load(std::memory_order_relaxed);
			//	if (snapshot_count > 0) {
			//		// 실제 Snapshot이 존재할때만 지운다.
			//		yul_snapshot_count.fetch_sub(1, std::memory_order_relaxed);
			//	}
			//	return;
			//}
			int bucket_ids[HashCuckooRepFactory::kMaxHashCount];
			yul_background_worker_written_ops.load(std::memory_order_relaxed);
			if (QuickInsertConcurrently(key, user_key, bucket_ids, initial_hash_id) == false) {
				// Hash 에 바로 넣기 실패했으면 Cuckoo Path 해줘야함.
				int local_cuckoo_path_[kCuckooPathMaxSearchSteps] = { 0, };
				cuckoo_path_building_mutex_.lock();
				if (FindCuckooPathConcurrently(key, user_key, &cuckoo_path_length, bucket_ids,
					local_cuckoo_path_, initial_hash_id) == false) {
					// 만약 Path 생성도 실패하고 빈 공간도 없으면
					// 그냥 Skiplist에 통째로 넣는다. (BackupTable)
					if (backup_table_.get() == nullptr) {
						VectorRepFactory factory(10);
						backup_table_.reset(
							factory.CreateMemTableRep(compare_, allocator_, nullptr, nullptr));
						is_nearly_full_ = true;
					}
					backup_table_->Insert(key);
					cuckoo_path_building_mutex_.unlock();
					//InsertBackupData(key, static_cast<unsigned int>(bucket_count_));
					InsertJobConcurrently(IndexJob(key, static_cast<unsigned int>(bucket_count_), kIndexJobBackup));
					is_nearly_full_ = true;
					return;
				}
				else {
					// Path 생성에 성공했다면..
					if (cuckoo_path_length == 0) return;
					char* indexkey = key;
					// 마지막 Path 만 Compare And Swap 해보고 안되면 Path 다시 찾게 하면됨
					int kicked_out_bid = local_cuckoo_path_[0];
					int current_bid = local_cuckoo_path_[1];
					yul_index_array_[current_bid].store(yul_index_array_[kicked_out_bid], std::memory_order_release);
					indexkey = cuckoo_array_[current_bid].load(std::memory_order_relaxed);
					char* st_key = cuckoo_array_[kicked_out_bid].load(std::memory_order_relaxed);
					if (st_key == nullptr && cuckoo_array_[kicked_out_bid].compare_exchange_weak(st_key, indexkey)) {
						//InsertJobConcurrently(IndexJob(indexkey, static_cast<unsigned int>(kicked_out_bid), kIndexJobUpdate));
						for (size_t i = 2; i < cuckoo_path_length; ++i) {
							kicked_out_bid = local_cuckoo_path_[i - 1];
							current_bid = local_cuckoo_path_[i];
							// since we only allow one writer at a time, it is safe to do relaxed read.
							indexkey = cuckoo_array_[current_bid].load(std::memory_order_relaxed);
							cuckoo_array_[kicked_out_bid]
								.store(indexkey, std::memory_order_release);
							// 쫒겨난 키들에 대해 Index update
							//InsertIndexData(indexkey, kicked_out_bid);
							//InsertJobConcurrently(IndexJob(indexkey, static_cast<unsigned int>(kicked_out_bid), kIndexJobUpdate));
							yul_index_array_[kicked_out_bid].store(yul_index_array_[current_bid].load(std::memory_order_relaxed), std::memory_order_release);
							yul_index_skip_array_[kicked_out_bid].store(yul_index_skip_array_[current_bid].load(std::memory_order_relaxed), std::memory_order_release);
						}
						int insert_key_bid = local_cuckoo_path_[cuckoo_path_length - 1];
						cuckoo_array_[insert_key_bid].store(key, std::memory_order_release);
						cuckoo_path_building_mutex_.unlock();
						//InsertIndexData(indexkey, insert_key_bid);
						InsertJobConcurrently(IndexJob(key, static_cast<unsigned int>(insert_key_bid), kIndexJobBucket));
					} // Swap 실패하면
					else {
						cuckoo_path_building_mutex_.unlock();
						goto CUCKOOCOLLISIONMOD;
					}
				}
			}
			// find cuckoo path

			// when reaching this point, means the insert can be done successfully.
			occupied_count_.fetch_add(1, std::memory_order_seq_cst); // occupied_count_++;
			if (occupied_count_.load(std::memory_order_relaxed) >= bucket_count_ * kMaxFullness) {
				is_nearly_full_ = true;
			}
		}

		void HashCuckooRep::Insert(KeyHandle handle) {
			static const float kMaxFullness = 0.90f;
			auto* key = static_cast<char*>(handle);
			//printf("[INSERT KEY] "); PrintKey(key);
			int initial_hash_id = 0;
			size_t cuckoo_path_length = 0;
			auto user_key = UserKey(key);
			//if (user_key == Slice("YUL_UNIQUE_GETSNAPSHOT")) {
			//	yul_snapshot_count.store(
			//		yul_snapshot_count.load(std::memory_order_relaxed) + 1,
			//		std::memory_order_relaxed);
			//	return;
			//}
			//else if (user_key == Slice("YUL_UNIQUE_RELSNAPSHOT")) {
			//	auto snapshot_count = yul_snapshot_count.load(std::memory_order_relaxed);
			//	if (snapshot_count > 0) {
			//		// 실제 Snapshot이 존재할때만 지운다.
			//		yul_snapshot_count.store(
			//			yul_snapshot_count.load(std::memory_order_relaxed) - 1,
			//			std::memory_order_relaxed);
			//	}
			//	return;
			//}
			// find cuckoo path
			if (FindCuckooPath(key, user_key, cuckoo_path_, &cuckoo_path_length,
				initial_hash_id) == false) {
				// if true, then we can't find a vacant bucket for this key even we
				// have used up all the hash functions.  Then use a backup memtable to
				// store such key, which will further make this mem-table become
				// immutable.
				// 만들기 실패하면 Skiplist에 통째로 넣자.
				if (backup_table_.get() == nullptr) {
					VectorRepFactory factory(10);
					backup_table_.reset(
						factory.CreateMemTableRep(compare_, allocator_, nullptr, nullptr));
					is_nearly_full_ = true;
				}
				backup_table_->Insert(key);
				//InsertBackupData(key, static_cast<unsigned int>(bucket_count_));
				InsertJobConcurrently(IndexJob(key, static_cast<unsigned int>(bucket_count_), kIndexJobBackup));
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
				//InsertIndexData(indexkey, kicked_out_bid);
				//InsertJobConcurrently(IndexJob(indexkey, static_cast<unsigned int>(kicked_out_bid), kIndexJobUpdate));
				yul_index_array_[kicked_out_bid].store(yul_index_array_[current_bid].load(std::memory_order_relaxed), std::memory_order_release);
				yul_index_skip_array_[kicked_out_bid].store(yul_index_skip_array_[current_bid].load(std::memory_order_relaxed), std::memory_order_release);

			}
			int insert_key_bid = cuckoo_path_[cuckoo_path_length - 1];
			cuckoo_array_[insert_key_bid].store(key, std::memory_order_release);
			//InsertIndexData(indexkey, insert_key_bid);
			InsertJobConcurrently(IndexJob(key, static_cast<unsigned int>(insert_key_bid), kIndexJobBucket));
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

		inline KeyIndex::Node* HashCuckooRep::InsertIndexData(const char* internal_key) {
			return KeyIndex_.InsertConcurrently(internal_key);
		}

		inline KeyIndex::Node* HashCuckooRep::InsertIndexDataOverwrite(const char* internal_key) {
			return KeyIndex_.InsertConcurrently(internal_key);
			//return KeyIndex_.InsertIndex(internal_key, bucket_id);
		}

		inline KeyIndex::Node* HashCuckooRep::InsertBackupData(const char* internal_key) {
			return KeyIndex_.InsertConcurrently(internal_key);
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
							is_there_dupliacated_key = true;
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
					while (true) {
						KeyIndex::Node* hint = yul_index_array_[cuckoo_bucket_id];
						if (hint != nullptr && yul_snapshot_count.load(std::memory_order_relaxed) == 0) {
							uint64_t hkey = GetSequenceNum(hint->Key());
							uint64_t ikey = GetSequenceNum(internal_key);
							if (ikey > hkey) {
								//hint->key = internal_key;
								//hint->UpdateKey(internal_key);
								InsertJob(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
							}
							return true;
						}

						//if (yul_snapshot_count.load(std::memory_order_relaxed) >= 1) {
						InsertJobConcurrently(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
						//}
						// 하고나면 끝
						return true;
					}
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
						InsertJobConcurrently(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
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
			bool is_key_update = false;
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
						is_key_update = true;
						is_there_dupliacated_key = true;
						cuckoo_bucket_id = bucket_ids[hid];
						break;
					}
				}
			}

			if (cuckoo_bucket_id != -1) {
				cuckoo_array_[cuckoo_bucket_id].store(const_cast<char*>(internal_key),
					std::memory_order_release);

				while (true) {
					KeyIndex::Node* hint = yul_index_array_[cuckoo_bucket_id];
					if (hint != nullptr && yul_snapshot_count.load(std::memory_order_relaxed) == 0) {
						uint64_t hkey = GetSequenceNum(hint->Key());
						uint64_t ikey = GetSequenceNum(internal_key);
						if (ikey > hkey) {
							//hint->key = internal_key;
							//if (!hint->UpdateKeyConcurrent(internal_key)) continue;
							InsertJob(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
						}
						return true;
					}
					InsertJob(IndexJob(internal_key, static_cast<unsigned int>(cuckoo_bucket_id), kIndexJobBucket));
					return true;
				}
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
			KeyIndex::Iterator* it, const KeyComparator& compare,
			std::atomic<char*>* cuckoo_arr, const unsigned int& bucket_count_,
			HashCuckooRep* list)
			: cit_(it),
			cuckoo_array_(cuckoo_arr),
			bucket_count_(bucket_count_),
			compare_(compare),
			list_(list) {

		}
		// Returns true iff the iterator is positioned at a valid node.
		bool HashCuckooRep::Iterator::Valid() const {
			return cit_->Valid();
		}

		// Returns the key at the current position.
		// REQUIRES: Valid()
		const char* HashCuckooRep::Iterator::key() const {
			assert(Valid());
			return cit_->key();
		}

		// Advances to the next position.
		// REQUIRES: Valid()
		void HashCuckooRep::Iterator::Next() {
			assert(Valid());
			if (list_->have_arena || !list_->is_there_dupliacated_key) {
				cit_->Next();
			}
			else if (cit_->Valid() && list_->is_there_dupliacated_key) {
				Slice obj = GetLengthPrefixedSlice(cit_->key());
				Slice ukey = Slice(obj.data(), obj.size() - 8);
				KeyIndex::Node* sp = nullptr;
				for (unsigned int hid = 0; hid < list_->hash_function_count_; ++hid) {
					auto HashId = list_->GetHash(ukey, hid);
					const char* bucket =
						cuckoo_array_[HashId].load(std::memory_order_acquire);
					if (bucket != nullptr) {
						Slice bucket_user_key = list_->UserKey(bucket);
						if (ukey == bucket_user_key) {
							sp = list_->yul_index_skip_array_[HashId].load(std::memory_order_acquire);
							break;
						}
					}
				}
				if (sp != nullptr) {
					cit_->SetNode(sp);
				}
				cit_->Next();
			}
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
			//const char* encoded_key =
			//      (memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
			//cit_->Seek(encoded_key);
			Slice ukey;
			static int count = 0;
			const char* encoded_key =
				(memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
			if (user_key.empty()) {
				if (memtable_key == nullptr) {
					cit_->Invalidate();
				}
				else {
					Slice obj = GetLengthPrefixedSlice(memtable_key);
					ukey = Slice(obj.data(), obj.size() - 8);
				}
			}
			else {
				ukey = Slice(user_key.data(), user_key.size() - 8);
			}
			for (unsigned int hid = 0; hid < list_->hash_function_count_; ++hid) {
				auto HashId = list_->GetHash(ukey, hid);
				const char* bucket =
					cuckoo_array_[HashId].load(std::memory_order_acquire);
				if (bucket != nullptr) {
					Slice bucket_user_key = list_->UserKey(bucket);
					if (ukey == bucket_user_key) {
						auto hint = list_->yul_index_array_[HashId].load(std::memory_order_acquire);
						if (hint != nullptr) {
							cit_->Seek(encoded_key, hint);

						}
						else {
							//printf("Hint Can not seek : %d / TODO : %zd / WRITEEN : %zd / bucket_count : %zd\n", ++count, list_->yul_background_worker_todo_ops.load(std::memory_order_relaxed),
							//      list_->yul_background_worker_written_ops.load(std::memory_order_relaxed), list_->bucket_count_);
							//printf("Target KEY : "); PrintKey(encoded_key);
							//printf("Bucket : "); PrintKey(bucket);
							//if (hint != nullptr) { printf("Hint Found : "); PrintKey(hint->key); }
							//else { printf("Hint Not Found!!\n"); }
							//cit_->Seek(encoded_key);
							//if (cit_->Valid()) {
							//      printf("Seek found : "); PrintKey(cit_->key());
							//}
							//else { printf("Seek Not Found!!\n"); }
							//
							//printf("\n");
							cit_->Invalidate();
						}
						//printf("[BACKEND SEEK]n");
						//printf("Target KEY : "); PrintKey(encoded_key);
						//printf("Bucket : "); PrintKey(bucket);
						//if (hint != nullptr) { printf("Hint Found : "); PrintKey(hint->key); }
						//else { printf("Hint Not Found!!\n"); }
						//if (cit_->Valid()) {
						//      printf("Seek found : "); PrintKey(cit_->key());
						//}
						//else { printf("Seek Not Found!!\n"); }
						//
						//printf("\n");
						return;
					}
				}
			}
			// Cuckoo 해시에 없으면 Backuptable도 뒤져야함. 없는 Key에 대한 검색이 올경우 어떻게 해야되지?
			// Cuckoo 해시 밖이라서 Shortcut 접근은 불가능.
			// 일단 별수 없다. Backuptable 이 있는지 검사하고 Skiplist를 전부 뒤지는 수밖에..
			if (list_->backup_table_.get() != nullptr) {
				cit_->Seek(encoded_key);
				return;
			}
			// 못찾으면 유효하지 않음
			//printf("Can not seek : %d\n", ++count);
			cit_->Invalidate();
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
		std::mutex yul_background_worker;
		std::unique_lock<std::mutex> lock(yul_background_worker);
		HashCuckooRep::IndexJob job;
		while (true) {
			auto queuesize = cuckoo->yul_background_worker_todo_ops.load(std::memory_order_relaxed)
				- cuckoo->yul_background_worker_written_ops.load(std::memory_order_relaxed);
			size_t ops_complete = 0;
			if (cuckoo->yul_background_worker_terminate.load(std::memory_order_acquire)) {
				cuckoo->yul_background_worker_done = true;
				//cuckoo->yul_background_worker_done_cv.notify_all();
				return;
			}

			if (queuesize == 0) {
				/*printf("TODO OPS : %zu WRITTEN OPS : %zu \n",
				cuckoo->yul_background_worker_todo_ops.load(std::memory_order_relaxed),
				cuckoo->yul_background_worker_written_ops.load(std::memory_order_relaxed));*/
				cuckoo->yul_background_worker_done = true;
				cuckoo->yul_background_worker_done_cv.notify_all();
				cuckoo->yul_background_worker_cv.wait(lock);
				cuckoo->yul_background_worker_done = false;
			}

			if (cuckoo->yul_background_worker_terminate.load(std::memory_order_acquire)) {
				cuckoo->yul_background_worker_done = true;
				//cuckoo->yul_background_worker_done_cv.notify_all();
				return;
			}

			while (cuckoo->yul_work_queue_.size_approx() != 0) {
				while (cuckoo->GetJob(job)) {
					auto key = job.IndexKey();
					unsigned int bid = job.BucketId();
					KeyIndex::Node* index = nullptr;
					ops_complete++;
					if (job.Type == kIndexJobBucket || kIndexJobBackup) {
						index = cuckoo->InsertIndexData(key);
					}
					//auto snap_count = cuckoo->yul_snapshot_count.load(std::memory_order_relaxed);
					//if (snap_count >= 1) {
					//	// Snapshot 켜졌으면 Append 방식으로 작동
					//	if (job.Type == kIndexJobBucket || kIndexJobBackup) {
					//		index = cuckoo->InsertIndexData(key);
					//	}
					//}
					//else if (snap_count == 0) {
					//	// Snapshot 꺼지면 Overwrite모드로 작동
					//	if (job.Type == kIndexJobBucket || job.Type == kIndexJobBackup) {
					//		index = cuckoo->InsertIndexDataOverwrite(key);
					//	}
					//}
					// Index array update
					if (index != nullptr) {
						cuckoo->UpdateSkipPointer(index, job);
						while (true) {
							// Hint는 무조건 최신버전 이여야함
							KeyIndex::Node* hint = cuckoo->yul_index_array_[bid].load(std::memory_order_acquire);
							if (hint != nullptr) {
								uint64_t hkey = cuckoo->GetSequenceNum(hint->Key());
								uint64_t ikey = cuckoo->GetSequenceNum(key);
								if (hkey < ikey) {
									if (cuckoo->yul_index_array_[bid].compare_exchange_weak(hint, index, std::memory_order_release)) {
										break;
									}
									else {
										continue;
									}
								}
								else {
									break;
								}
							}
							else {
								if (cuckoo->yul_index_array_[bid].compare_exchange_weak(hint, index, std::memory_order_release)) {
									break;
								}
								else {
									continue;
								}
							}
						}
						//printf("Make Shortcut ! BucketID : %zd",bid);PrintKey(key);
						//cuckoo->yul_index_array_[bid].store(index,std::memory_order_release);
					}
				}
			}

			cuckoo->yul_background_worker_written_ops.fetch_add(ops_complete, std::memory_order_relaxed);
		}
	}

	//void BackgroundWorker(HashCuckooRep* cuckoo)
	//{
	//      std::unique_lock<std::mutex> lock(cuckoo->yul_background_worker_mutex);
	//      while (true) {
	//              auto queuesize = cuckoo->yul_background_worker_todo_ops.load(std::memory_order_relaxed)
	//                      - cuckoo->yul_background_worker_written_ops.load(std::memory_order_relaxed);
	//              if (cuckoo->yul_background_worker_terminate) {
	//                      cuckoo->yul_background_worker_done = true;
	//                      //cuckoo->yul_background_worker_done_cv.notify_all();
	//                      break;
	//              }

	//              if (queuesize == 0) {
	//                      /*printf("TODO OPS : %zu WRITTEN OPS : %zu \n",
	//                      cuckoo->yul_background_worker_todo_ops.load(std::memory_order_relaxed),
	//                      cuckoo->yul_background_worker_written_ops.load(std::memory_order_relaxed));*/
	//                      cuckoo->yul_background_worker_done = true;
	//                      cuckoo->yul_background_worker_done_cv.notify_all();
	//                      cuckoo->yul_background_worker_cv.wait(lock);
	//                      cuckoo->yul_background_worker_done = false;
	//              }

	//              //size_t queuesize = cuckoo->GetWorkQueueSize();
	//              HashCuckooRep::IndexJob jobs[100];
	//              if (queuesize > 100) queuesize = 100;
	//              size_t jobsize = cuckoo->yul_work_queue_.try_dequeue_bulk(jobs, queuesize);

	//              // Sorting jobs.
	//              if (jobsize >= 1) {
	//                      for (size_t i = 0; i < jobsize; ++i) {
	//                              //printf("Fore : "); PrintKey(jobs[i].indexkey);
	//                              const HashCuckooRep::IndexJob& job = jobs[i];
	//                              auto key = job.IndexKey();
	//                              unsigned int bid = job.BucketId();
	//                              KeyIndex::Node* index = nullptr;
	//                              auto snap_count = cuckoo->yul_snapshot_count.load(std::memory_order_relaxed);
	//                              if (snap_count >= 1) {
	//                                      // Snapshot 켜졌으면 Append 방식으로 작동
	//                                      if (job.Type == kIndexJobBucket || job.Type == kIndexJobBackup) {
	//                                              index = cuckoo->InsertIndexData(key, bid);
	//                                      }
	//                                      else if (job.Type == kIndexJobUpdate) {
	//                                              index = cuckoo->InsertIndexDataUpdate(key, bid);
	//                                      }
	//                              }
	//                              else if (snap_count == 0) {
	//                                      // Snapshot 꺼지면 Overwrite모드로 작동
	//                                      if (job.Type == kIndexJobBucket || job.Type == kIndexJobBackup) {
	//                                              index = cuckoo->InsertIndexDataOverwrite(key, bid);
	//                                      }
	//                                      else if (job.Type == kIndexJobUpdate) {
	//                                              index = cuckoo->InsertIndexDataUpdate(key, bid);
	//                                      }
	//                              }
	//                              // Index array update
	//                              if (index != nullptr) {
	//                                      KeyIndex::Node* hint = cuckoo->yul_index_array_[bid];
	//                                      if (hint != nullptr) {
	//                                              uint64_t hkey = HashCuckooRep::GetSequenceNum(hint->key);
	//                                              uint64_t ikey = HashCuckooRep::GetSequenceNum(key);
	//                                              if (hkey < ikey) {
	//                                                      cuckoo->yul_index_array_[bid] = index;
	//                                              }
	//                                      }
	//                                      else {
	//                                              cuckoo->yul_index_array_[bid] = index;
	//                                      }
	//                                      //printf("Make Shortcut ! BucketID : %zd",bid);PrintKey(key);
	//                                      //cuckoo->yul_index_array_[bid].store(index,std::memory_order_release);
	//                              }
	//                      }
	//                      cuckoo->yul_background_worker_written_ops.fetch_add(jobsize, std::memory_order_relaxed);
	//              }
	//      }
	//}

	void BackgroundWorkerCaller(HashCuckooRep* cuckoo)
	{
		while (true) {
			// 2 초마다 깨워준다.
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			if (cuckoo->yul_background_worker_done) {
				cuckoo->yul_background_worker_done_cv.notify_all();
				cuckoo->yul_background_worker_cv.notify_all();
			}
		}
	}

	inline void PrintKey(const char* ikey)
	{
		Slice key = GetLengthPrefixedSlice(ikey);
		Slice kkey = Slice(key.data(), key.size() - 8);
		const uint64_t anum = DecodeFixed64(key.data() + key.size() - 8) >> 8;
		std::string tmp;
		tmp.assign(kkey.data(), kkey.size());
		//unsigned int val = next->value;
		printf("KEY : %16s | SEQ : %zu | VALUE : %s\n", tmp.c_str(), anum, tmp.c_str());
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
			c->yul_background_worker.push_back(new std::thread(BackgroundWorker, c));
			//new std::thread(BackgroundWorkerCaller, c);
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

