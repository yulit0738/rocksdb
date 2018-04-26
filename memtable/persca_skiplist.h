// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_DB_SKIPLIST_H_
#define STORAGE_LEVELDB_DB_SKIPLIST_H_

// Thread safety
// -------------
//
// Writes require external synchronization, most likely a mutex.
// Reads require a guarantee that the PerscaSkipList will not be destroyed
// while the read is in progress.  Apart from that, reads progress
// without any internal locking or synchronization.
//
// Invariants:
//
// (1) Allocated nodes are never deleted until the PerscaSkipList is
// destroyed.  This is trivially guaranteed by the code since we
// never delete any skip list nodes.
//
// (2) The contents of a Node except for the next/prev pointers are
// immutable after the Node has been linked into the PerscaSkipList.
// Only Insert() modifies the list, and it is careful to initialize
// a node and use release-stores to publish the nodes in one or
// more lists.
//
// ... prev vs. next pointer ordering ...

#include <assert.h>
#include <stdlib.h>
#include <atomic>
#include "port/port.h"
// YUIL
#include "util/concurrent_arena.h"
#include "util/random.h"
#include "rocksdb/memtablerep.h"

namespace rocksdb {

	class ConcurrentArena;

	template<typename Key>
	class PerscaSkipList {
	private:
		struct Node;

	public:
		// Create a new PerscaSkipList object that will use "cmp" for comparing keys,
		// and will allocate memory using "*arena".  Objects allocated in the arena
		// must remain allocated for the lifetime of the skiplist object.
		explicit PerscaSkipList(const MemTableRep::KeyComparator& cmp);

		// Insert key into the list.
		// REQUIRES: nothing that compares equal to key is currently in the list.
#ifndef YUIL // Insert() and GettAllSkiplist() Added in skiplist.h
		void InsertWrappedValue(const Key& key, const unsigned int& value);
		const char* GetWrappedValue(const Key & key);
		void Insert(const Key& key, unsigned int val);
		void GetAllSkiplist();
		// Return an iterator over the keys in this representation.
		virtual MemTableRep::Iterator* GetIterator(Arena* arena) {
			char* mem = nullptr;
			if (arena == nullptr) {
				return new Iterator(this);
			}
			else {
				mem = arena->AllocateAligned(sizeof(PerscaSkipList::Iterator));
				return new (mem) Iterator(this);
			}
		}
#else
		void Insert(const Key& key);
#endif

		// Returns true iff an entry that compares equal to key is in the list.
		bool Contains(const Key& key) const;

		// Iteration over the contents of a skip list
		class Iterator : public MemTableRep::Iterator {
		public:
			// Initialize an iterator over the specified list.
			// The returned iterator is not valid.
			explicit Iterator(const PerscaSkipList* list);

			// Returns the value at the current position.
			// REQUIRES: Valid()
#ifndef YUIL // Return Value() in Skiplist Iterator
			unsigned int Value() const;
			const char* WrappedValue() const;

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
#else
			// Initialize an iterator over the specified collection.
			// The returned iterator is not valid.
			// explicit Iterator(const MemTableRep* collection);
			~Iterator() {};

			// Returns true iff the iterator is positioned at a valid node.
			bool Valid() const;

			// Returns the key at the current position.
			// REQUIRES: Valid()
			const char* key() const;
			// Advances to the next position.
			// REQUIRES: Valid()
			void Next();

			// Advances to the previous position.
			// REQUIRES: Valid()
			void Prev();

			// Advance to the first entry with a key >= target
			void Seek(const char* target);

			// Position at the first entry in list.
			// Final state of iterator is Valid() iff list is not empty.
			void SeekToFirst();

			// Position at the last entry in list.
			// Final state of iterator is Valid() iff list is not empty.
			void SeekToLast();
#endif

		private:
			std::string tmp_;  // For passing to EncodeKey
			const PerscaSkipList* list_;
			Node* node_;
			// Intentionally copyable
		};

	private:
		enum { kMaxHeight = 12 };

		// Immutable after construction
		//const MemTableRep::KeyComparator& compare_;
		ConcurrentArena* const arena_;    // Arena used for allocations of nodes

		Node* const head_;

		// Modified only by Insert().  Read racily by readers, but stale
		// values are ok.
		std::atomic<int> max_height_;   // Height of the entire list

										// for comparing
		const MemTableRep::KeyComparator& compare_;

		inline int GetMaxHeight() const {
			return max_height_.load(std::memory_order_relaxed);
		}

		// Read/written only by Insert().
		Random rnd_;
#ifndef YUIL
		Node* NewNode(const Key& key, unsigned int value, int height);
		//std::atomic<size_t*> size_;
#else
		Node* NewNode(const Key& key, int height);
#endif
		int RandomHeight();
		bool Equal(const Key& a, const Key& b) const {
			Slice akey = GetLengthPrefixedSlice(a);
			Slice aakey = Slice(akey.data(), akey.size() - 8);
			Slice bkey = GetLengthPrefixedSlice(b);
			Slice bbkey = Slice(bkey.data(), bkey.size() - 8);
			return aakey.compare(bbkey) == 0;
		}

		// Return true if key is greater than the data stored in "n"
		bool KeyIsAfterNode(const Key& key, Node* n) const;

		// Return the earliest node that comes at or after key.
		// Return NULL if there is no such node.
		//
		// If prev is non-NULL, fills prev[level] with pointer to previous
		// node at "level" for every level in [0..max_height_-1].
#ifndef YUIL // FindGreaterOrEqual() => get exp and prev nodes for backtracking.
		Node* FindGreaterOrEqual(const Key& key, Node** exp, Node** prev) const;
#else
		Node* FindGreaterOrEqual(const Key& key, Node** prev) const;
#endif

		// Return the latest node with a key < key.
		// Return head_ if there is no such node.
		Node* FindLessThan(const Key& key) const;

		// Return the last node in the list.
		// Return head_ if list is empty.
		Node* FindLast() const;

		// No copying allowed
		PerscaSkipList(const PerscaSkipList&);
		void operator=(const PerscaSkipList&);
	};

	// Implementation details follow
	template<typename Key>
	struct PerscaSkipList<Key>::Node {
	public:
		explicit Node(const Key& k, unsigned int v, const Key& wk=nullptr) : key(k), value(v), wrappedkey_(wk){ }
		unsigned int value;
		Key const key;
		std::atomic<Key> wrappedkey_;

		// Accessors/mutators for links.  Wrapped in methods so we can
		// add the appropriate barriers as necessary.
#ifndef YUIL // Node* Next() modified (std::atomic)
		Node* Next(int n) {
			assert(n >= 0);
			// Use an 'acquire load' so that we observe a fully initialized
			// version of the returned Node.
			return next_[n].load(std::memory_order_acquire);
		}
#else
		Node* Next(int n) {
			assert(n >= 0);
			// Use an 'acquire load' so that we observe a fully initialized
			// version of the returned Node.
			return reinterpret_cast<Node*>(next_[n].Acquire_Load());
		}
#endif
#ifndef YUIL // (skiplist.h) update value pointer atomically
		bool SetWrappedValue(const char* expv, const char* newv) {
			return this->wrappedkey_.compare_exchange_weak(expv, newv);
		}

		bool SetNode(Node* expn, Node* newn, int n) {
			return this->next_[n].compare_exchange_weak(expn, newn);
		}
		void StashHeight(const int height) {
			assert(sizeof(int) <= sizeof(next_[0]));

			memcpy(&next_[0], &next_, sizeof(std::atomic<Node*>));
		}
#endif
		void SetNext(int n, Node* x) {
			assert(n >= 0);
			// Use a 'release store' so that anybody who reads through this
			// pointer observes a fully initialized version of the inserted node.
			next_[n].store(x,std::memory_order_release);
		}

		// No-barrier variants that can be safely used in a few locations.
		Node* NoBarrier_Next(int n) {
			assert(n >= 0);
			return next_[n].load(std::memory_order_relaxed);
		}
		void NoBarrier_SetNext(int n, Node* x) {
			assert(n >= 0);
			next_[n].store(x, std::memory_order_relaxed);
		}

	private:
		// Array of length equal to the node height.  next_[0] is lowest level link.
		// YUIL - next_ type is changed port::Atomic to std::atomic<void*>
		std::atomic<Node*> next_[1];
	};

#ifndef YUIL // next_ is allocated by sizeof(std::atomic<void*>) * (height - 1)
	template<typename Key>
	typename PerscaSkipList<Key>::Node*
		PerscaSkipList<Key>::NewNode(const Key& key, unsigned int value, int height) {
		auto prefix = sizeof(std::atomic<Node*>) * (height - 1);
		char* mem = arena_->AllocateAligned(prefix + sizeof(Node));
		Node *x = new (mem) Node(key, value);
		for (int i = 0; i < height; ++i) {
			x->NoBarrier_SetNext(i, nullptr);
		}
		return x;
	}
#else
	template<typename Key>
	typename PerscaSkipList<Key>::Node*
		PerscaSkipList<Key>::NewNode(const Key& key, int height) {
		char* mem = arena_->AllocateAligned(
			sizeof(Node) + sizeof(port::AtomicPointer) * (height - 1));
		return new (mem) Node(key);
	}
#endif

	template<typename Key>
	inline PerscaSkipList<Key>::Iterator::Iterator(const PerscaSkipList* list) {
		list_ = list;
		node_ = NULL;
	}

	template<typename Key>
	inline bool PerscaSkipList<Key>::Iterator::Valid() const {
		return node_ != NULL;
	}

	template<typename Key>
	inline const char* PerscaSkipList<Key>::Iterator::key() const {
		assert(Valid());
		return node_->key;
	}

#ifndef YUIL // return value in Skiplist
	template<typename Key>
	inline unsigned int PerscaSkipList<Key>::Iterator::Value() const {
		assert(Valid());
		return node_->value;
	}
	template<typename Key>
	inline const char* PerscaSkipList<Key>::Iterator::WrappedValue() const {
		assert(Valid());
		return node_->wrappedkey_;
	}
#endif

	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::Next() {
		assert(Valid());
		node_ = node_->Next(0);
	}

	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::Prev() {
		// Instead of using explicit "prev" links, we just search for the
		// last node that falls before key.
		assert(Valid());
		node_ = list_->FindLessThan(node_->key);
		if (node_ == list_->head_) {
			node_ = NULL;
		}
	}

	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::Seek(const Slice& user_key,
		const char* memtable_key) {
		const char* encoded_key =
			(memtable_key != nullptr) ? memtable_key : EncodeKey(&tmp_, user_key);
		node_ = list_->FindGreaterOrEqual(encoded_key, NULL, NULL);
	}
	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::SeekForPrev(const Slice& user_key,
		const char* memtable_key) {
	}

	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::SeekToFirst() {
		node_ = list_->head_->Next(0);
	}

	template<typename Key>
	inline void PerscaSkipList<Key>::Iterator::SeekToLast() {
		node_ = list_->FindLast();
		if (node_ == list_->head_) {
			node_ = NULL;
		}
	}

	template<typename Key>
	int PerscaSkipList<Key>::RandomHeight() {
		// Increase height with probability 1 in kBranching
		static const unsigned int kBranching = 4;
		int height = 1;
		while (height < kMaxHeight && ((rnd_.Next() % kBranching) == 0)) {
			height++;
		}
		assert(height > 0);
		assert(height <= kMaxHeight);
		return height;
	}

#ifndef YUIL // KeyIsAfterNode() If Compare(a,b)==0 return true.
	template<typename Key>
	bool PerscaSkipList<Key>::KeyIsAfterNode(const Key& key, Node* n) const {
		// NULL n is considered infinite
		return (n != NULL) && (n->key != NULL) && (compare_(n->key, key) < 0);
	}
#else
	template<typename Key>
	bool PerscaSkipList<Key>::KeyIsAfterNode(const Key& key, Node* n) const {
		// NULL n is considered infinite
		return (n != NULL) && (compare_(n->key, key) < 0);
	}
#endif
#ifndef YUIL // (skiplist.h) Skiplist::FindGreaterOrEqual => get backtrack node (prev, exp) for multithreaded atomic insert
	template<typename Key>
	typename PerscaSkipList<Key>::Node* PerscaSkipList<Key>::FindGreaterOrEqual(const Key& key, Node** exp, Node** prev)
		const {
		Node* x = head_;
		int level = GetMaxHeight() - 1;
		while (true) {
			Node* next = x->Next(level);
			if (KeyIsAfterNode(key, next)) {
				// Keep searching in this list
				x = next;
			}
			else {
				//if (x->key != NULL && Equal(key, x->key)) return NULL;
				if (prev != NULL) prev[level] = x;
				if (exp != NULL) exp[level] = prev[level]->Next(level);
				//if (x->key != NULL && Equal(key, x->key)) return next;
				if (level == 0) {
					return next;
				}
				else {
					// Switch to next list
					level--;
				}
			}
		}
	}
#else
	template<typename Key>
	typename PerscaSkipList<Key>::Node* PerscaSkipList<Key>::FindGreaterOrEqual(const Key& key, Node** prev)
		const {
		Node* x = head_;
		int level = GetMaxHeight() - 1;
		while (true) {
			Node* next = x->Next(level);
			if (KeyIsAfterNode(key, next)) {
				// Keep searching in this list
				x = next;
			}
			else {
				if (prev != NULL) prev[level] = x;
				if (level == 0) {
					return next;
				}
				else {
					// Switch to next list
					level--;
				}
			}
		}
	}
#endif
	template<typename Key>
	typename PerscaSkipList<Key>::Node*
		PerscaSkipList<Key>::FindLessThan(const Key& key) const {
		Node* x = head_;
		int level = GetMaxHeight() - 1;
		while (true) {
			assert(x == head_ || compare_(x->key, key) < 0);
			Node* next = x->Next(level);
			if (next == NULL || compare_(next->key, key) >= 0) {
				if (level == 0) {
					return x;
				}
				else {
					// Switch to next list
					level--;
				}
			}
			else {
				x = next;
			}
		}
	}

	template<typename Key>
	typename PerscaSkipList<Key>::Node* PerscaSkipList<Key>::FindLast()
		const {
		Node* x = head_;
		int level = GetMaxHeight() - 1;
		while (true) {
			Node* next = x->Next(level);
			if (next == NULL) {
				if (level == 0) {
					return x;
				}
				else {
					// Switch to next list
					level--;
				}
			}
			else {
				x = next;
			}
		}
	}

#ifndef YUIL
	template<typename Key>
	PerscaSkipList<Key>::PerscaSkipList(const MemTableRep::KeyComparator& cmp) 
		: compare_(cmp),
		//size_(memt_size),
		head_(NewNode(nullptr /* any key will do */, 0, kMaxHeight)),
		max_height_(1),
		arena_(new ConcurrentArena()),
		rnd_(0xdeadbeef) {
		for (int i = 0; i < kMaxHeight; i++) {
			head_->SetNext(i, nullptr);
		}
	}
#else
	template<typename Key>
	PerscaSkipList<Key>::PerscaSkipList(Comparator cmp, Arena* arena)
		: compare_(cmp),
		arena_(arena),
		head_(NewNode(0 /* any key will do */, kMaxHeight)),
		max_height_(reinterpret_cast<void*>(1)),
		rnd_(0xdeadbeef) {
		for (int i = 0; i < kMaxHeight; i++) {
			head_->SetNext(i, NULL);
		}
	}
#endif

#ifndef YUIL
	template<typename Key>
	const char* PerscaSkipList<Key>::GetWrappedValue(const Key & key) {
		Node* x = nullptr;
		x = FindGreaterOrEqual(key, nullptr, nullptr);
		if(x != nullptr)
			return x->wrappedkey_.load(std::memory_order_relaxed);
		return nullptr;
	}

	template<typename Key>
	void PerscaSkipList<Key>::InsertWrappedValue(const Key & key, const unsigned int& bucket_id) {
		// Our Skiplist does not allow duplicated keys insertion.
		// If value is NULL then It's a deleted Key. 
		// If duplicated is inserted then just replace value pointer.

		Node* prev[kMaxHeight], *exp[kMaxHeight];			// for Backtracking
		bool done = false;									// for Concurrency
		Node* x = NULL;										// for current position
		while (!done) {

			// :: SEARCH STAGE ::
			// find appropriate key's position and make backtracked list.
			x = FindGreaterOrEqual(key, exp, prev);

			// :: UPDATE STATE (DUPLICATED) ::
			// if key is duplicated then just update value pointer atomically.
			if (x != NULL && Equal(key, x->key)) {
				const char* exp_value;
				do {
					exp_value = x->wrappedkey_.load(std::memory_order_relaxed);
				} while (!x->SetWrappedValue(exp_value, key));
				done = true;
			}
			// :: UPDATE STATE (NEWKEY) ::
			// if key is not exist then make new node and try insert new node in list.
			else {
				int max_height = max_height_.load(std::memory_order_relaxed);
				int height = RandomHeight();
				while (height > max_height) {
					for (int i = GetMaxHeight(); i < height; ++i) {
						prev[i] = head_;
					}
					if (max_height_.compare_exchange_weak(max_height, height)) {
						// successfully updated it
						max_height = height;
						break;
					}
					// else retry, possibly exiting the loop because somebody else
					// increased it
				}
				Node* newObj = NewNode(key, bucket_id, height);

				for (int i = 0; i < height; ++i) {
					newObj->NoBarrier_SetNext(i, prev[i]->NoBarrier_Next(i));
				}

				// if height[0] update is failed then retry.
				if (!(prev[0]->SetNode(exp[0], newObj, 0))) {
					free(newObj);
					continue;
				}

				for (int i = 1; i < height; ++i) {
					if (!(prev[i]->SetNode(exp[i], newObj, i))) {
						break;
					}
				}
			}

			done = true;
		}
	}
	

	template<typename Key>
	void PerscaSkipList<Key>::Insert(const Key& key, unsigned int val) {
		// Our Skiplist does not allow duplicated keys insertion.
		// If value is NULL then It's a deleted Key. 
		// If duplicated is inserted then just replace value pointer.
		//size_t valsize = (size_t)log10(val) + 2;
		//char* value = arena_->AllocateAligned(valsize);
		//value = itoa(val, value, 10);
		//snprintf(value, valsize, "%u", val);
		Node* prev[kMaxHeight], *exp[kMaxHeight];			// for Backtracking
		bool done = false;									// for Concurrency
		Node* x = NULL;										// for current position
		while (!done) {

			// :: SEARCH STAGE ::
			// find appropriate key's position and make backtracked list.
			x = FindGreaterOrEqual(key, exp, prev);
			// :: UPDATE STATE (DUPLICATED) ::
			// if key is duplicated then just update value pointer atomically.
			if (x != NULL && Equal(key, x->key)) {
				if (x->value != val)x->value = val;
				return;
			}
			// :: UPDATE STATE (NEWKEY) ::
			// if key is not exist then make new node and try insert new node in list.
			else {
				int max_height = max_height_.load(std::memory_order_relaxed);
				int height = RandomHeight();
				while (height > max_height) {
					for (int i = GetMaxHeight(); i < height; ++i) {
						prev[i] = head_;
					}
					if (max_height_.compare_exchange_weak(max_height, height)) {
						// successfully updated it
						max_height = height;
						break;
					}
					// else retry, possibly exiting the loop because somebody else
					// increased it
				}

				Node* newObj = NewNode(key, val, height);

				for (int i = 0; i < height; ++i) {
					newObj->NoBarrier_SetNext(i, prev[i]->NoBarrier_Next(i));
				}

				// if height[0] update is failed then retry.
				if (!(prev[0]->SetNode(exp[0], newObj, 0))) {
					//free(newObj);
					continue;
				}

				for (int i = 1; i < height; ++i) {
					if (!(prev[i]->SetNode(exp[i], newObj, i))) {
						break;
					}
				}
			}

			// Memory Usage Update State
			done = true;
		}
	}
#endif
	template<typename Key>
	bool PerscaSkipList<Key>::Contains(const Key& key) const {
		Node* x = FindGreaterOrEqual(key, NULL);
		if (x != NULL && Equal(key, x->key)) {
			return true;
		}
		else {
			return false;
		}
	}

#ifndef YUIL // GetAllSkiplist() defined in skiplist.
	template<typename Key>
	void PerscaSkipList<Key>::GetAllSkiplist() {
		Node* x = head_;
		Node* next = x->Next(0);
		printf("====================================================================================\n");
		while (1) {
			Slice key = GetLengthPrefixedSlice(next->key);
			Slice kkey = Slice(key.data(), key.size() - 8);
			std::string tmp;
			tmp.assign(kkey.data(), kkey.size());
			unsigned int val = next->value;
			printf("KEY : %16s | VALUE : %10u\n", tmp.c_str(), val);
			next = next->Next(0);
			if (next == NULL)break;
		}
		printf("====================================================================================\n");
	}
#endif

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_DB_SKIPLIST_H_


