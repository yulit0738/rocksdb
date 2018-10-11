package org.rocksdb;

/**
 * The config for cuckoo hash mem-table representation.
 * Such mem-table representation contains a fix-sized array of
 * buckets, where each bucket points to a key-value pair (or null if the
 * bucket is empty).
 */
public class HashCuckooMemTableConfig extends MemTableConfig {
  public static final int DEFAULT_WRITE_BUFFER_SIZE = 256 * 1024 * 1024;
  public static final int DEFAULT_AVERAGE_DATA_SIZE = 256;
  public static final int DEFUALT_HASH_FUNCTION_COUNT = 4;

  /**
   * HashCuckooMemTableConfig constructor
   */
  public HashCuckooMemTableConfig() {
    writebuffersize_ = DEFAULT_WRITE_BUFFER_SIZE;
    averagedatasize_ = DEFAULT_AVERAGE_DATA_SIZE;
    hashfunctioncount_ = DEFUALT_HASH_FUNCTION_COUNT;
  }

  /**
   * Set the number of hash buckets used in the hash Cuckoo memtable.
   * Default = 1000000.
   *
   * @param size the number of hash buckets used in the hash
   *    Cuckoo memtable.
   * @return the reference to the current HashCuckooMemTableConfig.
   */
  public HashCuckooMemTableConfig setWriteBufferSize(
      final long size) {
    writebuffersize_ = size;
    return this;
  }

  /**
   * @return the number of hash buckets
   */
  public long writeBufferSize() {
    return writebuffersize_;
  }

  /**
   * Set the height of the skip list.  Default = 4.
   *
   * @param size height to set.
   *
   * @return the reference to the current HashCuckooMemTableConfig.
   */
  public HashCuckooMemTableConfig setAverageDataSize(final long size) {
    averagedatasize_ = size;
    return this;
  }

  /**
   * @return the height of the skip list.
   */
  public long averageDataSize() {
    return averagedatasize_;
  }

  /**
   * Set the branching factor used in the hash skip-list memtable.
   * This factor controls the probabilistic size ratio between adjacent
   * links in the skip list.
   *
   * @param hf the probabilistic size ratio between adjacent link
   *     lists in the skip list.
   * @return the reference to the current HashCuckooMemTableConfig.
   */
  public HashCuckooMemTableConfig setHashFunctionCount(
      final int hf) {
    hashfunctioncount_ = hf;
    return this;
  }

  /**
   * @return branching factor, the probabilistic size ratio between
   *     adjacent links in the skip list.
   */
  public int hashFunctionCount() {
    return hashfunctioncount_;
  }

  @Override protected long newMemTableFactoryHandle() {
    return newMemTableFactoryHandle(
        writebuffersize_, averagedatasize_, hashfunctioncount_);
  }

  private native long newMemTableFactoryHandle(
      long writeBufferSize, long averageDataSize, int hashFunctionCount)
      throws IllegalArgumentException;

  private long writebuffersize_;
  private long averagedatasize_;
  private int hashfunctioncount_;
}

