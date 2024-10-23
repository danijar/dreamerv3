import concurrent.futures


class ThreadPool:

  def __init__(self, workers, name):
    self.pool = concurrent.futures.ThreadPoolExecutor(workers, name)

  def submit(self, fn, *args, **kwargs):
    future = self.pool.submit(fn, *args, **kwargs)
    # Prevent deamon threads from hanging due to exit handlers registered by
    # the concurrent.futures modules.
    concurrent.futures.thread._threads_queues.clear()
    return future

  def close(self, wait=False):
    self.pool.shutdown(wait=wait)
