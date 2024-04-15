import contextlib
import threading


class RWLock:

  def __init__(self):
    self.lock = threading.Lock()
    self.active_writer_lock = threading.Lock()
    self.writer_count = 0
    self.waiting_reader_count = 0
    self.active_reader_count = 0
    self.readers_finished_cond = threading.Condition(self.lock)
    self.writers_finished_cond = threading.Condition(self.lock)

  @property
  @contextlib.contextmanager
  def reading(self):
    try:
      self.acquire_read()
      yield
    finally:
      self.release_read()

  @property
  @contextlib.contextmanager
  def writing(self):
    try:
      self.acquire_write()
      yield
    finally:
      self.release_write()

  def acquire_read(self):
    with self.lock:
      if self.writer_count:
        self.waiting_reader_count += 1
        while self.writer_count:
          self.writers_finished_cond.wait()
        self.waiting_reader_count -= 1
      self.active_reader_count += 1

  def release_read(self):
    with self.lock:
      assert self.active_reader_count > 0
      self.active_reader_count -= 1
      if not self.active_reader_count and self.writer_count:
        self.readers_finished_cond.notify_all()

  def acquire_write(self):
    with self.lock:
      self.writer_count += 1
      while self.active_reader_count:
        self.readers_finished_cond.wait()
    self.active_writer_lock.acquire()

  def release_write(self):
    self.active_writer_lock.release()
    with self.lock:
      assert self.writer_count > 0
      self.writer_count -= 1
      if not self.writer_count and self.waiting_reader_count:
        self.writers_finished_cond.notify_all()
