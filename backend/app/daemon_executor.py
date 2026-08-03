"""Thread-pool executor with daemon workers across supported CPython APIs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures.thread as _thread_pool
from threading import Thread
import weakref


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose workers never block interpreter shutdown."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_callback(
            _reference: object,
            work_queue=self._work_queue,
        ) -> None:
            work_queue.put(None)

        thread_count = len(self._threads)
        if thread_count >= self._max_workers:
            return
        thread_name = (
            f"{self._thread_name_prefix or self}_{thread_count}"
        )
        executor_reference = weakref.ref(self, weakref_callback)
        create_worker_context = getattr(
            self,
            "_create_worker_context",
            None,
        )
        if callable(create_worker_context):
            worker_args = (
                executor_reference,
                create_worker_context(),
                self._work_queue,
            )
        else:
            worker_args = (
                executor_reference,
                self._work_queue,
                getattr(self, "_initializer", None),
                getattr(self, "_initargs", ()),
            )
        worker = Thread(
            name=thread_name,
            target=_thread_pool._worker,
            args=worker_args,
            daemon=True,
        )
        worker.start()
        self._threads.add(worker)


__all__ = ["DaemonThreadPoolExecutor"]
