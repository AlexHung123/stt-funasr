import asyncio
import tempfile
import unittest
from pathlib import Path

from jobs import (
    QUEUE_FULL_MESSAGE,
    RESTART_MESSAGE,
    QueueFull,
    TranscriptionQueue,
)


class TranscriptionQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fourth_waiting_job_is_refused(self) -> None:
        queue = TranscriptionQueue(self.root, max_waiting=3)
        for name in ("a.wav", "b.wav", "c.mp4"):
            queue.begin_upload(name, "yue", True)
        with self.assertRaises(QueueFull) as raised:
            queue.begin_upload("d.wav", "yue", True)
        self.assertEqual(str(raised.exception), QUEUE_FULL_MESSAGE)
        self.assertEqual(queue.waiting_count(), 3)

    def test_cancel_frees_a_slot_and_keeps_speaker_flag(self) -> None:
        queue = TranscriptionQueue(self.root, max_waiting=1)
        job = queue.begin_upload("meeting.mp4", "yue", True)
        self.assertTrue(job.spk)
        queue.mark_queued(job.id)
        self.assertEqual(queue.cancel(job.id), "cancelled")
        self.assertEqual(queue.waiting_count(), 0)
        nxt = queue.begin_upload("next.wav", None, True)
        self.assertEqual(nxt.status, "uploading")

    def test_restart_fails_incomplete_jobs_and_keeps_finished_results(self) -> None:
        queue = TranscriptionQueue(self.root, max_waiting=3)
        unfinished = queue.begin_upload("live.mp4", "yue", True)
        queue.mark_queued(unfinished.id)
        (self.root / unfinished.id / "source.mp4").write_bytes(b"media")

        done = queue.begin_upload("done.wav", "yue", True)
        done.status = "succeeded"
        done.result = {"text": "hello"}
        queue._write_status(done)

        reloaded = TranscriptionQueue(self.root, max_waiting=3)
        reloaded.load_restarted()
        failed = reloaded.get(unfinished.id)
        self.assertIsNotNone(failed)
        assert failed is not None
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error, RESTART_MESSAGE)
        self.assertFalse((self.root / unfinished.id / "source.mp4").exists())
        kept = reloaded.get(done.id)
        self.assertIsNotNone(kept)
        assert kept is not None
        self.assertEqual(kept.status, "succeeded")
        self.assertEqual(kept.result, {"text": "hello"})

    def test_worker_skips_cancelled_and_runs_one_at_a_time(self) -> None:
        asyncio.run(self._worker_scenario())

    async def _worker_scenario(self) -> None:
        queue = TranscriptionQueue(self.root, max_waiting=3)
        first = queue.begin_upload("a.wav", "yue", True)
        queue.mark_queued(first.id)
        second = queue.begin_upload("b.wav", "yue", True)
        queue.mark_queued(second.id)
        self.assertEqual(queue.cancel(first.id), "cancelled")

        started: list[str] = []
        release = asyncio.Event()

        async def runner(job):
            started.append(job.id)
            if len(started) == 1:
                await release.wait()
            return {"text": job.filename}

        task = asyncio.create_task(queue.serve(runner))
        await queue.notify()
        for _ in range(50):
            if started:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(started, [second.id])
        self.assertEqual(queue.get(second.id).status, "running")

        third = queue.begin_upload("c.wav", "yue", True)
        queue.mark_queued(third.id)
        await queue.notify()
        await asyncio.sleep(0.05)
        self.assertEqual(started, [second.id])

        release.set()
        for _ in range(50):
            if len(started) == 2 and queue.get(third.id).status == "succeeded":
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(started, [second.id, third.id])
        self.assertEqual(queue.public_view(third.id)["result"], {"text": "c.wav"})
        self.assertFalse(Path(third.source_path).exists())


if __name__ == "__main__":
    unittest.main()
