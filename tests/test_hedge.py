"""first_to_start: a slow synthesis gets a second request, and the first audio wins.

Run: python -m unittest discover -s tests   (from the repo root)
"""
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.modules.setdefault("edge_tts", mock.MagicMock())  # no network in these tests

from tts_handler import first_to_start  # noqa: E402


class FakeSynthesis:
    """Starts its audio after `starts_after` seconds (never when None), then ends."""

    made = []

    def __init__(self, starts_after, ends_after=None):
        self.started, self.done = threading.Event(), threading.Event()
        self.cancelled = False
        FakeSynthesis.made.append(self)

        def run():
            if starts_after is not None:
                time.sleep(starts_after)
                self.started.set()
            if ends_after is not None:
                time.sleep(ends_after)
                self.done.set()

        threading.Thread(target=run, daemon=True).start()

    def cancel(self):
        self.cancelled = True


def maker(*specs):
    specs = list(specs)
    FakeSynthesis.made = []
    return lambda: FakeSynthesis(*specs.pop(0))


class FirstToStartTest(unittest.TestCase):
    def test_a_fast_request_is_used_alone(self):
        winner = first_to_start(maker((0.02,)), hedge_after=0.2)
        self.assertEqual(FakeSynthesis.made, [winner])

    def test_a_slow_request_gets_a_second_one_that_can_win(self):
        winner = first_to_start(maker((1.0,), (0.02,)), hedge_after=0.1)
        first, second = FakeSynthesis.made
        self.assertIs(winner, second)
        self.assertTrue(first.cancelled)
        self.assertFalse(second.cancelled)

    def test_the_first_request_still_wins_when_it_starts_first(self):
        winner = first_to_start(maker((0.15,), (1.0,)), hedge_after=0.1)
        first, second = FakeSynthesis.made
        self.assertIs(winner, first)
        self.assertTrue(second.cancelled)

    def test_a_request_that_ends_without_audio_is_retried_at_once(self):
        began = time.monotonic()
        winner = first_to_start(maker((None, 0.01), (0.02,)), hedge_after=5.0)
        self.assertIs(winner, FakeSynthesis.made[1])
        self.assertLess(time.monotonic() - began, 1.0)

    def test_two_requests_without_audio_end_the_wait(self):
        winner = first_to_start(maker((None, 0.01), (None, 0.01)), hedge_after=0.05)
        self.assertIs(winner, FakeSynthesis.made[1])


if __name__ == "__main__":
    unittest.main()
