"""Window24 is optional and changes only the single-step historical bank."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rebuild_manager_states import rebuild_split


class HistoryWindowTests(unittest.TestCase):
    def test_default_fullprefix_and_window_boundaries(self):
        turns = [{"dia_id": f"D1:{i+1}", "speaker": "A", "text": str(i)}
                 for i in range(40)]
        facts = {i: [{"text": f"fact {i}"}] for i in range(40)}
        sample = {"sample_id": "conv-26",
                  "conversation": {"speaker_a": "A", "speaker_b": "B", "session_1": turns},
                  "qa": [{"question": "where?", "answer": "here", "category": 1,
                          "evidence": ["D1:1", f"D1:{i+1}"]} for i in (0, 3, 24, 30)]}
        with patch("scripts.rebuild_manager_states.load_cached_facts", return_value=facts):
            full, timeline, _ = rebuild_split([sample], Path("unused"), "train")
            window, window_timeline, stats = rebuild_split(
                [sample], Path("unused"), "train", history_window=24)
        self.assertEqual(timeline, window_timeline)
        self.assertEqual(len(window_timeline[0]["turns"]), 40)
        self.assertEqual(stats["bindable"], 4)
        for original, limited in zip(full, window):
            target = original["target_turn"]
            self.assertEqual([e["source_turn"] for e in original["old_bank"]], list(range(target)))
            self.assertEqual([e["source_turn"] for e in limited["old_bank"]],
                             list(range(max(0, target-24), target)))
            self.assertEqual(limited["old_bank"], original["old_bank"][-24:])
            for key in original.keys() - {"old_bank", "history_window", "history_start_turn"}:
                self.assertEqual(original[key], limited[key])

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            rebuild_split([], Path("unused"), "train", history_window=0)


if __name__ == "__main__":
    unittest.main()
