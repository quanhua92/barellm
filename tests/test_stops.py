from __future__ import annotations

import time

from barellm.sampling.stops import FINISH_ABORT, FINISH_LENGTH, FINISH_STOP, check_stop

EOS = {151645, 151643}


class TestCheckStop:
    def test_eos_im_end_stops(self) -> None:
        result = check_stop(
            token_id=151645,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result == (FINISH_STOP, 151645)

    def test_eos_endoftext_stops(self) -> None:
        result = check_stop(
            token_id=151643,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result == (FINISH_STOP, 151643)

    def test_max_tokens_stops(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=100,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result == (FINISH_LENGTH, None)

    def test_max_tokens_exceeds_stops(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=101,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result == (FINISH_LENGTH, None)

    def test_stop_string_stops(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello\n\n",
            stop_strings=["\n\n"],
            deadline=None,
        )
        assert result == (FINISH_STOP, "\n\n")

    def test_stop_string_not_present_continues(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello world",
            stop_strings=["\n\n"],
            deadline=None,
        )
        assert result is None

    def test_multiple_stop_strings(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="goodbye",
            stop_strings=["\n\n", "bye"],
            deadline=None,
        )
        assert result == (FINISH_STOP, "bye")

    def test_deadline_stops(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=time.monotonic() - 1,
        )
        assert result == (FINISH_ABORT, None)

    def test_deadline_not_passed_continues(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=time.monotonic() + 60,
        )
        assert result is None

    def test_no_conditions_met_continues(self) -> None:
        result = check_stop(
            token_id=872,
            eos_ids=EOS,
            generated_count=5,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result is None

    def test_eos_checked_before_max_tokens(self) -> None:
        result = check_stop(
            token_id=151645,
            eos_ids=EOS,
            generated_count=100,
            max_new_tokens=100,
            text_so_far="hello",
            stop_strings=None,
            deadline=None,
        )
        assert result == (FINISH_STOP, 151645)
