"""
End-to-end evaluation harness for LLMService.generate_response(). Runs real
user_input strings through the real bot pipeline (real Gemini calls, real
tool execution against real Riot/AWS APIs) and uses a separate "teacher" LLM
to judge whether the right tools were used and whether the final answer is
actually good.

This hits real, billed APIs -- not meant to run on every commit. Run
directly:
    python -m scuttle_bot.test.e2e_chat_test
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from scuttle_bot.infra.db_client import DatabaseClient
from scuttle_bot.service.llm import LLMService

# A stronger model than the bot's own gemini-2.5-flash, and always a fresh
# instance with no shared state -- it's grading the student, not helping it.
# "-latest" alias avoids hardcoding a specific version that gets deprecated
# out from under this (pinned model names churn fast on this API).
TEACHER_MODEL = "gemini-pro-latest"


class ToolSelectionVerdict(BaseModel):
    correct: bool = Field(description="Whether the agent called the right tool(s) -- neither missing one it clearly needed, nor calling ones the question didn't call for")
    reasoning: str = Field(description="One or two sentences explaining the verdict")


class AnswerQualityVerdict(BaseModel):
    score: int = Field(description="Answer quality from 1 (unusable/wrong) to 5 (accurate, complete, well-formed)", ge=1, le=5)
    reasoning: str = Field(description="One or two sentences explaining the score")


@dataclass
class TestCase:
    name: str
    user_input: str
    expected_tools: list[str]  # tool names expected for a well-behaved run; not necessarily exact
    discord_id: Optional[str] = None
    notes: str = ""  # extra context for the teacher, e.g. why expected_tools is conditional


@dataclass
class TestResult:
    case: TestCase
    tool_calls: list[dict]
    final_answer: str
    tool_verdict: ToolSelectionVerdict
    answer_verdict: AnswerQualityVerdict

    @property
    def passed(self) -> bool:
        return self.tool_verdict.correct and self.answer_verdict.score >= 3


class TeacherJudge:
    """Wraps a separate LLM used purely to grade the bot's behavior."""

    def __init__(self, model_name: str = TEACHER_MODEL):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
        self.tool_judge = self.llm.with_structured_output(ToolSelectionVerdict)
        self.answer_judge = self.llm.with_structured_output(AnswerQualityVerdict)

    def judge_tool_selection(self, case: TestCase, tool_calls: list[dict]) -> ToolSelectionVerdict:
        called = [c["tool"] for c in tool_calls] or ["(none)"]
        prompt = f"""You are grading a League of Legends Discord bot's tool-use decisions.

User asked: {case.user_input!r}

Tools the bot actually called, in order: {called}
Tools expected for a question like this: {case.expected_tools or ['(none needed)']}
{f"Context: {case.notes}" if case.notes else ""}

Judge whether the bot's tool choices were correct for answering this question. It doesn't need to
match the expected list exactly (an equivalent tool is fine, and conditional follow-up tools are
fine to skip if their precondition wasn't met) -- but it should not have skipped a tool it clearly
needed, or called ones the question didn't call for."""
        return self.tool_judge.invoke(prompt)

    def judge_answer_quality(self, case: TestCase, tool_calls: list[dict], final_answer: str) -> AnswerQualityVerdict:
        observations = "\n".join(f"- {c['tool']}: {c['observation']}" for c in tool_calls) or "(no tools were called)"
        prompt = f"""You are grading a League of Legends Discord bot's final answer to a user.

User asked: {case.user_input!r}

Tool results the bot had available:
{observations}

Bot's final answer: {final_answer!r}

Judge the answer's quality: is it accurate given the tool results (or general LoL knowledge if no
tools were needed), does it actually answer what was asked, and is it well-formed for a Discord
chat reply? Score 1 (unusable/wrong) to 5 (accurate, complete, well-formed)."""
        return self.answer_judge.invoke(prompt)


DEFAULT_TEST_CASES = [
    TestCase(
        name="general_knowledge_no_tool",
        user_input="What does the champion Yasuo's passive do?",
        expected_tools=[],
    ),
    TestCase(
        name="summoner_lookup",
        user_input="What are Sorrrymakerrr#DOINB's ranked stats?",
        expected_tools=["search_summoner"],
    ),
    TestCase(
        name="live_game_lookup",
        user_input="Is Sorrrymakerrr#DOINB currently in a game?",
        expected_tools=["get_active_game"],
    ),
    TestCase(
        name="win_probability_direct",
        user_input=(
            "Predict the win probability for this draft (NA, patch 15.1): "
            "Blue: Nwrodh#1156 top Mordekaiser, dubcoww#gbs jungle Ivern, Sorrrymakerrr#DOINB mid Sylas, "
            "P1ll bosby#bill adc Yunara, itsjoever69#420 support Pantheon. "
            "Red: nard#00000 top Jayce, HexSilent#7777 jungle Graves, Cain#boy7 mid Renekton, "
            "Mitsuo Lancuo#wapo adc Ashe, deewgon#NA1 support Seraphine."
        ),
        expected_tools=["predict_win_probability"],
    ),
    TestCase(
        name="live_game_then_predict_chained",
        user_input="Check if Sorrrymakerrr#DOINB is in a game and if so predict who wins.",
        expected_tools=["get_active_game", "predict_win_probability"],
        notes="predict_win_probability is only expected if get_active_game found a live game -- "
              "correctly reporting 'not in a game' after only calling get_active_game is a pass.",
    ),
]


class EndToEndChatTest:
    """
    Drives LLMService.generate_response() through a set of real user inputs
    and grades each run with a teacher LLM on two axes: did it call the
    right tool(s), and is the final answer actually good. Hits real Gemini
    and Riot APIs -- meant to be run deliberately, not on every commit.
    """

    def __init__(self, llm_service: Optional[LLMService] = None, teacher: Optional[TeacherJudge] = None, db_path: str = "src/scuttle_bot/cache/scuttle_bot.db"):
        load_dotenv()
        self.llm_service = llm_service or LLMService(db=DatabaseClient(db_path))
        self.teacher = teacher or TeacherJudge()

    def run_case(self, case: TestCase) -> TestResult:
        print(f"\n=== {case.name} ===")
        print(f"Input: {case.user_input}")

        final_answer = self.llm_service.generate_response(case.user_input, discord_id=case.discord_id)
        tool_calls = self.llm_service.last_tool_calls

        print(f"Tools called: {[c['tool'] for c in tool_calls]}")
        print(f"Answer: {final_answer}")

        tool_verdict = self.teacher.judge_tool_selection(case, tool_calls)
        answer_verdict = self.teacher.judge_answer_quality(case, tool_calls, final_answer)

        print(f"Tool selection: {'PASS' if tool_verdict.correct else 'FAIL'} -- {tool_verdict.reasoning}")
        print(f"Answer quality: {answer_verdict.score}/5 -- {answer_verdict.reasoning}")

        return TestResult(case=case, tool_calls=tool_calls, final_answer=final_answer, tool_verdict=tool_verdict, answer_verdict=answer_verdict)

    def run(self, cases: Optional[list[TestCase]] = None) -> list[TestResult]:
        cases = cases if cases is not None else DEFAULT_TEST_CASES
        results = [self.run_case(case) for case in cases]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.case.name}: tools={'ok' if r.tool_verdict.correct else 'BAD'}, answer={r.answer_verdict.score}/5")

        passed = sum(1 for r in results if r.passed)
        print(f"\n{passed}/{len(results)} passed")

        return results


if __name__ == "__main__":
    EndToEndChatTest().run()
