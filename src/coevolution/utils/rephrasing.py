import re
from typing import List

from loguru import logger

from coevolution.core.interfaces.data import Problem
from coevolution.utils.prompt_manager import PromptManager
from infrastructure.llm_client import LLMClient


class ProblemRephraser:
    """
    Utility class to generate multiple rephrasings of a problem statement using an LLM.
    This can be used to enhance diversity
    """

    def __init__(
        self,
        llm_client: LLMClient,
        n_rephrasings: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.pm = PromptManager()
        self.n = max(1, int(n_rephrasings))

    def generate_rephrasings(self, problem: Problem) -> Problem:
        """Generate up to `n_rephrasings` distinct rephrasings for `problem`.

        Returns a new `Problem` (via `with_rephrasings`) and does not mutate the
        input instance.
        """
        if problem is None:
            return problem

        # Render prompt
        prompt = self.pm.render_prompt(
            "problem_rephrase.j2", original_problem_description=problem.question_content
        )

        results: List[str] = []

        for i in range(self.n):
            try:
                resp = self.llm_client.generate(prompt)
            except Exception as e:
                logger.warning(f"Rephraser call #{i + 1} failed: {e}")
                continue

            if not resp:
                continue

            text = resp.strip()
            # extract the <problem>...</problem> with regex
            match = re.search(r"<problem>(.*?)</problem>", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                logger.debug(
                    f"Rephraser call #{i + 1} produced rephrasing with length {len(text)}"
                )
                logger.trace(f"Rephraser call #{i + 1} produced rephrasing:\n{text}")
                results.append(text)
            else:
                logger.warning(
                    f"Rephraser call #{i + 1} did not contain expected <problem> tags. Response was:\n{text}"
                )
                continue

        # Deduplicate while preserving order
        deduped: List[str] = []
        seen = set()
        for r in results:
            if r not in seen:
                seen.add(r)
                deduped.append(r)

        # Truncate to requested count
        deduped = deduped[: self.n]

        if not deduped:
            # No rephrasings produced; return original problem unchanged
            return problem

        return problem.with_rephrasings(deduped)


__all__ = ["ProblemRephraser"]
