from lev.scoring import Score, Scorer, ScoringContext


class ContainsStringScorer(Scorer):
    """Static scorer that checks if answer contains a specific string."""

    def __init__(self, target_string: str, case_sensitive: bool = False):
        self.target_string = target_string
        self.case_sensitive = case_sensitive

    async def score(self, ctx: ScoringContext) -> Score:
        """Score based on whether the answer contains the target string."""
        if not ctx.answer:
            return Score(0.0, f"No answer to check for '{self.target_string}'")

        search_text = ctx.answer if self.case_sensitive else ctx.answer.lower()
        target = self.target_string if self.case_sensitive else self.target_string.lower()

        if target in search_text:
            return Score(1.0, f"Found '{self.target_string}' in answer")
        else:
            return Score(0.0, f"'{self.target_string}' not found in answer")


def create_contains_string_scorer(target_string: str, case_sensitive: bool = False, **kwargs) -> ContainsStringScorer:
    """Factory method to create a contains string scorer."""
    return ContainsStringScorer(target_string, case_sensitive)
