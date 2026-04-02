"""
utils/metrics.py
All 5 NLP metrics with graceful fallbacks if libraries are not installed.
"""
import re


def _numbers(text: str) -> set:
    return set(re.findall(r'\b\d+\.?\d*\b', text))


def exact_match_score(candidate: str, reference: str) -> float:
    ref_nums = _numbers(reference)
    can_nums = _numbers(candidate)
    if not ref_nums:
        rw = set(reference.lower().split())
        cw = set(candidate.lower().split())
        return round(len(rw & cw) / max(len(rw), 1) * 100, 1)
    matched = ref_nums & can_nums
    return round(len(matched) / len(ref_nums) * 100, 1)


def rouge_score_calc(candidate: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        r = scorer.score(reference, candidate)
        return round(r['rougeL'].fmeasure * 100, 1)
    except ImportError:
        rw = set(reference.lower().split())
        cw = set(candidate.lower().split())
        overlap = rw & cw
        rec  = len(overlap) / max(len(rw), 1)
        prec = len(overlap) / max(len(cw), 1)
        f1   = 2 * rec * prec / max(rec + prec, 1e-8)
        return round(f1 * 100, 1)


def bleu_score_calc(candidate: str, reference: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        ref_tokens = reference.lower().split()
        can_tokens = candidate.lower().split()
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], can_tokens, smoothing_function=smoothie)
        return round(score * 100, 1)
    except ImportError:
        # Fallback: simple unigram overlap
        rw = set(reference.lower().split())
        cw = candidate.lower().split()
        matched = sum(1 for w in cw if w in rw)
        return round(matched / max(len(cw), 1) * 100, 1)


def meteor_score_calc(candidate: str, reference: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        for res in ['wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'corpora/{res}')
            except LookupError:
                nltk.download(res, quiet=True)
        score = meteor_score([reference.lower().split()], candidate.lower().split())
        return round(score * 100, 1)
    except ImportError:
        rw = set(reference.lower().split())
        cw = set(candidate.lower().split())
        return round(len(rw & cw) / max(len(rw), 1) * 100, 1)


def bert_score_calc(candidate: str, reference: str) -> float:
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(
            [candidate],
            [reference],
            lang="en",
            model_type="distilbert-base-uncased",
            device="cpu",
            verbose=False
        )
        return round(F1.mean().item() * 100, 1)
    except Exception:
        # Fallback: Jaccard similarity
        rw = set(reference.lower().split())
        cw = set(candidate.lower().split())
        inter = rw & cw
        union = rw | cw
        return round(len(inter) / max(len(union), 1) * 100, 1)


def run_all_metrics(candidate: str, reference: str) -> dict:
    return {
        "exact":  exact_match_score(candidate, reference),
        "rouge":  rouge_score_calc(candidate, reference),
        "bleu":   bleu_score_calc(candidate, reference),
        "meteor": meteor_score_calc(candidate, reference),
        "bert":   bert_score_calc(candidate, reference),
    }