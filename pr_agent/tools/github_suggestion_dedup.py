import hashlib
import posixpath
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


FINDING_MARKER_RE = re.compile(r"<!--\s*pr-agent-finding:\s*([0-9a-f]{64})\s*-->", re.IGNORECASE)
CODE_FINDING_MARKER_RE = re.compile(r"<!--\s*pr-agent-finding-code:\s*([0-9a-f]{64})\s*-->", re.IGNORECASE)
MALFORMED_FINDING_MARKER_RE = re.compile(r"<!--\s*pr-agent-finding(?:-code)?:", re.IGNORECASE)
DECISION_RE = re.compile(r"^\s*pr-agent:\s*(ignore|accepted-risk|fixed)\s*$", re.IGNORECASE | re.MULTILINE)
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
MAINTAINER_PERMISSIONS = {"admin", "maintain", "write"}

SUGGESTION_BLOCK_RE = re.compile(r"```suggestion[^\n]*\n(.*?)```", re.DOTALL)
DIFF_BLOCK_RE = re.compile(r"```diff[^\n]*\n(.*?)```", re.DOTALL)
DETAILS_BLOCK_RE = re.compile(r"<details>.*?(?:</details>|\Z)", re.DOTALL | re.IGNORECASE)

# A one-line fix can legitimately recur at unrelated locations in the same file, so
# matching on the proposed code alone is only trusted above this length.
SHORT_FIX_CHARS = 60


def normalize_path(path: str) -> str:
    path = str(path or "").strip().strip("`").replace("\\", "/")
    path = posixpath.normpath(path).lstrip("/")
    if path in {"", "."} or path == ".." or path.startswith("../"):
        return ""
    return path


def normalize_text(value: str) -> str:
    value = re.sub(r"<!--.*?-->", " ", str(value or ""), flags=re.DOTALL)
    value = re.sub(r"```(?:suggestion)?|```", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE)
    return " ".join(value.split())[:4000]


def normalize_code(value: str) -> str:
    lines = []
    for line in str(value or "").replace("\r\n", "\n").splitlines():
        line = re.sub(r"^\s*[-+ ]", "", line).strip()
        if line and not line.startswith("@@"):
            lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)[-4000:]


def added_lines_of_diff(value: str) -> str:
    """Reconstruct the proposed code from a rendered ```diff block, dropping removals."""
    kept = []
    for line in str(value or "").replace("\r\n", "\n").splitlines():
        if line.startswith("-"):
            continue
        kept.append(line)
    return normalize_code("\n".join(kept))


def finding_parts(suggestion: dict[str, Any]) -> tuple[str, str, str]:
    path = normalize_path(suggestion.get("relevant_file", ""))
    semantic = normalize_text(" ".join(str(suggestion.get(key, "")) for key in (
        "one_sentence_summary", "label", "suggestion_content", "relevant_symbol", "symbol"
    )))
    context = normalize_code(suggestion.get("existing_code", "") or suggestion.get("nearby_code", ""))
    return path, semantic, context


def legacy_finding_parts(suggestion: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return fields represented in pre-marker inline comments, plus the proposed code."""
    path = normalize_path(suggestion.get("relevant_file", ""))
    semantic = normalize_text(" ".join(str(suggestion.get(key, "")) for key in (
        "suggestion_content", "label", "relevant_symbol", "symbol"
    )))
    context = normalize_code(suggestion.get("existing_code", "") or suggestion.get("nearby_code", ""))
    improved = normalize_code(suggestion.get("improved_code", ""))
    return path, semantic, context, improved


def finding_fingerprint(suggestion: dict[str, Any]) -> str:
    return hashlib.sha256("\x1f".join(finding_parts(suggestion)).encode("utf-8")).hexdigest()


def code_finding_fingerprint(suggestion: dict[str, Any]) -> str | None:
    """Fingerprint the identity of the fix itself: path plus before/after code.

    Model wording drifts between runs at any temperature above zero, which changes
    ``finding_fingerprint``. The code of a finding is far more stable, so this second
    fingerprint is what usually matches an earlier run. Returns None when the
    suggestion carries no code to hash.
    """
    path = normalize_path(suggestion.get("relevant_file", ""))
    existing = normalize_code(suggestion.get("existing_code", "") or suggestion.get("nearby_code", ""))
    improved = normalize_code(suggestion.get("improved_code", ""))
    if not path or not (existing or improved):
        return None
    return hashlib.sha256("\x1f".join((path, existing, improved)).encode("utf-8")).hexdigest()


def marker_for(fingerprint: str) -> str:
    return f"<!-- pr-agent-finding: {fingerprint} -->"


def code_marker_for(fingerprint: str) -> str:
    return f"<!-- pr-agent-finding-code: {fingerprint} -->"


def parse_marker(body: str) -> str | None:
    match = FINDING_MARKER_RE.search(str(body or ""))
    return match.group(1).lower() if match else None


def parse_code_marker(body: str) -> str | None:
    match = CODE_FINDING_MARKER_RE.search(str(body or ""))
    return match.group(1).lower() if match else None


def parse_decision(body: str) -> str | None:
    match = DECISION_RE.search(str(body or ""))
    return match.group(1).lower() if match else None


def is_maintainer(comment: dict[str, Any]) -> bool:
    return (str(comment.get("author_association", "")).upper() in MAINTAINER_ASSOCIATIONS
            or str(comment.get("author_permission", "")).lower() in MAINTAINER_PERMISSIONS)


def preserved_markers(body: str) -> str:
    """Return the finding markers present in a body, ready to re-append.

    Publication fallbacks truncate a comment body at the ```suggestion fence, which
    is where the markers live. Re-appending what this returns keeps the comment
    matchable on later runs.
    """
    markers = []
    for pattern, render in ((FINDING_MARKER_RE, marker_for), (CODE_FINDING_MARKER_RE, code_marker_for)):
        match = pattern.search(str(body or ""))
        if match:
            markers.append(render(match.group(1).lower()))
    return "\n".join(markers)


@dataclass(frozen=True)
class PriorFinding:
    fingerprint: str | None
    code_fingerprint: str | None
    path: str
    semantic: str
    context: str
    proposed_code: str
    resolved: bool
    outdated: bool
    decision: str | None


def _prior_proposed_code(body: str) -> str:
    """Extract the code an earlier comment proposed, from either rendering.

    Committable comments carry a ```suggestion fence. Comments that fell outside a
    hunk are re-rendered by the provider as a collapsed ```diff block, whose removal
    lines must be dropped to recover the proposed code.
    """
    match = SUGGESTION_BLOCK_RE.search(str(body or ""))
    if match:
        return normalize_code(match.group(1))
    match = DIFF_BLOCK_RE.search(str(body or ""))
    if match:
        return added_lines_of_diff(match.group(1))
    return ""


def _prior_semantic(body: str) -> str:
    """Prose of an earlier comment, with both code renderings removed.

    Markers are parsed from the raw body before this runs, so stripping the
    collapsed <details> block here does not lose them.
    """
    body = DETAILS_BLOCK_RE.sub(" ", str(body or ""))
    body = body.split("```suggestion", 1)[0]
    body = re.sub(r"^\s*\*\*Suggestion:\*\*\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r",\s*importance:\s*\d+(?:\.\d+)?", "", body, flags=re.IGNORECASE)
    return normalize_text(body)


def _legacy_parts(comment: dict[str, Any]) -> tuple[str, str, str, str]:
    body = str(comment.get("body", ""))
    return (
        normalize_path(comment.get("path", "")),
        _prior_semantic(body),
        normalize_code(comment.get("diff_hunk", "") or comment.get("nearby_code", "")),
        _prior_proposed_code(body),
    )


def index_prior_findings(history: dict[str, Any], bot_logins: list[str], honor_decisions: bool) -> list[PriorFinding]:
    comments = history.get("comments", [])
    configured_logins = {login.casefold() for login in bot_logins if login}
    authenticated_login = str(history.get("bot_login", "")).casefold()
    recognized_logins = configured_logins or ({authenticated_login} if authenticated_login else set())
    by_root: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        comment_id = comment.get("id")
        root_id = comment.get("in_reply_to_id") or comment_id
        if isinstance(root_id, int):
            by_root.setdefault(root_id, []).append(comment)

    findings = []
    thread_states = history.get("thread_states", {})
    for root_id, thread_comments in by_root.items():
        root = next((comment for comment in thread_comments if comment.get("id") == root_id), None)
        if not root or str(root.get("author_login", "")).casefold() not in recognized_logins:
            continue
        body = str(root.get("body", ""))
        marker = parse_marker(body)
        code_marker = parse_code_marker(body)
        if marker is None and code_marker is None and MALFORMED_FINDING_MARKER_RE.search(body):
            continue
        decision = None
        if honor_decisions:
            for reply in sorted(thread_comments, key=lambda item: item.get("id", 0)):
                if reply is root or str(reply.get("author_login", "")).casefold() in recognized_logins:
                    continue
                if not is_maintainer(reply):
                    continue
                decision = parse_decision(reply.get("body", "")) or decision
        path, semantic, context, proposed = _legacy_parts(root)
        state = thread_states.get(root_id, {})
        findings.append(PriorFinding(marker, code_marker, path, semantic, context, proposed,
                                     bool(state.get("resolved")), bool(state.get("outdated")), decision))
    return findings


def _similarity(left: str, right: str) -> float:
    tokens, prior_tokens = set(left.split()), set(right.split())
    token_ratio = (len(tokens & prior_tokens) / max(len(tokens), len(prior_tokens))) if tokens and prior_tokens else 0.0
    return max(SequenceMatcher(None, left, right, autojunk=False).ratio(), token_ratio)


def _context_compatible(context: str, prior_context: str, threshold: float) -> bool:
    """Whether two code contexts can describe the same site.

    The prior context is a GitHub diff hunk, which is normally much larger than the
    ``existing_code`` of a suggestion, so containment is checked before ratio.
    """
    if not context or not prior_context:
        return True
    if context in prior_context or prior_context in context:
        return True
    return SequenceMatcher(None, context, prior_context, autojunk=False).ratio() >= threshold


def _legacy_equivalent(parts: tuple[str, str, str, str], prior: PriorFinding, threshold: float) -> bool:
    path, semantic, context, _ = parts
    if not path or path != prior.path or not semantic or not prior.semantic:
        return False
    return (_similarity(semantic, prior.semantic) >= threshold
            and _context_compatible(context, prior.context, threshold))


def _code_equivalent(parts: tuple[str, str, str, str], prior: PriorFinding, threshold: float) -> bool:
    """Match on the proposed fix, which survives the model rewording its prose."""
    path, _, context, improved = parts
    if not path or path != prior.path or not improved or not prior.proposed_code:
        return False
    if SequenceMatcher(None, improved, prior.proposed_code, autojunk=False).ratio() < threshold:
        return False
    if len(improved) < SHORT_FIX_CHARS:
        return _context_compatible(context, prior.context, threshold)
    return True


def _code_marker_equivalent(parts: tuple[str, str, str, str], code_fingerprint: str | None,
                            prior: PriorFinding, threshold: float) -> bool:
    """Match the code marker without conflating repeated short fixes in one file."""
    if code_fingerprint is None or prior.code_fingerprint != code_fingerprint:
        return False
    _, _, context, improved = parts
    return len(improved) >= SHORT_FIX_CHARS or _context_compatible(context, prior.context, threshold)


def _decision_equivalent(parts: tuple[str, str, str, str], prior: PriorFinding, threshold: float) -> bool:
    """Match a durable maintainer decision even when nearby code has moved or changed."""
    path, semantic, _, _ = parts
    if not path or path != prior.path or not semantic or not prior.semantic:
        return False
    return _similarity(semantic, prior.semantic) >= threshold


def filter_duplicate_suggestions(suggestions: list[dict[str, Any]], history: dict[str, Any], *,
                                 bot_logins: list[str], include_resolved: bool,
                                 honor_decisions: bool, similarity_threshold: float) -> list[dict[str, Any]]:
    threshold = min(1.0, max(0.0, float(similarity_threshold)))
    prior_findings = index_prior_findings(history, bot_logins, honor_decisions)
    filtered = []
    for suggestion in suggestions:
        fingerprint = finding_fingerprint(suggestion)
        code_fingerprint = code_finding_fingerprint(suggestion)
        legacy_parts = legacy_finding_parts(suggestion)
        suppress = False
        for prior in prior_findings:
            equivalent = (prior.fingerprint == fingerprint
                          or _code_marker_equivalent(legacy_parts, code_fingerprint, prior, threshold)
                          or _legacy_equivalent(legacy_parts, prior, threshold)
                          or _code_equivalent(legacy_parts, prior, threshold))
            durable_decision = prior.decision in {"ignore", "accepted-risk"}
            decision_equivalent = equivalent or (
                durable_decision and _decision_equivalent(legacy_parts, prior, threshold)
            )
            if not equivalent and not decision_equivalent:
                continue
            # Open threads remain authoritative even when GitHub marks their original diff position as outdated.
            # Resolved threads are rechecked by default, while explicit ignore/accepted-risk decisions are durable.
            # A "fixed" decision never suppresses a finding that analysis detects again.
            suppress = durable_decision or (
                equivalent and prior.decision != "fixed" and (include_resolved or not prior.resolved)
            )
            if suppress:
                break
        if not suppress:
            item = dict(suggestion)
            item["finding_fingerprint"] = fingerprint
            if code_fingerprint is not None:
                item["finding_code_fingerprint"] = code_fingerprint
            filtered.append(item)
    return filtered
