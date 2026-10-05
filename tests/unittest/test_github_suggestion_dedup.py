import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.tools.github_suggestion_dedup import (code_finding_fingerprint,
                                                    code_marker_for,
                                                    filter_duplicate_suggestions,
                                                    finding_fingerprint,
                                                    marker_for,
                                                    preserved_markers)


def suggestion(**overrides):
    item = {
        "one_sentence_summary": "Avoid duplicated work",
        "label": "maintainability",
        "suggestion_content": "Use the shared helper.",
        "relevant_file": "src/app.py",
        "existing_code": "return old()",
        "improved_code": "return shared()",
    }
    item.update(overrides)
    return item


def comment(item, *, comment_id=1, body=None, path=None, diff_hunk=None, author="pr-agent[bot]",
            reply_to=None, association="NONE"):
    return {
        "id": comment_id,
        "in_reply_to_id": reply_to,
        "body": body if body is not None else marker_for(finding_fingerprint(item)),
        "path": path if path is not None else item["relevant_file"],
        "diff_hunk": diff_hunk if diff_hunk is not None else item["existing_code"],
        "author_login": author,
        "author_association": association,
    }


def rendered_comment_body(item):
    return (f"**Suggestion:** {item['suggestion_content']} [{item['label']}]\n"
            f"```suggestion\n{item['improved_code']}\n```\n\n{marker_for(finding_fingerprint(item))}")


def apply_filter(items, comments, *, states=None, **options):
    history = {"comments": comments, "thread_states": states or {}, "bot_login": "pr-agent[bot]"}
    defaults = {"bot_logins": [], "include_resolved": False, "honor_decisions": True,
                "similarity_threshold": 0.82}
    defaults.update(options)
    return filter_duplicate_suggestions(items, history, **defaults)


def test_exact_fingerprint_suppresses_unresolved_and_moved_line():
    item = suggestion(relevant_lines_start=42)
    assert apply_filter([item], [comment(item)]) == []


def test_same_message_in_different_files_is_not_suppressed():
    old = suggestion(relevant_file="src/other.py")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_paths_that_differ_only_by_case_are_distinct():
    old = suggestion(relevant_file="src/App.py")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_distinct_findings_in_same_file_are_not_suppressed():
    old = suggestion(suggestion_content="Close the socket.", existing_code="socket.open()")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_resolved_finding_obeys_configuration():
    item = suggestion()
    states = {1: {"resolved": True, "outdated": True}}
    assert len(apply_filter([item], [comment(item)], states=states)) == 1
    assert apply_filter([item], [comment(item)], states=states, include_resolved=True) == []


def test_outdated_unresolved_finding_remains_suppressed():
    item = suggestion()
    states = {1: {"resolved": False, "outdated": True}}
    assert apply_filter([item], [comment(item)], states=states) == []
    assert apply_filter([item], [comment(item)], states=states, include_resolved=False) == []


def test_marked_finding_uses_similarity_fallback_when_model_rephrases_it():
    old = suggestion(suggestion_content="Use the shared helper.")
    new = suggestion(suggestion_content="Reuse the shared helper.")
    assert finding_fingerprint(old) != finding_fingerprint(new)
    assert apply_filter([new], [comment(old, body=rendered_comment_body(old))]) == []


@pytest.mark.parametrize("command", ["ignore", "accepted-risk"])
def test_maintainer_decisions_are_case_insensitive_and_allow_whitespace(command):
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="maintainer", association="MEMBER",
                    body=f"  PR-Agent: {command.upper()}  ")
    assert apply_filter([item], [comment(item), reply], include_resolved=False) == []


def test_fixed_allows_reintroduced_finding():
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="maintainer", association="OWNER",
                    body="pr-agent: fixed")
    assert len(apply_filter([item], [comment(item), reply])) == 1


def test_accepted_risk_survives_nearby_code_changes():
    old = suggestion(existing_code="return old()")
    new = suggestion(existing_code="value = prepare()\nreturn old(value, strict=True)")
    reply = comment(old, comment_id=2, reply_to=1, author="maintainer", association="OWNER",
                    body="pr-agent: accepted-risk")
    assert finding_fingerprint(old) != finding_fingerprint(new)
    assert apply_filter([new], [comment(old, body=rendered_comment_body(old)), reply]) == []


@pytest.mark.parametrize("permission", ["write", "maintain", "admin"])
def test_private_member_decision_is_honored_by_repository_permission(permission):
    # GITHUB_TOKEN reports private org members as CONTRIBUTOR.
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="private-member", association="CONTRIBUTOR",
                    body="pr-agent: accepted-risk")
    reply["author_permission"] = permission
    assert apply_filter([item], [comment(item), reply], states={1: {"resolved": True}}) == []


@pytest.mark.parametrize("permission", [None, "", "read", "none"])
def test_decision_without_write_permission_is_ignored(permission):
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="reader", association="CONTRIBUTOR",
                    body="pr-agent: ignore")
    if permission is not None:
        reply["author_permission"] = permission
    assert len(apply_filter([item], [comment(item), reply], states={1: {"resolved": True}})) == 1


def test_bot_authored_command_is_ignored():
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, body="pr-agent: ignore")
    assert len(apply_filter([item], [comment(item), reply], states={1: {"resolved": True}},
                            include_resolved=False)) == 1


def test_material_code_change_changes_fingerprint_and_is_not_suppressed():
    old = suggestion(existing_code="return old()")
    new = suggestion(existing_code="return old(value, strict=True)")
    assert len(apply_filter([new], [comment(old)])) == 1


def test_legacy_comment_without_marker_is_matched_without_generated_summary():
    item = suggestion(one_sentence_summary="A summary that was never rendered in the old comment")
    legacy_body = "**Suggestion:** Use the shared helper. [maintainability]\n```suggestion\nreturn shared()\n```"
    assert apply_filter([item], [comment(item, body=legacy_body)]) == []


def test_legacy_comment_matches_code_inside_larger_diff_hunk_at_default_threshold():
    item = suggestion(one_sentence_summary="Different generated summary")
    legacy_body = ("**Suggestion:** Use the shared helper. [maintainability, importance: 8]\n"
                   "```suggestion\nreturn shared()\n```")
    diff_hunk = "@@ -10,3 +10,3 @@ def run():\n context()\n-return old()\n+return shared()"
    assert apply_filter([item], [comment(item, body=legacy_body, diff_hunk=diff_hunk)],
                        similarity_threshold=0.88) == []


def test_malformed_marker_is_ignored():
    item = suggestion()
    malformed = comment(item, body="<!-- pr-agent-finding: not-a-hash -->")
    assert len(apply_filter([item], [malformed])) == 1


def test_configured_bot_logins_restrict_roots():
    item = suggestion()
    assert len(apply_filter([item], [comment(item)], bot_logins=["another-bot"])) == 1


class PaginatedComments:
    def __init__(self, comments):
        self.comments = comments

    def __iter__(self):
        yield from self.comments


def _raw_comment(comment_id):
    raw = {"id": comment_id, "body": "body", "path": "src/app.py", "diff_hunk": "return old()",
           "user": {"login": "pr-agent[bot]"}, "author_association": "NONE"}
    return SimpleNamespace(id=comment_id, raw_data=raw)


def test_provider_retrieves_paginated_rest_comments_and_graphql_threads():
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 7
    provider.github_user_id = "pr-agent[bot]"
    provider.pr = SimpleNamespace(get_comments=lambda: PaginatedComments([_raw_comment(1), _raw_comment(2)]))
    responses = [
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [{"isResolved": False, "isOutdated": False,
                       "comments": {"nodes": [{"databaseId": 1}]}}],
            "pageInfo": {"hasNextPage": True, "endCursor": "next"}}}}}},
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [{"isResolved": True, "isOutdated": True,
                       "comments": {"nodes": [{"databaseId": 2}]}}],
            "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}},
    ]
    requester = MagicMock()
    requester.requestJson.side_effect = [(200, {}, json.dumps(response)) for response in responses]
    provider.github_client = SimpleNamespace(_Github__requester=requester)

    history = provider.get_code_suggestion_history()

    assert [item["id"] for item in history["comments"]] == [1, 2]
    assert history["thread_states"][2] == {"resolved": True, "outdated": True}
    assert requester.requestJson.call_count == 2


def test_provider_graphql_error_is_raised_for_caller_fail_open():
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 7
    provider.github_user_id = "pr-agent[bot]"
    provider.pr = SimpleNamespace(get_comments=lambda: [])
    requester = MagicMock()
    requester.requestJson.return_value = (200, {}, json.dumps({"errors": [{"message": "denied"}]}))
    provider.github_client = SimpleNamespace(_Github__requester=requester)
    with pytest.raises(RuntimeError):
        provider.get_code_suggestion_history()


def test_provider_rest_error_is_raised_for_caller_fail_open():
    provider = GithubProvider.__new__(GithubProvider)
    provider.pr = SimpleNamespace(get_comments=MagicMock(side_effect=RuntimeError("REST unavailable")))
    with pytest.raises(RuntimeError):
        provider.get_code_suggestion_history()


def code_marked_comment(item, **overrides):
    return comment(item, body=code_marker_for(code_finding_fingerprint(item)), **overrides)


def test_code_fingerprint_survives_a_completely_rewritten_message():
    old = suggestion()
    new = suggestion(one_sentence_summary="Route the call through the shared abstraction",
                     suggestion_content="This duplicates logic that the shared helper already owns.")
    assert finding_fingerprint(old) != finding_fingerprint(new)
    assert code_finding_fingerprint(old) == code_finding_fingerprint(new)
    assert apply_filter([new], [code_marked_comment(old)]) == []


def test_code_fingerprint_is_absent_without_code_to_hash():
    assert code_finding_fingerprint(suggestion(existing_code="", improved_code="")) is None
    assert code_finding_fingerprint(suggestion(relevant_file="")) is None


def test_filter_exposes_both_fingerprints_for_marker_rendering():
    published = apply_filter([suggestion()], [])[0]
    assert published["finding_fingerprint"] == finding_fingerprint(suggestion())
    assert published["finding_code_fingerprint"] == code_finding_fingerprint(suggestion())


def test_suggestion_without_code_still_publishes_a_prose_fingerprint_only():
    item = suggestion(existing_code="", improved_code="")
    published = apply_filter([item], [])[0]
    assert "finding_code_fingerprint" not in published


LONG_FIX = ("if not current_user.has_account_access(account_id):\n"
            "    raise Forbidden('account scope mismatch')")


def test_substantial_fix_matches_on_code_when_prose_diverges():
    old = suggestion(improved_code=LONG_FIX, suggestion_content="Scope the query to the account.")
    new = suggestion(improved_code=LONG_FIX, suggestion_content="Reject requests for other accounts.",
                     existing_code="return Ledger.query.get(ledger_id)")
    body = f"**Suggestion:** {old['suggestion_content']} [bug]\n```suggestion\n{LONG_FIX}\n```"
    assert apply_filter([new], [comment(old, body=body)]) == []


def test_short_fix_alone_does_not_suppress_an_unrelated_location():
    old = suggestion()
    new = suggestion(one_sentence_summary="Unrelated cache defect",
                     suggestion_content="Scope the cache key to the tenant.",
                     existing_code="cache.set(key, value)")
    body = f"**Suggestion:** {old['suggestion_content']} [bug]\n```suggestion\n{old['improved_code']}\n```"
    prior = comment(old, body=body, diff_hunk="@@ -1,2 +1,2 @@ def run():\n-return old()\n+return shared()")
    assert len(apply_filter([new], [prior])) == 1


def test_short_code_marker_does_not_suppress_an_unrelated_location():
    old = suggestion()
    new = suggestion(one_sentence_summary="Unrelated cache defect",
                     suggestion_content="Scope the cache key to the tenant.",
                     existing_code="cache.set(key, value)")
    prior = code_marked_comment(
        old,
        diff_hunk="@@ -1,2 +1,2 @@ def run():\n-return old()\n+return shared()",
    )
    assert len(apply_filter([new], [prior])) == 1


def test_transformed_out_of_hunk_comment_is_matched_through_its_diff_block():
    old = suggestion(improved_code=LONG_FIX)
    new = suggestion(improved_code=LONG_FIX, suggestion_content="Enforce account scoping here.",
                     existing_code="return Ledger.query.get(ledger_id)")
    transformed = ("**Suggestion:** Scope the query. [bug, importance: 9]\n\n"
                   "<details><summary>New proposed code:</summary>\n\n```diff\n"
                   "-return Ledger.query.get(ledger_id)\n"
                   + "\n".join(f"+{line}" for line in LONG_FIX.splitlines())
                   + "\n```\n\n</details>")
    assert apply_filter([new], [comment(old, body=transformed)]) == []


def test_details_block_does_not_hide_a_marker_from_the_prose_fingerprint():
    item = suggestion()
    body = (f"**Suggestion:** anything [bug]\n\n<details><summary>New proposed code:</summary>\n\n"
            f"```diff\n+return shared()\n```\n\n{marker_for(finding_fingerprint(item))}\n\n</details>")
    assert apply_filter([item], [comment(item, body=body)]) == []


def test_malformed_code_marker_is_ignored():
    item = suggestion()
    malformed = comment(item, body="<!-- pr-agent-finding-code: not-a-hash -->")
    assert len(apply_filter([item], [malformed])) == 1


def test_preserved_markers_round_trip_both_marker_kinds():
    item = suggestion()
    prose, code = finding_fingerprint(item), code_finding_fingerprint(item)
    body = f"**Suggestion:** x\n```suggestion\ny\n```\n\n{marker_for(prose)}\n{code_marker_for(code)}"
    assert preserved_markers(body) == f"{marker_for(prose)}\n{code_marker_for(code)}"
    assert preserved_markers("no markers here") == ""


def test_truncating_fallback_keeps_markers_matchable():
    item = suggestion()
    provider = GithubProvider.__new__(GithubProvider)
    body = (f"**Suggestion:** Use the shared helper. [maintainability]\n"
            f"```suggestion\nreturn shared()\n```\n\n"
            f"{marker_for(finding_fingerprint(item))}\n{code_marker_for(code_finding_fingerprint(item))}")
    fixed = provider._try_fix_invalid_inline_comments([{"body": body, "start_line": 3, "start_side": "RIGHT"}])
    assert "```suggestion" not in fixed[0]["body"]
    assert apply_filter([item], [comment(item, body=fixed[0]["body"])]) == []


def _decision_provider(raw_comments, repo_obj):
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 7
    provider.github_user_id = "pr-agent[bot]"
    provider.repo_obj = repo_obj
    provider.pr = SimpleNamespace(get_comments=lambda: PaginatedComments(raw_comments))
    response = {"data": {"repository": {"pullRequest": {"reviewThreads": {
        "nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
    requester = MagicMock()
    requester.requestJson.return_value = (200, {}, json.dumps(response))
    provider.github_client = SimpleNamespace(_Github__requester=requester)
    return provider


def _raw_reply(comment_id, login, body):
    raw = {"id": comment_id, "in_reply_to_id": 1, "body": body, "path": "src/app.py",
           "diff_hunk": "return old()", "user": {"login": login}, "author_association": "CONTRIBUTOR"}
    return SimpleNamespace(id=comment_id, raw_data=raw)


def test_provider_adds_permission_only_for_decision_replies_once_per_author():
    repo_obj = MagicMock(full_name="owner/repo")
    repo_obj.get_collaborator_permission.return_value = "write"
    provider = _decision_provider([_raw_comment(1),
                                   _raw_reply(2, "member", "pr-agent: ignore"),
                                   _raw_reply(3, "member", "pr-agent: accepted-risk"),
                                   _raw_reply(4, "other", "Will fix later.")], repo_obj)

    comments = provider.get_code_suggestion_history()["comments"]

    assert [item.get("author_permission") for item in comments] == [None, "write", "write", None]
    repo_obj.get_collaborator_permission.assert_called_once_with("member")


def test_provider_permission_lookup_error_falls_back_to_association():
    repo_obj = MagicMock(full_name="owner/repo")
    repo_obj.get_collaborator_permission.side_effect = RuntimeError("forbidden")
    provider = _decision_provider([_raw_comment(1), _raw_reply(2, "member", "pr-agent: ignore")], repo_obj)

    comments = provider.get_code_suggestion_history()["comments"]

    assert comments[1]["author_permission"] == ""
