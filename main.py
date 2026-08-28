import copy
import hashlib
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

DAG = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}

INPUT_NAMES = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

NODE_DEPENDENCIES = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig"],
    "train": [
        "prepareArtifact",
        "trainCode",
        "trainConfig",
        "runtime",
    ],
    "evaluate": [
        "trainArtifact",
        "canonicalData",
        "evaluateCode",
        "evaluateConfig",
    ],
    "register": [
        "evaluateArtifact",
        "schemaDigest",
    ],
    "publish": [
        "registerArtifact",
        "publishConfig",
    ],
}

VALID_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

EVENT_FIELDS = [
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
]

LOCK = threading.RLock()

# Global process state.
# Each session has completely isolated state.
SESSIONS = {}


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_utf8(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def content_key(values):
    # Exact requirement:
    # lowercase SHA-256 over UTF-8 compact JSON arrays.
    return sha256_utf8(compact_json(values))


def safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_INT_MAX
    )


def non_empty_string(value):
    return isinstance(value, str) and len(value) > 0


def error_response(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


def empty_session():
    return {
        "revision": None,
        "inputs": None,

        # eventId -> canonical compact JSON event
        "event_canonical": {},

        # eventId -> event object
        "event_data": {},

        # node -> current state
        #
        # State:
        # None
        # {
        #   "status": "...",
        #   "attempt": n,
        #   "key": "...",
        #   "eventId": "..."
        #   "artifactDigest": "..."   # only success
        # }
        "state": {
            node: None
            for node in DAG
        },

        # node -> key -> immutable evidence
        #
        # {
        #   "artifactDigest": "...",
        #   "eventId": "..."
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


def valid_request_shape(body):
    if not isinstance(body, dict):
        return False

    if set(body.keys()) != {
        "session",
        "revision",
        "inputs",
        "events",
    }:
        return False

    if not non_empty_string(body["session"]):
        return False

    if not safe_positive_integer(body["revision"]):
        return False

    if not isinstance(body["inputs"], dict):
        return False

    if not isinstance(body["events"], list):
        return False

    for name in INPUT_NAMES:
        if not non_empty_string(body["inputs"].get(name)):
            return False

    return True


def valid_event_shape(event):
    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not non_empty_string(event["eventId"]):
        return False

    if not safe_positive_integer(event["revision"]):
        return False

    if not non_empty_string(event["node"]):
        return False

    if not safe_positive_integer(event["attempt"]):
        return False

    if event["status"] not in VALID_STATUSES:
        return False

    if not non_empty_string(event["key"]):
        return False

    status = event["status"]

    if status == "succeeded":
        if not non_empty_string(event["artifactDigest"]):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register/publish success requires exact receipt.
    if (
        event["node"] in {"register", "publish"}
        and status == "succeeded"
    ):
        expected = (
            f"receipt:{event['node']}:{event['key']}"
        )
        if event["receiptId"] != expected:
            return False
    else:
        if event["receiptId"] is not None:
            return False

    return True


def canonical_event(event):
    # Canonical event JSON uses exactly the required eight fields
    # in the specified order.
    ordered = {
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest": event["artifactDigest"],
        "receiptId": event["receiptId"],
    }

    return compact_json(ordered)


def input_fingerprint(inputs):
    # Includes ALL metadata, including extra metadata.
    # Thus any same-revision input change conflicts.
    return compact_json(inputs)


def get_cache(session_state, node, key):
    return session_state["cache"][node].get(key)


def get_artifact(session_state, node, key):
    entry = get_cache(session_state, node, key)

    if entry is None:
        return None

    return entry["artifactDigest"]


def compute_keys(inputs, artifacts):
    keys = {}

    # verify_data
    keys["verify_data"] = content_key([
        inputs["generation"],
        inputs["checksum"],
    ])

    # prepare
    if artifacts.get("verify_data") is None:
        keys["prepare"] = None
    else:
        keys["prepare"] = content_key([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    # train
    if artifacts.get("prepare") is None:
        keys["train"] = None
    else:
        keys["train"] = content_key([
            artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    # evaluate
    if artifacts.get("train") is None:
        keys["evaluate"] = None
    else:
        keys["evaluate"] = content_key([
            artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    # register
    if artifacts.get("evaluate") is None:
        keys["register"] = None
    else:
        keys["register"] = content_key([
            artifacts["evaluate"],
            inputs["schemaDigest"],
        ])

    # publish
    if artifacts.get("register") is None:
        keys["publish"] = None
    else:
        keys["publish"] = content_key([
            artifacts["register"],
            inputs["publishConfig"],
        ])

    return keys


def reusable_artifacts(session_state, inputs):
    """
    Walk the DAG and recover artifacts exclusively from immutable
    content-addressed cache entries.

    A node becomes reusable only when its parent artifact is available
    and the current key matches a successful cached entry.
    """
    artifacts = {}

    keys = compute_keys(inputs, artifacts)

    for node in DAG:
        key = keys.get(node)

        if key is None:
            break

        entry = get_cache(session_state, node, key)

        if entry is None:
            break

        artifacts[node] = entry["artifactDigest"]

        keys = compute_keys(inputs, artifacts)

    return artifacts, keys


def event_parent_available(session_state, inputs, node):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = reusable_artifacts(
        session_state,
        inputs,
    )

    parent_key = keys.get(parent)

    if parent_key is None:
        return False

    return (
        get_cache(
            session_state,
            parent,
            parent_key,
        )
        is not None
    )


def apply_success(session_state, event):
    node = event["node"]
    key = event["key"]
    artifact = event["artifactDigest"]
    event_id = event["eventId"]

    cache = session_state["cache"][node]

    # First successful evidence is immutable.
    existing = cache.get(key)

    if existing is None:
        cache[key] = {
            "artifactDigest": artifact,
            "eventId": event_id,
        }

    # Current state is also successful.
    session_state["state"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event_id,
        "artifactDigest": artifact,
    }


def process_valid_current_event(session_state, event):
    """
    Returns:
        ("accept", None)
        ("ignore", None)
        ("conflict", ERROR_CODE)
    """

    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    cache_entry = get_cache(
        session_state,
        node,
        key,
    )

    # Immutable successful cache entry.
    if cache_entry is not None:
        if status == "succeeded":
            if (
                event["artifactDigest"]
                != cache_entry["artifactDigest"]
            ):
                return "conflict", "EVIDENCE_CONFLICT"

            # Exact immutable evidence replay.
            return "ignore", None

        # Once a key has succeeded, any new non-success transition
        # is a status conflict.
        return "conflict", "STATUS_CONFLICT"

    current = session_state["state"].get(node)

    # No current state.
    if current is None:
        if status == "started" and attempt == 1:
            session_state["state"][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId": event["eventId"],
            }
            return "accept", None

        # Completion or attempt > 1 without the first start is ignored.
        return "ignore", None

    current_key = current["key"]
    current_status = current["status"]
    current_attempt = current["attempt"]

    # Event for an old/non-current key.
    if key != current_key:
        return "ignore", None

    # started(n) -> completion at n
    if current_status == "started":
        if (
            status in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }
            and attempt == current_attempt
        ):
            if status == "succeeded":
                apply_success(
                    session_state,
                    event,
                )
            else:
                session_state["state"][node] = {
                    "status": status,
                    "attempt": attempt,
                    "key": key,
                    "eventId": event["eventId"],
                }

            return "accept", None

        # Lower attempts are stale.
        if attempt < current_attempt:
            return "ignore", None

        return "conflict", "STATUS_CONFLICT"

    # retryable_failed(n) -> started(n+1)
    if current_status == "retryable_failed":
        if (
            status == "started"
            and attempt == current_attempt + 1
        ):
            session_state["state"][node] = {
                "status": "started",
                "attempt": attempt,
                "key": key,
                "eventId": event["eventId"],
            }
            return "accept", None

        if attempt < current_attempt:
            return "ignore", None

        return "conflict", "STATUS_CONFLICT"

    # Successful current state.
    if current_status == "succeeded":
        return "conflict", "STATUS_CONFLICT"

    # Terminal failure cannot transition.
    if current_status == "terminal_failed":
        return "conflict", "STATUS_CONFLICT"

    return "conflict", "STATUS_CONFLICT"


def dependency_values(
    node,
    inputs,
    artifacts,
):
    if node == "verify_data":
        return {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }

    if node == "prepare":
        return {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
        }

    if node == "train":
        return {
            "prepareArtifact": artifacts.get("prepare"),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    if node == "evaluate":
        return {
            "trainArtifact": artifacts.get("train"),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    if node == "register":
        return {
            "evaluateArtifact": artifacts.get("evaluate"),
            "schemaDigest": inputs["schemaDigest"],
        }

    if node == "publish":
        return {
            "registerArtifact": artifacts.get("register"),
            "publishConfig": inputs["publishConfig"],
        }

    return {}


def build_nodes(session_state, inputs):
    artifacts = {}
    keys = compute_keys(inputs, artifacts)

    node_info = {}

    for index, node in enumerate(DAG):
        key = keys.get(node)

        # Parent unavailable => this node cannot even calculate
        # a content-addressed key.
        if key is None:
            parent = PARENT[node]

            if parent is not None:
                parent_info = node_info[parent]

                if (
                    parent_info["reason"]
                    == "TERMINAL_FAILURE"
                ):
                    reason = "UPSTREAM_TERMINAL"
                    trigger_ids = parent_info["trigger_ids"]
                else:
                    reason = "UPSTREAM_PENDING"
                    trigger_ids = parent_info["trigger_ids"]
            else:
                reason = "UPSTREAM_PENDING"
                trigger_ids = []

            node_info[node] = {
                "action": "block",
                "reason": reason,
                "trigger_ids": trigger_ids,
                "key": None,
                "artifacts": dict(artifacts),
            }

            continue

        cache_entry = get_cache(
            session_state,
            node,
            key,
        )

        # Successful immutable cache hit.
        if cache_entry is not None:
            artifacts[node] = cache_entry["artifactDigest"]

            keys = compute_keys(
                inputs,
                artifacts,
            )

            node_info[node] = {
                "action": "reuse",
                "reason": "CACHE_HIT",
                "trigger_ids": [
                    cache_entry["eventId"]
                ],
                "key": key,
                "artifacts": dict(artifacts),
            }

            continue

        current = session_state["state"].get(node)

        if (
            current is not None
            and current["key"] == key
        ):
            if current["status"] == "started":
                action = "block"
                reason = "RUNNING"
                trigger_ids = [current["eventId"]]

            elif current["status"] == "retryable_failed":
                action = "rerun"
                reason = "RETRYABLE_FAILURE"
                trigger_ids = [current["eventId"]]

            elif current["status"] == "terminal_failed":
                action = "block"
                reason = "TERMINAL_FAILURE"
                trigger_ids = [current["eventId"]]

            else:
                action = "rerun"
                reason = "CACHE_MISS"
                trigger_ids = []

        else:
            action = "rerun"
            reason = "CACHE_MISS"
            trigger_ids = []

        node_info[node] = {
            "action": action,
            "reason": reason,
            "trigger_ids": trigger_ids,
            "key": key,
            "artifacts": dict(artifacts),
        }

        # Since this node is not reusable, all later nodes are
        # blocked by this pending/failed node.
        for descendant in DAG[index + 1:]:
            if descendant not in node_info:
                node_info[descendant] = {
                    "action": "block",
                    "reason": (
                        "UPSTREAM_TERMINAL"
                        if reason == "TERMINAL_FAILURE"
                        else "UPSTREAM_PENDING"
                    ),
                    "trigger_ids": trigger_ids,
                    "key": None,
                    "artifacts": dict(artifacts),
                }

        break

    result = []

    # Construct response strictly in DAG order.
    for node in DAG:
        info = node_info.get(node)

        if info is None:
            info = {
                "action": "block",
                "reason": "UPSTREAM_PENDING",
                "trigger_ids": [],
                "key": None,
                "artifacts": dict(artifacts),
            }

        deps = dependency_values(
            node,
            inputs,
            info["artifacts"],
        )

        # dependencyDigests includes the named dependencies plus cacheKey.
        if info["key"] is not None:
            deps["cacheKey"] = info["key"]

        result.append({
            "node": node,
            "action": info["action"],
            "reasonCodes": [info["reason"]],
            "dependencyDigests": deps,
            "triggeringEventIds": info["trigger_ids"],
        })

    return result


@app.post("/pipeline")
async def pipeline(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response("INVALID_REQUEST")

    if not valid_request_shape(body):
        return error_response("INVALID_REQUEST")

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Validate the entire event batch before changing anything.
    for event in events:
        if not valid_event_shape(event):
            return error_response("INVALID_EVENT")

    with LOCK:
        original = SESSIONS.get(session_id)

        # First request for this session.
        if original is None:
            original = empty_session()

        # Same revision must have identical complete inputs,
        # including extra metadata.
        if original["revision"] is not None:
            if revision == original["revision"]:
                if (
                    original["inputs"]
                    != input_fingerprint(inputs)
                ):
                    return error_response(
                        "REVISION_CONFLICT"
                    )

            # An older request is stale. Its well-formed events are
            # ignored and must not mutate current state.
            elif revision < original["revision"]:
                return JSONResponse(
                    status_code=200,
                    content={
                        "revision": original["revision"],
                        "acceptedEventIds": [],
                        "ignoredEventIds": [
                            event["eventId"]
                            for event in events
                        ],
                        "nodes": build_nodes(
                            original,
                            json.loads(original["inputs"]),
                        ),
                    },
                )

        # Transactional working copy.
        #
        # This is important: if a later event causes a 409,
        # successful cache entries created by earlier events in
        # this batch must also roll back.
        working = copy.deepcopy(original)

        # New revision:
        # replace inputs and clear attempt/terminal state,
        # but preserve successful content-addressed cache entries.
        if (
            working["revision"] is None
            or revision > working["revision"]
        ):
            old_cache = copy.deepcopy(
                working["cache"]
            )

            working = empty_session()

            working["revision"] = revision
            working["inputs"] = input_fingerprint(
                inputs
            )

            working["cache"] = old_cache

        else:
            working["revision"] = revision
            working["inputs"] = input_fingerprint(
                inputs
            )

        accepted_ids = []
        ignored_ids = []

        # Process valid events strictly in input order.
        for event in events:
            event_id = event["eventId"]
            event_canonical = canonical_event(event)

            # Global event ID within this session.
            if event_id in working["event_canonical"]:
                if (
                    working["event_canonical"][event_id]
                    == event_canonical
                ):
                    # Exact replay.
                    ignored_ids.append(event_id)
                    continue

                # Same ID but different canonical event.
                return error_response(
                    "EVENT_ID_CONFLICT"
                )

            # Older/future revision event is ignored.
            if event["revision"] != working["revision"]:
                ignored_ids.append(event_id)
                continue

            node = event["node"]

            # Wrong/unknown node is ignored.
            if node not in DAG:
                ignored_ids.append(event_id)
                continue

            # Parent must be reusable before this event can be processed.
            if not event_parent_available(
                working,
                inputs,
                node,
            ):
                ignored_ids.append(event_id)
                continue

            # Compute current content-addressed key.
            artifacts, keys = reusable_artifacts(
                working,
                inputs,
            )

            current_key = keys.get(node)

            # Wrong/stale key is ignored.
            if (
                current_key is None
                or event["key"] != current_key
            ):
                ignored_ids.append(event_id)
                continue

            result, code = process_valid_current_event(
                working,
                event,
            )

            if result == "conflict":
                # Nothing has been committed to SESSIONS yet.
                # Therefore the complete batch rolls back atomically.
                return error_response(code)

            if result == "accept":
                # Accepted IDs are consumed permanently.
                working["event_canonical"][
                    event_id
                ] = event_canonical

                working["event_data"][
                    event_id
                ] = copy.deepcopy(event)

                accepted_ids.append(event_id)

            else:
                # Ignored events do NOT consume their IDs.
                ignored_ids.append(event_id)

        # Commit the complete transaction.
        SESSIONS[session_id] = working

        response_nodes = build_nodes(
            working,
            inputs,
        )

        return JSONResponse(
            status_code=200,
            content={
                "revision": working["revision"],
                "acceptedEventIds": accepted_ids,
                "ignoredEventIds": ignored_ids,
                "nodes": response_nodes,
            },
        )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "content-addressed-ml-pipeline",
    }