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

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

SESSIONS = {}


def new_session():
    return {
        "revision": None,
        "inputs": None,

        # eventId -> canonical event JSON
        "event_canonical": {},

        # eventId -> event
        "event_data": {},

        # Current execution state for every node.
        "state": {
            node: None
            for node in DAG
        },

        # Immutable successful content-addressed cache.
        #
        # cache[node][key] = {
        #     "artifactDigest": "...",
        #     "eventId": "..."
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


# ---------------------------------------------------------------------
# JSON / hashing helpers
# ---------------------------------------------------------------------

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_string(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def content_key(values):
    """
    Lowercase SHA-256 of UTF-8 compact JSON array.
    """
    return sha256_string(
        compact_json(values)
    )


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= SAFE_INT_MAX
    )


def nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def conflict(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ---------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------

def valid_request(body):
    if not isinstance(body, dict):
        return False

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if set(body.keys()) != required:
        return False

    if not nonempty_string(body["session"]):
        return False

    if not safe_positive_int(body["revision"]):
        return False

    if not isinstance(body["inputs"], dict):
        return False

    if not isinstance(body["events"], list):
        return False

    for name in INPUT_NAMES:
        if not nonempty_string(
            body["inputs"].get(name)
        ):
            return False

    return True


def valid_event(event):
    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not safe_positive_int(event["revision"]):
        return False

    if not nonempty_string(event["node"]):
        return False

    if event["node"] not in DAG:
        # Unknown nodes are handled as invalid event structure.
        return False

    if not safe_positive_int(event["attempt"]):
        return False

    if event["status"] not in VALID_STATUSES:
        return False

    if not nonempty_string(event["key"]):
        return False

    if event["status"] == "succeeded":
        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    # Receipt is only allowed/required for successful
    # register and publish events.
    if (
        event["node"] in {"register", "publish"}
        and event["status"] == "succeeded"
    ):
        expected = (
            "receipt:"
            + event["node"]
            + ":"
            + event["key"]
        )

        if event["receiptId"] != expected:
            return False
    else:
        if event["receiptId"] is not None:
            return False

    return True


def canonical_event(event):
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


# ---------------------------------------------------------------------
# Input identity
# ---------------------------------------------------------------------

def canonical_inputs(inputs):
    """
    The complete inputs object is the revision identity.

    Extra metadata therefore participates in the identity exactly as
    required: changing any extra metadata changes the revision inputs.
    """
    return compact_json(inputs)


# ---------------------------------------------------------------------
# Content-addressed DAG
# ---------------------------------------------------------------------

def calculate_key(node, inputs, artifacts):
    """
    Calculate the current content-addressed key.

    A downstream key is deliberately None until its parent artifact
    exists. This is the parent-gating rule.
    """

    if node == "verify_data":
        return content_key([
            inputs["generation"],
            inputs["checksum"],
        ])

    if node == "prepare":
        parent_artifact = artifacts.get(
            "verify_data"
        )

        if parent_artifact is None:
            return None

        return content_key([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    if node == "train":
        parent_artifact = artifacts.get(
            "prepare"
        )

        if parent_artifact is None:
            return None

        return content_key([
            parent_artifact,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    if node == "evaluate":
        parent_artifact = artifacts.get(
            "train"
        )

        if parent_artifact is None:
            return None

        return content_key([
            parent_artifact,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    if node == "register":
        parent_artifact = artifacts.get(
            "evaluate"
        )

        if parent_artifact is None:
            return None

        return content_key([
            parent_artifact,
            inputs["schemaDigest"],
        ])

    if node == "publish":
        parent_artifact = artifacts.get(
            "register"
        )

        if parent_artifact is None:
            return None

        return content_key([
            parent_artifact,
            inputs["publishConfig"],
        ])

    return None


def reusable_chain(session, inputs):
    """
    Recover the longest reusable successful cache chain.

    Importantly, only successful immutable cache entries are used.
    A current execution state by itself never makes a downstream key
    available.
    """

    artifacts = {}
    keys = {}

    for node in DAG:
        key = calculate_key(
            node,
            inputs,
            artifacts,
        )

        keys[node] = key

        if key is None:
            break

        entry = session["cache"][node].get(key)

        if entry is None:
            break

        artifacts[node] = entry["artifactDigest"]

    return artifacts, keys


# ---------------------------------------------------------------------
# Dependency output
# ---------------------------------------------------------------------

def dependency_values(
    node,
    inputs,
    artifacts,
):
    """
    Preserve the exact dependency order from the specification.
    """

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
            "prepareArtifact": artifacts.get(
                "prepare"
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    if node == "evaluate":
        return {
            "trainArtifact": artifacts.get(
                "train"
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    if node == "register":
        return {
            "evaluateArtifact": artifacts.get(
                "evaluate"
            ),
            "schemaDigest": inputs["schemaDigest"],
        }

    if node == "publish":
        return {
            "registerArtifact": artifacts.get(
                "register"
            ),
            "publishConfig": inputs["publishConfig"],
        }

    return {}


# ---------------------------------------------------------------------
# Parent availability
# ---------------------------------------------------------------------

def parent_is_reusable(session, inputs, node):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = reusable_chain(
        session,
        inputs,
    )

    parent_key = keys.get(parent)

    if parent_key is None:
        return False

    return (
        parent_key in session["cache"][parent]
        and parent in artifacts
    )


# ---------------------------------------------------------------------
# Immutable evidence
# ---------------------------------------------------------------------

def commit_success(session, event):
    node = event["node"]
    key = event["key"]
    artifact = event["artifactDigest"]
    event_id = event["eventId"]

    existing = session["cache"][node].get(key)

    if existing is None:
        # Permanently bind key to first successful artifact/event.
        session["cache"][node][key] = {
            "artifactDigest": artifact,
            "eventId": event_id,
        }

    session["state"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event_id,
        "artifactDigest": artifact,
    }


# ---------------------------------------------------------------------
# Event state machine
# ---------------------------------------------------------------------

def process_event(session, event):
    """
    Return:
        ("accept", None)
        ("ignore", None)
        ("conflict", ERROR_CODE)
    """

    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # Immutable successful cache.
    cache_entry = session["cache"][node].get(key)

    if cache_entry is not None:

        if status == "succeeded":
            if (
                event["artifactDigest"]
                != cache_entry["artifactDigest"]
            ):
                return (
                    "conflict",
                    "EVIDENCE_CONFLICT",
                )

            # Exact same successful evidence.
            return "ignore", None

        # Any other new event after successful evidence
        # is a status conflict.
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    current = session["state"].get(node)

    # -------------------------------------------------------------
    # No current state
    # -------------------------------------------------------------

    if current is None:

        # Only started(1) can begin a node.
        if (
            status == "started"
            and attempt == 1
        ):
            session["state"][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        # Completion without start, or attempt > 1,
        # is stale/invalid for the current execution and ignored.
        return "ignore", None

    current_key = current["key"]
    current_status = current["status"]
    current_attempt = current["attempt"]

    # Event for a different/non-current key is stale.
    if key != current_key:
        return "ignore", None

    # -------------------------------------------------------------
    # started(n)
    # -------------------------------------------------------------

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
                commit_success(
                    session,
                    event,
                )
            else:
                session["state"][node] = {
                    "status": status,
                    "attempt": attempt,
                    "key": key,
                    "eventId": event["eventId"],
                }

            return "accept", None

        # Lower attempt is stale.
        if attempt < current_attempt:
            return "ignore", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # -------------------------------------------------------------
    # retryable_failed(n)
    # -------------------------------------------------------------

    if current_status == "retryable_failed":

        if (
            status == "started"
            and attempt == current_attempt + 1
        ):
            session["state"][node] = {
                "status": "started",
                "attempt": attempt,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        if attempt < current_attempt:
            return "ignore", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # -------------------------------------------------------------
    # succeeded
    # -------------------------------------------------------------

    if current_status == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # -------------------------------------------------------------
    # terminal_failed
    # -------------------------------------------------------------

    if current_status == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ---------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------

def build_nodes(session, inputs):
    artifacts, keys = reusable_chain(
        session,
        inputs,
    )

    info = {}

    for node in DAG:
        key = keys.get(node)

        # ---------------------------------------------------------
        # Parent-gated node
        # ---------------------------------------------------------

        if key is None:
            parent = PARENT[node]

            if parent is None:
                reason = "UPSTREAM_PENDING"
                trigger_ids = []
            else:
                parent_info = info.get(parent)

                if (
                    parent_info is not None
                    and parent_info["reason"]
                    == "TERMINAL_FAILURE"
                ):
                    reason = "UPSTREAM_TERMINAL"
                    trigger_ids = parent_info[
                        "trigger_ids"
                    ]
                else:
                    reason = "UPSTREAM_PENDING"

                    if parent_info is not None:
                        trigger_ids = parent_info[
                            "trigger_ids"
                        ]
                    else:
                        trigger_ids = []

            info[node] = {
                "action": "block",
                "reason": reason,
                "trigger_ids": trigger_ids,
                "key": None,
                "artifacts": dict(artifacts),
            }

            continue

        # ---------------------------------------------------------
        # Cache hit
        # ---------------------------------------------------------

        cache_entry = session["cache"][node].get(
            key
        )

        if cache_entry is not None:
            artifacts[node] = (
                cache_entry["artifactDigest"]
            )

            # Recalculate downstream keys immediately after
            # recovering this artifact.
            keys = {}

            for downstream in DAG:
                keys[downstream] = calculate_key(
                    downstream,
                    inputs,
                    artifacts,
                )

            info[node] = {
                "action": "reuse",
                "reason": "CACHE_HIT",
                "trigger_ids": [
                    cache_entry["eventId"]
                ],
                "key": key,
                "artifacts": dict(artifacts),
            }

            continue

        # ---------------------------------------------------------
        # Cache miss / current execution state
        # ---------------------------------------------------------

        current = session["state"].get(node)

        if (
            current is not None
            and current["key"] == key
        ):
            if current["status"] == "started":
                action = "block"
                reason = "RUNNING"
                trigger_ids = [
                    current["eventId"]
                ]

            elif (
                current["status"]
                == "retryable_failed"
            ):
                action = "rerun"
                reason = "RETRYABLE_FAILURE"
                trigger_ids = [
                    current["eventId"]
                ]

            elif (
                current["status"]
                == "terminal_failed"
            ):
                action = "block"
                reason = "TERMINAL_FAILURE"
                trigger_ids = [
                    current["eventId"]
                ]

            else:
                action = "rerun"
                reason = "CACHE_MISS"
                trigger_ids = []

        else:
            action = "rerun"
            reason = "CACHE_MISS"
            trigger_ids = []

        info[node] = {
            "action": action,
            "reason": reason,
            "trigger_ids": trigger_ids,
            "key": key,
            "artifacts": dict(artifacts),
        }

        # Every downstream node is parent-gated.
        for later in DAG[
            DAG.index(node) + 1:
        ]:
            if later not in info:
                info[later] = {
                    "action": "block",
                    "reason": (
                        "UPSTREAM_TERMINAL"
                        if reason
                        == "TERMINAL_FAILURE"
                        else "UPSTREAM_PENDING"
                    ),
                    "trigger_ids": trigger_ids,
                    "key": None,
                    "artifacts": dict(
                        artifacts
                    ),
                }

        break

    # -------------------------------------------------------------
    # Final DAG-ordered response
    # -------------------------------------------------------------

    result = []

    for node in DAG:
        node_info = info.get(node)

        if node_info is None:
            node_info = {
                "action": "block",
                "reason": "UPSTREAM_PENDING",
                "trigger_ids": [],
                "key": None,
                "artifacts": dict(artifacts),
            }

        dependencies = dependency_values(
            node,
            inputs,
            node_info["artifacts"],
        )

        # cacheKey is always included when the key exists.
        if node_info["key"] is not None:
            dependencies["cacheKey"] = (
                node_info["key"]
            )

        result.append({
            "node": node,
            "action": node_info["action"],
            "reasonCodes": [
                node_info["reason"]
            ],
            "dependencyDigests": dependencies,
            "triggeringEventIds": (
                node_info["trigger_ids"]
            ),
        })

    return result


# ---------------------------------------------------------------------
# POST /pipeline
# ---------------------------------------------------------------------

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return conflict("INVALID_REQUEST")

    if not valid_request(body):
        return conflict("INVALID_REQUEST")

    session_id = body["session"]
    request_revision = body["revision"]
    request_inputs = body["inputs"]
    request_events = body["events"]

    # Validate the complete event batch before mutating anything.
    for event in request_events:
        if not valid_event(event):
            return conflict("INVALID_EVENT")

    with LOCK:

        existing = SESSIONS.get(
            session_id
        )

        if existing is None:
            existing = new_session()

        # ---------------------------------------------------------
        # Existing revision checks
        # ---------------------------------------------------------

        if existing["revision"] is not None:

            if (
                request_revision
                == existing["revision"]
            ):
                # Same revision must have byte-for-byte equivalent
                # compact input representation.
                if (
                    canonical_inputs(
                        request_inputs
                    )
                    != existing["inputs"]
                ):
                    return conflict(
                        "REVISION_CONFLICT"
                    )

            elif (
                request_revision
                < existing["revision"]
            ):
                # Entire request is stale.
                # Events from an older revision are ignored.
                ignored = [
                    event["eventId"]
                    for event in request_events
                ]

                nodes = build_nodes(
                    existing,
                    json.loads(
                        existing["inputs"]
                    ),
                )

                return JSONResponse(
                    status_code=200,
                    content={
                        "revision": existing[
                            "revision"
                        ],
                        "acceptedEventIds": [],
                        "ignoredEventIds": ignored,
                        "nodes": nodes,
                    },
                )

        # ---------------------------------------------------------
        # Transactional copy
        # ---------------------------------------------------------

        working = copy.deepcopy(existing)

        # ---------------------------------------------------------
        # New revision
        # ---------------------------------------------------------

        if (
            working["revision"] is None
            or request_revision
            > working["revision"]
        ):
            # Successful cache survives revision changes.
            old_cache = copy.deepcopy(
                working["cache"]
            )

            working = new_session()

            working["revision"] = (
                request_revision
            )

            working["inputs"] = (
                canonical_inputs(
                    request_inputs
                )
            )

            working["cache"] = old_cache

        else:
            working["revision"] = (
                request_revision
            )

            working["inputs"] = (
                canonical_inputs(
                    request_inputs
                )
            )

        accepted = []
        ignored = []

        # ---------------------------------------------------------
        # Process events in exact input order
        # ---------------------------------------------------------

        for event in request_events:

            event_id = event["eventId"]

            # -----------------------------------------------------
            # IMPORTANT:
            # Wrong revision is ignored BEFORE event-ID conflict
            # checking. Older-revision events must never consume IDs.
            # -----------------------------------------------------

            if (
                event["revision"]
                != working["revision"]
            ):
                ignored.append(event_id)
                continue

            event_json = canonical_event(event)

            # -----------------------------------------------------
            # Global event ID
            # -----------------------------------------------------

            if (
                event_id
                in working["event_canonical"]
            ):
                if (
                    working[
                        "event_canonical"
                    ][event_id]
                    == event_json
                ):
                    ignored.append(event_id)
                    continue

                return conflict(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # -----------------------------------------------------
            # Parent gate
            # -----------------------------------------------------

            if not parent_is_reusable(
                working,
                request_inputs,
                node,
            ):
                ignored.append(event_id)
                continue

            # -----------------------------------------------------
            # Current content-addressed key
            # -----------------------------------------------------

            artifacts, keys = reusable_chain(
                working,
                request_inputs,
            )

            current_key = keys.get(node)

            if (
                current_key is None
                or event["key"] != current_key
            ):
                ignored.append(event_id)
                continue

            # -----------------------------------------------------
            # State-machine transition
            # -----------------------------------------------------

            result, code = process_event(
                working,
                event,
            )

            if result == "conflict":
                # Atomic rollback:
                # SESSIONS has not been modified.
                return conflict(code)

            if result == "accept":

                working[
                    "event_canonical"
                ][event_id] = event_json

                working[
                    "event_data"
                ][event_id] = copy.deepcopy(
                    event
                )

                accepted.append(event_id)

            else:
                # Ignored events do not consume their IDs.
                ignored.append(event_id)

        # ---------------------------------------------------------
        # Commit entire batch atomically
        # ---------------------------------------------------------

        SESSIONS[session_id] = working

        nodes = build_nodes(
            working,
            request_inputs,
        )

        return JSONResponse(
            status_code=200,
            content={
                "revision": working["revision"],
                "acceptedEventIds": accepted,
                "ignoredEventIds": ignored,
                "nodes": nodes,
            },
        )


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "content-addressed-ml-pipeline",
    }
