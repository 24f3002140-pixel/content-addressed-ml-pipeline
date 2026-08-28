import copy
import hashlib
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE_INTEGER = 9007199254740991

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

EVENT_NAMES = [
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
]

VALID_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

LOCK = threading.RLock()

# Process-persistent state.
# Each session gets its own completely independent state.
SESSIONS = {}


# ================================================================
# HTTP ERRORS
# ================================================================

def error_response(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ================================================================
# JSON / HASHING
# ================================================================

def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def content_key(values):
    """
    Lowercase SHA-256 over UTF-8 compact JSON array.
    """
    return digest_text(compact(values))


def positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= MAX_SAFE_INTEGER
    )


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


# ================================================================
# SESSION STATE
# ================================================================

def empty_session():
    return {
        "revision": None,
        "inputs": None,

        # eventId -> canonical event JSON
        "event_ids": {},

        # Current execution state by node.
        "state": {
            node: None
            for node in DAG
        },

        # Successful immutable cache.
        #
        # node -> key -> {
        #     artifactDigest,
        #     eventId
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


# ================================================================
# REQUEST VALIDATION
# ================================================================

def validate_request(body):
    if not isinstance(body, dict):
        return False

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if not required.issubset(body.keys()):
        return False

    if not nonempty_string(body["session"]):
        return False

    if not positive_safe_integer(body["revision"]):
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


def validate_event(event):
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != set(EVENT_NAMES):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not positive_safe_integer(
        event["revision"]
    ):
        return False

    # Node syntax itself must be a string.
    # Unknown nodes are handled as ignored events.
    if not nonempty_string(event["node"]):
        return False

    if not positive_safe_integer(
        event["attempt"]
    ):
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

    # Receipt is mandatory only for successful
    # register/publish events.
    if (
        event["status"] == "succeeded"
        and event["node"] in {
            "register",
            "publish",
        }
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
    return compact([
        event["eventId"],
        event["revision"],
        event["node"],
        event["attempt"],
        event["status"],
        event["key"],
        event["artifactDigest"],
        event["receiptId"],
    ])


# ================================================================
# EXACT KEY DEPENDENCIES
# ================================================================

def key_dependencies(node, inputs, artifacts):
    """
    EXACT arrays from the assignment.
    """

    if node == "verify_data":
        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    if node == "prepare":
        return [
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ]

    if node == "train":
        if "prepare" not in artifacts:
            return None

        return [
            artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        if "train" not in artifacts:
            return None

        return [
            artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        if "evaluate" not in artifacts:
            return None

        return [
            artifacts["evaluate"],
            inputs["schemaDigest"],
        ]

    if node == "publish":
        if "register" not in artifacts:
            return None

        return [
            artifacts["register"],
            inputs["publishConfig"],
        ]

    return None


# ================================================================
# CACHE RECOVERY
# ================================================================

def recover_cache(session, inputs):
    """
    Walk the DAG in order.

    A node can only become reusable if:
      - its parent is reusable, and
      - its exact content key exists in cache.

    Once a node misses cache, downstream keys are null.
    """

    artifacts = {}
    keys = {}

    for node in DAG:

        deps = key_dependencies(
            node,
            inputs,
            artifacts,
        )

        if deps is None:
            keys[node] = None
            break

        key = content_key(deps)
        keys[node] = key

        entry = session["cache"][node].get(key)

        if entry is None:
            break

        artifacts[node] = entry[
            "artifactDigest"
        ]

    return artifacts, keys


# ================================================================
# RESPONSE DEPENDENCIES
# ================================================================

def response_dependencies(
    node,
    inputs,
    artifacts,
    key,
):
    if node == "verify_data":
        result = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }

    elif node == "prepare":
        result = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
        }

    elif node == "train":
        result = {
            "prepareArtifact": artifacts.get(
                "prepare"
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    elif node == "evaluate":
        result = {
            "trainArtifact": artifacts.get(
                "train"
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    elif node == "register":
        result = {
            "evaluateArtifact": artifacts.get(
                "evaluate"
            ),
            "schemaDigest": inputs["schemaDigest"],
        }

    elif node == "publish":
        result = {
            "registerArtifact": artifacts.get(
                "register"
            ),
            "publishConfig": inputs["publishConfig"],
        }

    else:
        result = {}

    if key is not None:
        result["cacheKey"] = key

    return result


# ================================================================
# SUCCESSFUL PARENT CHECK
# ================================================================

def parent_is_reusable(
    session,
    inputs,
    node,
):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = recover_cache(
        session,
        inputs,
    )

    return (
        parent in artifacts
        and keys.get(parent) is not None
    )


# ================================================================
# IMMUTABLE SUCCESS CACHE
# ================================================================

def record_success(
    session,
    event,
):
    node = event["node"]
    key = event["key"]

    # First successful evidence permanently binds
    # this content key to this artifact.
    if key not in session["cache"][node]:
        session["cache"][node][key] = {
            "artifactDigest":
                event["artifactDigest"],
            "eventId":
                event["eventId"],
        }

    session["state"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event["eventId"],
        "artifactDigest":
            event["artifactDigest"],
    }


# ================================================================
# EVENT STATE MACHINE
# ================================================================

def apply_transition(
    session,
    event,
):
    node = event["node"]
    key = event["key"]
    attempt = event["attempt"]
    status = event["status"]

    # ------------------------------------------------------------
    # Immutable cache already exists.
    # ------------------------------------------------------------

    cached = session["cache"][node].get(key)

    if cached is not None:

        if status == "succeeded":

            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return (
                    "conflict",
                    "EVIDENCE_CONFLICT",
                )

            return "ignore", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # No current state.
    # ------------------------------------------------------------

    current = session["state"][node]

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

        # Completion without initial start is ignored.
        return "ignore", None

    # ------------------------------------------------------------
    # Old content key.
    # ------------------------------------------------------------

    if current["key"] != key:
        return "ignore", None

    previous_status = current["status"]
    previous_attempt = current["attempt"]

    # ------------------------------------------------------------
    # started(n)
    # ------------------------------------------------------------

    if previous_status == "started":

        if attempt < previous_attempt:
            return "ignore", None

        if (
            attempt == previous_attempt
            and status in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }
        ):

            if status == "succeeded":
                record_success(
                    session,
                    event,
                )
            else:
                session["state"][node] = {
                    "status": status,
                    "attempt": attempt,
                    "key": key,
                    "eventId":
                        event["eventId"],
                }

            return "accept", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # retryable_failed(n)
    # ------------------------------------------------------------

    if previous_status == "retryable_failed":

        if attempt < previous_attempt:
            return "ignore", None

        if (
            status == "started"
            and attempt == previous_attempt + 1
        ):
            session["state"][node] = {
                "status": "started",
                "attempt": attempt,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return "accept", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # Terminal state is final.
    # ------------------------------------------------------------

    if previous_status == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # Successful current state is final.
    # ------------------------------------------------------------

    if previous_status == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ================================================================
# RESPONSE NODE CONSTRUCTION
# ================================================================

def build_nodes(
    session,
    inputs,
):
    artifacts, keys = recover_cache(
        session,
        inputs,
    )

    output = []

    blocking_reason = None
    blocking_event_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # --------------------------------------------------------
        # Parent-gated / unavailable.
        # --------------------------------------------------------

        if key is None:

            if (
                blocking_reason
                == "TERMINAL_FAILURE"
            ):
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            output.append({
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests":
                    response_dependencies(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    blocking_event_ids,
            })

            continue

        # --------------------------------------------------------
        # Cache hit.
        # --------------------------------------------------------

        cache_entry = session[
            "cache"
        ][node].get(key)

        if cache_entry is not None:

            output.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    response_dependencies(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [
                    cache_entry["eventId"]
                ],
            })

            continue

        # --------------------------------------------------------
        # Cache miss: inspect execution state.
        # --------------------------------------------------------

        state = session["state"][node]

        if (
            state is not None
            and state["key"] == key
        ):

            if state["status"] == "started":
                action = "block"
                reason = "RUNNING"

            elif (
                state["status"]
                == "retryable_failed"
            ):
                action = "rerun"
                reason = "RETRYABLE_FAILURE"

            elif (
                state["status"]
                == "terminal_failed"
            ):
                action = "block"
                reason = "TERMINAL_FAILURE"

            else:
                action = "rerun"
                reason = "CACHE_MISS"

            trigger_ids = [
                state["eventId"]
            ]

        else:
            action = "rerun"
            reason = "CACHE_MISS"
            trigger_ids = []

        output.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests":
                response_dependencies(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds":
                trigger_ids,
        })

        # --------------------------------------------------------
        # Every descendant is blocked by this pending/terminal
        # node.
        # --------------------------------------------------------

        blocking_reason = reason
        blocking_event_ids = trigger_ids

        for later in DAG[index + 1:]:

            output.append({
                "node": later,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                    if reason == "TERMINAL_FAILURE"
                    else "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    response_dependencies(
                        later,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    trigger_ids,
            })

        break

    return output


# ================================================================
# POST /pipeline
# ================================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return error_response(
            "INVALID_REQUEST"
        )

    if not validate_request(body):
        return error_response(
            "INVALID_REQUEST"
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Validate event shape first.
    for event in events:
        if not validate_event(event):
            return error_response(
                "INVALID_EVENT"
            )

    with LOCK:

        current = SESSIONS.get(
            session_id
        )

        if current is None:
            current = empty_session()

        # ========================================================
        # OLD REVISION
        # ========================================================

        if (
            current["revision"] is not None
            and revision < current["revision"]
        ):

            old_inputs = json.loads(
                current["inputs"]
            )

            return JSONResponse(
                status_code=200,
                content={
                    "revision":
                        current["revision"],
                    "acceptedEventIds": [],
                    "ignoredEventIds": [
                        e["eventId"]
                        for e in events
                    ],
                    "nodes":
                        build_nodes(
                            current,
                            old_inputs,
                        ),
                },
            )

        # ========================================================
        # SAME REVISION
        # ========================================================

        if (
            current["revision"] is not None
            and revision == current["revision"]
        ):

            # Entire input object is immutable for a revision.
            if (
                canonical_inputs(
                    inputs
                )
                != current["inputs"]
            ):
                return error_response(
                    "REVISION_CONFLICT"
                )

        # ========================================================
        # ATOMIC WORKING COPY
        # ========================================================

        working = copy.deepcopy(
            current
        )

        # ========================================================
        # NEW REVISION
        # ========================================================

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            # Successful cache is the ONLY state preserved.
            old_cache = copy.deepcopy(
                working["cache"]
            )

            working = empty_session()

            working["revision"] = revision
            working["inputs"] = (
                canonical_inputs(inputs)
            )
            working["cache"] = old_cache

        else:
            working["revision"] = revision
            working["inputs"] = (
                canonical_inputs(inputs)
            )

        accepted_ids = []
        ignored_ids = []

        # ========================================================
        # EVENTS IN EXACT INPUT ORDER
        # ========================================================

        for event in events:

            event_id = event["eventId"]

            # ----------------------------------------------------
            # Wrong revision = ignored.
            # ----------------------------------------------------

            if event["revision"] != revision:
                ignored_ids.append(
                    event_id
                )
                continue

            # ----------------------------------------------------
            # Unknown node = ignored.
            # ----------------------------------------------------

            if event["node"] not in DAG:
                ignored_ids.append(
                    event_id
                )
                continue

            canonical = canonical_event(
                event
            )

            # ----------------------------------------------------
            # Event ID replay/conflict.
            # ----------------------------------------------------

            if event_id in working[
                "event_ids"
            ]:

                if (
                    working[
                        "event_ids"
                    ][event_id]
                    == canonical
                ):
                    ignored_ids.append(
                        event_id
                    )
                    continue

                # Same ID but different canonical event.
                return error_response(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # ----------------------------------------------------
            # Parent must be reusable.
            # ----------------------------------------------------

            if not parent_is_reusable(
                working,
                inputs,
                node,
            ):
                ignored_ids.append(
                    event_id
                )
                continue

            # ----------------------------------------------------
            # Calculate current exact key.
            # ----------------------------------------------------

            artifacts, keys = recover_cache(
                working,
                inputs,
            )

            expected_key = keys.get(node)

            if expected_key is None:
                ignored_ids.append(
                    event_id
                )
                continue

            # Wrong/stale key is ignored.
            if event["key"] != expected_key:
                ignored_ids.append(
                    event_id
                )
                continue

            # ----------------------------------------------------
            # State transition.
            # ----------------------------------------------------

            outcome, code = apply_transition(
                working,
                event,
            )

            if outcome == "conflict":
                # Absolutely nothing from this batch commits.
                return error_response(code)

            if outcome == "accept":

                working[
                    "event_ids"
                ][event_id] = canonical

                accepted_ids.append(
                    event_id
                )

            else:
                # Ignored events do NOT consume IDs.
                ignored_ids.append(
                    event_id
                )

        # ========================================================
        # ATOMIC COMMIT
        # ========================================================

        SESSIONS[session_id] = working

        return JSONResponse(
            status_code=200,
            content={
                "revision":
                    working["revision"],
                "acceptedEventIds":
                    accepted_ids,
                "ignoredEventIds":
                    ignored_ids,
                "nodes":
                    build_nodes(
                        working,
                        inputs,
                    ),
            },
        )


def canonical_inputs(inputs):
    return compact(inputs)


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service":
            "content-addressed-ml-pipeline",
    }
