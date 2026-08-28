import copy
import hashlib
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SAFE_MAX = 9007199254740991

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

INPUTS = [
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

STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

LOCK = threading.RLock()
SESSIONS = {}


# ================================================================
# BASIC HELPERS
# ================================================================

def conflict(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 1 <= x <= SAFE_MAX
    )


def nonempty(x):
    return isinstance(x, str) and len(x) > 0


def compact(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def make_key(values):
    return sha256(compact(values))


# ================================================================
# EMPTY SESSION
# ================================================================

def new_session():
    return {
        "revision": None,
        "inputs": None,

        # eventId -> canonical compact JSON
        "events": {},

        # Current attempt state
        "state": {
            node: None
            for node in DAG
        },

        # Successful content-addressed evidence.
        #
        # cache[node][key] = {
        #     artifactDigest: ...,
        #     eventId: ...
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

    if not all(
        x in body
        for x in [
            "session",
            "revision",
            "inputs",
            "events",
        ]
    ):
        return False

    if not nonempty(body["session"]):
        return False

    if not safe_int(body["revision"]):
        return False

    if not isinstance(body["inputs"], dict):
        return False

    if not isinstance(body["events"], list):
        return False

    # Required inputs must be non-empty strings.
    for name in INPUTS:
        if not nonempty(
            body["inputs"].get(name)
        ):
            return False

    # Extra metadata is allowed.

    return True


def validate_event(event):

    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty(event["eventId"]):
        return False

    if not safe_int(event["revision"]):
        return False

    if not nonempty(event["node"]):
        return False

    if not safe_int(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if not nonempty(event["key"]):
        return False

    if event["status"] == "succeeded":
        if not nonempty(
            event["artifactDigest"]
        ):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register/publish success requires exact receipt.
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


def event_canonical(event):
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
# EXACT DAG DEPENDENCY ARRAYS
# ================================================================

def dependency_array(
    node,
    inputs,
    artifacts,
):
    """
    These are the exact arrays specified by the task.
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
# RECOVER CACHE CHAIN
# ================================================================

def recover(session, inputs):

    artifacts = {}
    keys = {}

    for node in DAG:

        deps = dependency_array(
            node,
            inputs,
            artifacts,
        )

        # Parent has not produced reusable artifact.
        if deps is None:
            keys[node] = None
            break

        key = make_key(deps)

        keys[node] = key

        entry = session["cache"][node].get(
            key
        )

        # Cache miss means this node is not reusable,
        # and therefore every downstream key is null.
        if entry is None:
            break

        artifacts[node] = entry[
            "artifactDigest"
        ]

    return artifacts, keys


# ================================================================
# DEPENDENCY DIGEST OBJECT
# ================================================================

def dependency_digests(
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
            "canonicalData":
                inputs["canonicalData"],
            "prepareCode":
                inputs["prepareCode"],
            "prepareConfig":
                inputs["prepareConfig"],
        }

    elif node == "train":

        result = {
            "prepareArtifact":
                artifacts.get("prepare"),
            "trainCode":
                inputs["trainCode"],
            "trainConfig":
                inputs["trainConfig"],
            "runtime":
                inputs["runtime"],
        }

    elif node == "evaluate":

        result = {
            "trainArtifact":
                artifacts.get("train"),
            "canonicalData":
                inputs["canonicalData"],
            "evaluateCode":
                inputs["evaluateCode"],
            "evaluateConfig":
                inputs["evaluateConfig"],
        }

    elif node == "register":

        result = {
            "evaluateArtifact":
                artifacts.get("evaluate"),
            "schemaDigest":
                inputs["schemaDigest"],
        }

    elif node == "publish":

        result = {
            "registerArtifact":
                artifacts.get("register"),
            "publishConfig":
                inputs["publishConfig"],
        }

    else:
        result = {}

    if key is not None:
        result["cacheKey"] = key

    return result


# ================================================================
# SUCCESS CACHE
# ================================================================

def store_success(
    session,
    event,
):

    node = event["node"]
    key = event["key"]

    # First successful evidence wins permanently.
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
# APPLY EVENT
# ================================================================

def apply_event(
    session,
    event,
):

    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # ------------------------------------------------------------
    # Existing successful cache evidence.
    # ------------------------------------------------------------

    cached = session["cache"][node].get(
        key
    )

    if cached is not None:

        if status == "succeeded":

            if (
                event["artifactDigest"]
                == cached["artifactDigest"]
            ):
                return "ignore", None

            return (
                "conflict",
                "EVIDENCE_CONFLICT",
            )

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # Existing live state.
    # ------------------------------------------------------------

    current = session["state"][node]

    # ------------------------------------------------------------
    # No state.
    # ------------------------------------------------------------

    if current is None:

        # Only started(1) can create state.
        if (
            status == "started"
            and attempt == 1
        ):

            session["state"][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return "accept", None

        # Completion or attempt > 1 without start.
        return "ignore", None

    # Different key is stale/non-current.
    if current["key"] != key:
        return "ignore", None

    old_status = current["status"]
    old_attempt = current["attempt"]

    # ------------------------------------------------------------
    # started(n)
    # ------------------------------------------------------------

    if old_status == "started":

        if attempt < old_attempt:
            return "ignore", None

        if (
            attempt == old_attempt
            and status == "succeeded"
        ):
            store_success(
                session,
                event,
            )
            return "accept", None

        if (
            attempt == old_attempt
            and status in {
                "retryable_failed",
                "terminal_failed",
            }
        ):

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

    if old_status == "retryable_failed":

        if attempt < old_attempt:
            return "ignore", None

        if (
            status == "started"
            and attempt == old_attempt + 1
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
    # terminal_failed is final.
    # ------------------------------------------------------------

    if old_status == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # succeeded is final.
    # ------------------------------------------------------------

    if old_status == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ================================================================
# BUILD RESPONSE
# ================================================================

def build_response_nodes(
    session,
    inputs,
):

    artifacts, keys = recover(
        session,
        inputs,
    )

    result = []

    # The first unresolved node controls descendants.
    upstream_reason = None
    upstream_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # --------------------------------------------------------
        # Key unavailable because parent isn't reusable.
        # --------------------------------------------------------

        if key is None:

            reason = (
                "UPSTREAM_TERMINAL"
                if upstream_reason
                == "TERMINAL_FAILURE"
                else "UPSTREAM_PENDING"
            )

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    upstream_ids,
            })

            continue

        # --------------------------------------------------------
        # Cache hit.
        # --------------------------------------------------------

        cached = session["cache"][node].get(
            key
        )

        if cached is not None:

            result.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [
                    cached["eventId"]
                ],
            })

            continue

        # --------------------------------------------------------
        # Cache miss.
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

        result.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests":
                dependency_digests(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds":
                trigger_ids,
        })

        # Everything after a non-reusable node is blocked.
        upstream_reason = reason
        upstream_ids = trigger_ids

        for descendant in DAG[index + 1:]:

            result.append({
                "node": descendant,
                "action": "block",
                "reasonCodes": [
                    (
                        "UPSTREAM_TERMINAL"
                        if reason
                        == "TERMINAL_FAILURE"
                        else "UPSTREAM_PENDING"
                    )
                ],
                "dependencyDigests":
                    dependency_digests(
                        descendant,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    trigger_ids,
            })

        break

    return result


# ================================================================
# ENDPOINT
# ================================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return conflict(
            "INVALID_REQUEST"
        )

    if not validate_request(body):
        return conflict(
            "INVALID_REQUEST"
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Event structure is validated before any mutation.
    for event in events:

        if not validate_event(event):
            return conflict(
                "INVALID_EVENT"
            )

    with LOCK:

        existing = SESSIONS.get(
            session_id
        )

        if existing is None:
            existing = new_session()

        # ========================================================
        # OLD REVISION
        # ========================================================

        if (
            existing["revision"] is not None
            and revision
            < existing["revision"]
        ):

            old_inputs = json.loads(
                existing["inputs"]
            )

            return {
                "revision":
                    existing["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": [
                    event["eventId"]
                    for event in events
                ],
                "nodes":
                    build_response_nodes(
                        existing,
                        old_inputs,
                    ),
            }

        # ========================================================
        # SAME REVISION MUST HAVE IDENTICAL INPUTS
        # ========================================================

        if (
            existing["revision"] is not None
            and revision
            == existing["revision"]
        ):

            if (
                compact(inputs)
                != existing["inputs"]
            ):
                return conflict(
                    "REVISION_CONFLICT"
                )

        # ========================================================
        # ATOMIC WORKING COPY
        # ========================================================

        working = copy.deepcopy(
            existing
        )

        # ========================================================
        # NEW REVISION
        # ========================================================

        if (
            working["revision"] is None
            or revision
            > working["revision"]
        ):

            # Successful cache survives revision changes.
            preserved_cache = copy.deepcopy(
                working["cache"]
            )

            # Everything else is reset.
            working = new_session()

            working["revision"] = revision
            working["inputs"] = compact(
                inputs
            )

            working["cache"] = (
                preserved_cache
            )

        else:

            working["revision"] = revision
            working["inputs"] = compact(
                inputs
            )

        accepted = []
        ignored = []

        # ========================================================
        # PROCESS BATCH IN INPUT ORDER
        # ========================================================

        for event in events:

            event_id = event["eventId"]

            # ----------------------------------------------------
            # Old/wrong revision.
            # ----------------------------------------------------

            if event["revision"] != revision:

                ignored.append(event_id)
                continue

            # ----------------------------------------------------
            # Unknown node.
            # ----------------------------------------------------

            if event["node"] not in DAG:

                ignored.append(event_id)
                continue

            canonical = event_canonical(
                event
            )

            # ----------------------------------------------------
            # Event ID replay.
            # ----------------------------------------------------

            if event_id in working["events"]:

                if (
                    working["events"][event_id]
                    == canonical
                ):

                    ignored.append(
                        event_id
                    )
                    continue

                return conflict(
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

                ignored.append(
                    event_id
                )
                continue

            # ----------------------------------------------------
            # Recalculate exact current key.
            # ----------------------------------------------------

            artifacts, keys = recover(
                working,
                inputs,
            )

            expected = keys.get(node)

            if expected is None:

                ignored.append(
                    event_id
                )
                continue

            # Wrong/stale content key.
            if event["key"] != expected:

                ignored.append(
                    event_id
                )
                continue

            # ----------------------------------------------------
            # Apply state transition.
            # ----------------------------------------------------

            outcome, code = apply_event(
                working,
                event,
            )

            if outcome == "conflict":

                # Entire working copy is discarded.
                return conflict(code)

            if outcome == "accept":

                working["events"][
                    event_id
                ] = canonical

                accepted.append(
                    event_id
                )

            else:

                # Ignored IDs are NOT consumed.
                ignored.append(
                    event_id
                )

        # ========================================================
        # COMMIT ONLY AFTER WHOLE BATCH SUCCEEDS
        # ========================================================

        SESSIONS[session_id] = working

        return {
            "revision":
                working["revision"],
            "acceptedEventIds":
                accepted,
            "ignoredEventIds":
                ignored,
            "nodes":
                build_response_nodes(
                    working,
                    inputs,
                ),
        }


def parent_is_reusable(
    session,
    inputs,
    node,
):

    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = recover(
        session,
        inputs,
    )

    return (
        parent in artifacts
        and keys.get(parent) is not None
    )


# ================================================================
# HEALTH
# ================================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service":
            "content-addressed-ml-pipeline",
    }
