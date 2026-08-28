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


def err(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


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
    return sha256_utf8(compact_json(values))


def safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_INT_MAX
    )


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def new_session():
    return {
        "revision": None,
        "inputs": None,

        "events": {},

        "state": {
            node: None
            for node in DAG
        },

        "cache": {
            node: {}
            for node in DAG
        },
    }


# ================================================================
# REQUEST VALIDATION
# ================================================================

def valid_request(body):
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

    if not nonempty_string(body.get("session")):
        return False

    if not safe_positive_int(body.get("revision")):
        return False

    if not isinstance(body.get("inputs"), dict):
        return False

    if not isinstance(body.get("events"), list):
        return False

    for name in INPUTS:
        if not nonempty_string(
            body["inputs"].get(name)
        ):
            return False

    return True


def valid_event(event):
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not safe_positive_int(event["revision"]):
        return False

    if event["node"] not in DAG:
        return False

    if not safe_positive_int(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
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

    if (
        event["node"] in {
            "register",
            "publish",
        }
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
    return compact_json({
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest": event["artifactDigest"],
        "receiptId": event["receiptId"],
    })


def canonical_inputs(inputs):
    return compact_json(inputs)


# ================================================================
# EXACT CONTENT-ADDRESSED DEPENDENCY ARRAYS
# ================================================================

def dependency_array(node, inputs, artifacts):

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
        return [
            artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        return [
            artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        return [
            artifacts["evaluate"],
            inputs["schemaDigest"],
        ]

    if node == "publish":
        return [
            artifacts["register"],
            inputs["publishConfig"],
        ]

    return None


# ================================================================
# RECOVER REUSABLE CONTENT-ADDRESSED CHAIN
# ================================================================

def recover(session, inputs):
    artifacts = {}
    keys = {}

    for node in DAG:

        parent = PARENT[node]

        if parent is not None:
            if parent not in artifacts:
                keys[node] = None
                break

        deps = dependency_array(
            node,
            inputs,
            artifacts,
        )

        if deps is None:
            keys[node] = None
            break

        key = content_key(deps)
        keys[node] = key

        cached = session["cache"][node].get(key)

        if cached is None:
            break

        artifacts[node] = cached[
            "artifactDigest"
        ]

    return artifacts, keys


# ================================================================
# RESPONSE DEPENDENCIES
# ================================================================

def dependency_object(
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
# PARENT AVAILABILITY
# ================================================================

def parent_reusable(session, inputs, node):
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
# SUCCESSFUL IMMUTABLE EVIDENCE
# ================================================================

def save_success(session, event):
    node = event["node"]
    key = event["key"]

    if key not in session["cache"][node]:
        session["cache"][node][key] = {
            "artifactDigest": event[
                "artifactDigest"
            ],
            "eventId": event["eventId"],
        }

    session["state"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event["eventId"],
        "artifactDigest": event[
            "artifactDigest"
        ],
    }


# ================================================================
# EVENT TRANSITIONS
# ================================================================

def transition(session, event):

    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    cached = session["cache"][node].get(key)

    # ------------------------------------------------------------
    # Immutable successful evidence
    # ------------------------------------------------------------

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

    current = session["state"][node]

    # ------------------------------------------------------------
    # No previous state
    # ------------------------------------------------------------

    if current is None:

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

        return "ignore", None

    # Different content key is stale.
    if current["key"] != key:
        return "ignore", None

    previous = current["status"]
    previous_attempt = current["attempt"]

    # ------------------------------------------------------------
    # started(n)
    # ------------------------------------------------------------

    if previous == "started":

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
                save_success(
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

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # retryable_failed(n)
    # ------------------------------------------------------------

    if previous == "retryable_failed":

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
                "eventId": event["eventId"],
            }

            return "accept", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # ------------------------------------------------------------
    # Terminal / succeeded are immutable state.
    # ------------------------------------------------------------

    if previous == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    if previous == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ================================================================
# RESPONSE
# ================================================================

def build_nodes(session, inputs):

    artifacts, keys = recover(
        session,
        inputs,
    )

    result = []

    blocking_reason = None
    blocking_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # --------------------------------------------------------
        # Parent isn't reusable.
        # --------------------------------------------------------

        if key is None:

            reason = (
                "UPSTREAM_TERMINAL"
                if blocking_reason
                == "TERMINAL_FAILURE"
                else "UPSTREAM_PENDING"
            )

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests":
                    dependency_object(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    blocking_ids,
            })

            continue

        # --------------------------------------------------------
        # Cache hit.
        # --------------------------------------------------------

        cached = session["cache"][node].get(key)

        if cached is not None:

            result.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    dependency_object(
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
        # Current execution state.
        # --------------------------------------------------------

        current = session["state"][node]

        if (
            current is not None
            and current["key"] == key
        ):

            if current["status"] == "started":
                action = "block"
                reason = "RUNNING"

            elif (
                current["status"]
                == "retryable_failed"
            ):
                action = "rerun"
                reason = "RETRYABLE_FAILURE"

            elif (
                current["status"]
                == "terminal_failed"
            ):
                action = "block"
                reason = "TERMINAL_FAILURE"

            else:
                action = "rerun"
                reason = "CACHE_MISS"

            ids = [current["eventId"]]

        else:
            action = "rerun"
            reason = "CACHE_MISS"
            ids = []

        result.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests":
                dependency_object(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds": ids,
        })

        blocking_reason = reason
        blocking_ids = ids

        # --------------------------------------------------------
        # Remaining descendants are blocked.
        # --------------------------------------------------------

        for later in DAG[index + 1:]:

            later_key = keys.get(later)

            result.append({
                "node": later,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                    if reason == "TERMINAL_FAILURE"
                    else "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_object(
                        later,
                        inputs,
                        artifacts,
                        later_key,
                    ),
                "triggeringEventIds": ids,
            })

        break

    return result


# ================================================================
# PIPELINE
# ================================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return err("INVALID_REQUEST")

    if not valid_request(body):
        return err("INVALID_REQUEST")

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Validate event structure before mutating anything.
    for event in events:
        if not valid_event(event):
            return err("INVALID_EVENT")

    with LOCK:

        current = SESSIONS.get(
            session_id
        )

        if current is None:
            current = new_session()

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
                    "revision": current[
                        "revision"
                    ],
                    "acceptedEventIds": [],
                    "ignoredEventIds": [
                        e["eventId"]
                        for e in events
                    ],
                    "nodes": build_nodes(
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

            if (
                canonical_inputs(inputs)
                != current["inputs"]
            ):
                return err(
                    "REVISION_CONFLICT"
                )

        # ========================================================
        # ATOMIC COPY
        # ========================================================

        working = copy.deepcopy(current)

        # ========================================================
        # NEW REVISION
        # ========================================================

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            # Cache survives.
            old_cache = copy.deepcopy(
                working["cache"]
            )

            # Attempt state and event IDs do not.
            working = new_session()

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

        accepted = []
        ignored = []

        # ========================================================
        # PROCESS EVENTS IN INPUT ORDER
        # ========================================================

        for event in events:

            event_id = event["eventId"]

            # Wrong revision = ignore.
            if event["revision"] != revision:
                ignored.append(event_id)
                continue

            canonical = canonical_event(event)

            # ----------------------------------------------------
            # Event ID already known.
            # ----------------------------------------------------

            if event_id in working["events"]:

                if (
                    working["events"][event_id]
                    == canonical
                ):
                    ignored.append(event_id)
                    continue

                return err(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # ----------------------------------------------------
            # Parent must be reusable.
            # ----------------------------------------------------

            if not parent_reusable(
                working,
                inputs,
                node,
            ):
                ignored.append(event_id)
                continue

            # ----------------------------------------------------
            # Event key must be current content key.
            # ----------------------------------------------------

            reusable, keys = recover(
                working,
                inputs,
            )

            expected_key = keys.get(node)

            if expected_key is None:
                ignored.append(event_id)
                continue

            if event["key"] != expected_key:
                ignored.append(event_id)
                continue

            # ----------------------------------------------------
            # Transition.
            # ----------------------------------------------------

            outcome, code = transition(
                working,
                event,
            )

            if outcome == "conflict":
                # Atomic rollback.
                return err(code)

            if outcome == "accept":

                working["events"][event_id] = (
                    canonical
                )

                accepted.append(event_id)

            else:
                # Important: ignored events do NOT consume IDs.
                ignored.append(event_id)

        # ========================================================
        # COMMIT
        # ========================================================

        SESSIONS[session_id] = working

        return JSONResponse(
            status_code=200,
            content={
                "revision": working["revision"],
                "acceptedEventIds": accepted,
                "ignoredEventIds": ignored,
                "nodes": build_nodes(
                    working,
                    inputs,
                ),
            },
        )


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "content-addressed-ml-pipeline",
    }
