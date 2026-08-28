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

# ------------------------------------------------------------------
# Persistent process state, isolated by session.
# ------------------------------------------------------------------

SESSIONS = {}


def new_session():
    return {
        "revision": None,
        "inputs": None,

        # Only accepted event IDs are stored here.
        "events": {},

        # Current execution state.
        "state": {
            node: None
            for node in DAG
        },

        # Immutable successful content-addressed evidence.
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


# ------------------------------------------------------------------
# JSON / hashing
# ------------------------------------------------------------------

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
    return sha256_utf8(
        compact_json(values)
    )


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


def conflict(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ------------------------------------------------------------------
# Request validation
# ------------------------------------------------------------------

def valid_request(body):
    if not isinstance(body, dict):
        return False

    if set(body.keys()) != {
        "session",
        "revision",
        "inputs",
        "events",
    }:
        return False

    if not nonempty_string(body["session"]):
        return False

    if not safe_positive_int(body["revision"]):
        return False

    if not isinstance(body["inputs"], dict):
        return False

    if not isinstance(body["events"], list):
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

    if not nonempty_string(event["node"]):
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

    # Receipt is required only for successful register/publish.
    if (
        event["node"] in ("register", "publish")
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
    # Complete object, including any extra metadata.
    return compact_json(inputs)


# ------------------------------------------------------------------
# Exact dependency arrays.
#
# These are deliberately kept in the specification's order.
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Current reusable cache chain.
#
# A downstream key is NEVER calculated until the parent has a
# successful reusable cache entry.
# ------------------------------------------------------------------

def recover(session, inputs):
    artifacts = {}
    keys = {}

    for node in DAG:

        parent = PARENT[node]

        if parent is not None:
            # Parent must have been successfully recovered.
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


# ------------------------------------------------------------------
# Response dependency object.
# ------------------------------------------------------------------

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

    # cacheKey comes after all named dependencies.
    if key is not None:
        result["cacheKey"] = key

    return result


# ------------------------------------------------------------------
# Parent gating.
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Successful immutable evidence.
# ------------------------------------------------------------------

def save_success(session, event):
    node = event["node"]
    key = event["key"]

    # First evidence wins permanently.
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


# ------------------------------------------------------------------
# State transition.
# ------------------------------------------------------------------

def transition(session, event):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    cached = session["cache"][node].get(key)

    # --------------------------------------------------------------
    # Existing successful immutable evidence.
    # --------------------------------------------------------------

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

            # Exact successful evidence is replayable.
            return "ignore", None

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    current = session["state"][node]

    # --------------------------------------------------------------
    # No current attempt.
    # --------------------------------------------------------------

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

        # Completion without started(1), or attempt > 1:
        # ignored.
        return "ignore", None

    # Different key is stale.
    if current["key"] != key:
        return "ignore", None

    previous = current["status"]
    previous_attempt = current["attempt"]

    # --------------------------------------------------------------
    # started(n)
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # succeeded / terminal_failed
    # --------------------------------------------------------------

    if previous == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    if previous == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ------------------------------------------------------------------
# Response generation.
# ------------------------------------------------------------------

def build_nodes(session, inputs):
    artifacts, keys = recover(
        session,
        inputs,
    )

    result = []

    upstream_reason = None
    upstream_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------------
        # Parent-gated key is unavailable.
        # ----------------------------------------------------------

        if key is None:

            if upstream_reason == "TERMINAL_FAILURE":
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

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
                    upstream_ids,
            })

            continue

        # ----------------------------------------------------------
        # Successful cache.
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Current execution state.
        # ----------------------------------------------------------

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

        # Every descendant is now upstream-blocked.
        upstream_reason = reason
        upstream_ids = ids

        for later in DAG[index + 1:]:
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
                        None,
                    ),
                "triggeringEventIds": ids,
            })

        break

    return result


# ------------------------------------------------------------------
# POST /pipeline
# ------------------------------------------------------------------

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

    # Validate all event structures before beginning transaction.
    for event in events:
        if not valid_event(event):
            return err("INVALID_EVENT")

    with LOCK:

        current = SESSIONS.get(
            session_id
        )

        if current is None:
            current = new_session()

        # ==========================================================
        # REVISION
        # ==========================================================

        if current["revision"] is not None:

            # Older revision:
            # completely ignored, no IDs consumed.
            if revision < current["revision"]:

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

            # Same revision requires identical complete inputs,
            # including extra metadata.
            if revision == current["revision"]:

                if (
                    canonical_inputs(inputs)
                    != current["inputs"]
                ):
                    return err(
                        "REVISION_CONFLICT"
                    )

        # ==========================================================
        # TRANSACTIONAL COPY
        # ==========================================================

        working = copy.deepcopy(current)

        # ==========================================================
        # NEW REVISION
        # ==========================================================

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            # Preserve ONLY successful cache.
            old_cache = copy.deepcopy(
                working["cache"]
            )

            # Clear attempts, terminal state, and event IDs.
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

        # ==========================================================
        # EVENT BATCH
        # ==========================================================

        for event in events:

            event_id = event["eventId"]

            # Wrong revision is ignored.
            if event["revision"] != revision:
                ignored.append(event_id)
                continue

            canonical = event_canonical(event)

            # ======================================================
            # EVENT ID
            # ======================================================

            if event_id in working["events"]:

                if (
                    working["events"][event_id]
                    == canonical
                ):
                    # Exact replay.
                    ignored.append(event_id)
                    continue

                return err(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # Unknown node is ignored rather than creating state.
            if node not in DAG:
                ignored.append(event_id)
                continue

            # ======================================================
            # PARENT GATE
            # ======================================================

            if not parent_reusable(
                working,
                inputs,
                node,
            ):
                ignored.append(event_id)
                continue

            # ======================================================
            # EXACT CURRENT KEY
            # ======================================================

            reusable, keys = recover(
                working,
                inputs,
            )

            expected = keys.get(node)

            if expected is None:
                ignored.append(event_id)
                continue

            if event["key"] != expected:
                ignored.append(event_id)
                continue

            # ======================================================
            # TRANSITION
            # ======================================================

            outcome, code = transition(
                working,
                event,
            )

            if outcome == "conflict":
                # Do NOT commit working.
                return err(code)

            if outcome == "accept":

                working["events"][event_id] = (
                    canonical
                )

                accepted.append(event_id)

            else:
                # Ignored IDs remain unused.
                ignored.append(event_id)

        # ==========================================================
        # ATOMIC COMMIT
        # ==========================================================

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
