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

STATUSES = {
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
SESSIONS = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def sha256(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def key_for(values):
    # EXACTLY:
    # lowercase SHA-256 over UTF-8 compact JSON array.
    return sha256(compact(values))


def safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= SAFE_INT_MAX
    )


def string(value):
    return isinstance(value, str) and len(value) > 0


def err(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# SESSION
# ============================================================

def fresh_session():
    return {
        "revision": None,
        "inputs": None,

        # IDs consumed by accepted events only.
        "eventCanonical": {},
        "eventData": {},

        # Current execution state.
        "state": {
            node: None
            for node in DAG
        },

        # Successful immutable cache.
        "cache": {
            node: {}
            for node in DAG
        },
    }


# ============================================================
# REQUEST VALIDATION
# ============================================================

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

    if not string(body["session"]):
        return False

    if not safe_int(body["revision"]):
        return False

    if not isinstance(body["inputs"], dict):
        return False

    if not isinstance(body["events"], list):
        return False

    for name in INPUTS:
        if not string(body["inputs"].get(name)):
            return False

    return True


def valid_event(event):
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not string(event["eventId"]):
        return False

    if not safe_int(event["revision"]):
        return False

    if event["node"] not in DAG:
        return False

    if not safe_int(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if not string(event["key"]):
        return False

    if event["status"] == "succeeded":
        if not string(event["artifactDigest"]):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    if (
        event["node"] in ("register", "publish")
        and event["status"] == "succeeded"
    ):
        required = (
            "receipt:"
            + event["node"]
            + ":"
            + event["key"]
        )

        if event["receiptId"] != required:
            return False
    else:
        if event["receiptId"] is not None:
            return False

    return True


def event_canonical(event):
    # Event field order is fixed by the specification.
    return compact({
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest": event["artifactDigest"],
        "receiptId": event["receiptId"],
    })


def inputs_canonical(inputs):
    # Preserve complete input metadata.
    # Extra metadata participates in same-revision identity.
    return compact(inputs)


# ============================================================
# CONTENT-ADDRESSED CACHE
# ============================================================

def node_key(node, inputs, reusable):
    """
    A node's key is calculated only when its parent has a
    reusable successful artifact.
    """

    if node == "verify_data":
        return key_for([
            inputs["generation"],
            inputs["checksum"],
        ])

    if node == "prepare":
        parent = reusable.get("verify_data")

        if parent is None:
            return None

        return key_for([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    if node == "train":
        parent = reusable.get("prepare")

        if parent is None:
            return None

        return key_for([
            parent,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    if node == "evaluate":
        parent = reusable.get("train")

        if parent is None:
            return None

        return key_for([
            parent,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    if node == "register":
        parent = reusable.get("evaluate")

        if parent is None:
            return None

        return key_for([
            parent,
            inputs["schemaDigest"],
        ])

    if node == "publish":
        parent = reusable.get("register")

        if parent is None:
            return None

        return key_for([
            parent,
            inputs["publishConfig"],
        ])

    return None


def recover_cache(session, inputs):
    """
    Walk the fixed DAG from the beginning.

    A cache entry is reusable only if:
      1. its key can currently be computed, and
      2. the successful cache entry exists for that exact key.

    The first cache miss stops the chain.
    """

    reusable = {}
    keys = {}

    for node in DAG:
        key = node_key(
            node,
            inputs,
            reusable,
        )

        keys[node] = key

        if key is None:
            break

        entry = session["cache"][node].get(key)

        if entry is None:
            break

        reusable[node] = entry["artifactDigest"]

    return reusable, keys


# ============================================================
# DEPENDENCY RESPONSE
# ============================================================

def dependencies(node, inputs, reusable):
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
            "prepareArtifact": reusable.get(
                "prepare"
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    if node == "evaluate":
        return {
            "trainArtifact": reusable.get(
                "train"
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    if node == "register":
        return {
            "evaluateArtifact": reusable.get(
                "evaluate"
            ),
            "schemaDigest": inputs["schemaDigest"],
        }

    if node == "publish":
        return {
            "registerArtifact": reusable.get(
                "register"
            ),
            "publishConfig": inputs["publishConfig"],
        }

    return {}


# ============================================================
# EVENT PROCESSING
# ============================================================

def parent_ready(session, inputs, node):
    parent = PARENT[node]

    if parent is None:
        return True

    reusable, keys = recover_cache(
        session,
        inputs,
    )

    parent_key = keys.get(parent)

    if parent_key is None:
        return False

    return (
        parent in reusable
        and parent_key in session["cache"][parent]
    )


def accept_success(session, event):
    node = event["node"]
    key = event["key"]

    # Immutable first successful evidence.
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


def transition(session, event):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    cached = session["cache"][node].get(key)

    # --------------------------------------------------------
    # Immutable successful cache
    # --------------------------------------------------------

    if cached is not None:

        if status == "succeeded":
            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return "conflict", "EVIDENCE_CONFLICT"

            return "ignore", None

        return "conflict", "STATUS_CONFLICT"

    current = session["state"][node]

    # --------------------------------------------------------
    # No current state
    # --------------------------------------------------------

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

        # Success without start and attempt > 1 are ignored.
        return "ignore", None

    # --------------------------------------------------------
    # Different key = stale event
    # --------------------------------------------------------

    if current["key"] != key:
        return "ignore", None

    previous_status = current["status"]
    previous_attempt = current["attempt"]

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

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
                accept_success(
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

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------

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
                "eventId": event["eventId"],
            }

            return "accept", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # succeeded
    # --------------------------------------------------------

    if previous_status == "succeeded":
        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # terminal_failed
    # --------------------------------------------------------

    if previous_status == "terminal_failed":
        return "conflict", "STATUS_CONFLICT"

    return "conflict", "STATUS_CONFLICT"


# ============================================================
# RESPONSE NODES
# ============================================================

def make_nodes(session, inputs):
    reusable, keys = recover_cache(
        session,
        inputs,
    )

    result = []

    blocked_reason = None
    blocked_ids = []

    for node in DAG:

        key = keys.get(node)

        # ----------------------------------------------------
        # Parent not reusable => key must be null.
        # ----------------------------------------------------

        if key is None:

            if blocked_reason == "TERMINAL_FAILURE":
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            deps = dependencies(
                node,
                inputs,
                reusable,
            )

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests": deps,
                "triggeringEventIds": blocked_ids,
            })

            continue

        # ----------------------------------------------------
        # Successful cache
        # ----------------------------------------------------

        cached = session["cache"][node].get(key)

        if cached is not None:

            deps = dependencies(
                node,
                inputs,
                reusable,
            )

            deps["cacheKey"] = key

            result.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": ["CACHE_HIT"],
                "dependencyDigests": deps,
                "triggeringEventIds": [
                    cached["eventId"]
                ],
            })

            # Recover this artifact for downstream nodes.
            reusable[node] = cached[
                "artifactDigest"
            ]

            # Recalculate downstream keys.
            for later in DAG:
                keys[later] = node_key(
                    later,
                    inputs,
                    reusable,
                )

            continue

        # ----------------------------------------------------
        # No cache: inspect current execution state.
        # ----------------------------------------------------

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

        deps = dependencies(
            node,
            inputs,
            reusable,
        )

        deps["cacheKey"] = key

        result.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests": deps,
            "triggeringEventIds": ids,
        })

        # All later nodes are now blocked by this node.
        blocked_reason = reason
        blocked_ids = ids

        # Continue creating exact DAG output.
        for later in DAG[
            DAG.index(node) + 1:
        ]:
            deps2 = dependencies(
                later,
                inputs,
                reusable,
            )

            result.append({
                "node": later,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                    if reason
                    == "TERMINAL_FAILURE"
                    else "UPSTREAM_PENDING"
                ],
                "dependencyDigests": deps2,
                "triggeringEventIds": ids,
            })

        break

    return result


# ============================================================
# PIPELINE ENDPOINT
# ============================================================

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

    # Validate every event before touching state.
    for event in events:
        if not valid_event(event):
            return err("INVALID_EVENT")

    with LOCK:

        current = SESSIONS.get(
            session_id
        )

        if current is None:
            current = fresh_session()

        # ====================================================
        # REVISION HANDLING
        # ====================================================

        if current["revision"] is not None:

            if revision < current["revision"]:

                # Older well-formed events are ignored.
                ignored = [
                    e["eventId"]
                    for e in events
                ]

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
                        "ignoredEventIds": ignored,
                        "nodes": make_nodes(
                            current,
                            old_inputs,
                        ),
                    },
                )

            if revision == current["revision"]:

                if (
                    inputs_canonical(inputs)
                    != current["inputs"]
                ):
                    return err(
                        "REVISION_CONFLICT"
                    )

        # ====================================================
        # ATOMIC WORKING COPY
        # ====================================================

        working = copy.deepcopy(current)

        # ====================================================
        # NEW REVISION
        # ====================================================

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            # Successful cache survives.
            cache = copy.deepcopy(
                working["cache"]
            )

            # Everything execution-specific is reset.
            working = fresh_session()

            working["revision"] = revision
            working["inputs"] = (
                inputs_canonical(inputs)
            )
            working["cache"] = cache

        else:
            working["revision"] = revision
            working["inputs"] = (
                inputs_canonical(inputs)
            )

        accepted = []
        ignored = []

        # ====================================================
        # EVENTS IN INPUT ORDER
        # ====================================================

        for event in events:

            event_id = event["eventId"]

            # Older/future revision is ignored.
            if event["revision"] != revision:
                ignored.append(event_id)
                continue

            canonical = event_canonical(event)

            # =================================================
            # GLOBAL EVENT ID
            # =================================================

            if event_id in working[
                "eventCanonical"
            ]:

                if (
                    working[
                        "eventCanonical"
                    ][event_id]
                    == canonical
                ):
                    ignored.append(event_id)
                    continue

                return err(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # =================================================
            # PARENT GATE
            # =================================================

            if not parent_ready(
                working,
                inputs,
                node,
            ):
                ignored.append(event_id)
                continue

            # =================================================
            # CURRENT KEY
            # =================================================

            reusable, keys = recover_cache(
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

            # =================================================
            # TRANSITION
            # =================================================

            outcome, code = transition(
                working,
                event,
            )

            if outcome == "conflict":
                # Atomic rollback: SESSIONS is untouched.
                return err(code)

            if outcome == "accept":

                working[
                    "eventCanonical"
                ][event_id] = canonical

                working[
                    "eventData"
                ][event_id] = copy.deepcopy(
                    event
                )

                accepted.append(event_id)

            else:
                # Ignored IDs are deliberately NOT stored.
                ignored.append(event_id)

        # ====================================================
        # ATOMIC COMMIT
        # ====================================================

        SESSIONS[session_id] = working

        return JSONResponse(
            status_code=200,
            content={
                "revision": working["revision"],
                "acceptedEventIds": accepted,
                "ignoredEventIds": ignored,
                "nodes": make_nodes(
                    working,
                    inputs,
                ),
            },
        )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "content-addressed-ml-pipeline",
    }
