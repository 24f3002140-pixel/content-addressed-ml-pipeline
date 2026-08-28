import copy
import hashlib
import json
import os
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

VALID_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

LOCK = threading.RLock()

STATE_FILE = os.environ.get(
    "PIPELINE_STATE_FILE",
    "/tmp/content_addressed_ml_pipeline_state.json",
)


# ============================================================
# JSON / HASH HELPERS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest().lower()


def cache_key(values):
    return sha256_text(compact_json(values))


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


def error_response(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# PERSISTENCE
# ============================================================

def empty_state():
    return {
        "sessions": {}
    }


def load_state():
    try:
        if not os.path.exists(STATE_FILE):
            return empty_state()

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            value = json.load(f)

        if not isinstance(value, dict):
            return empty_state()

        if not isinstance(
            value.get("sessions"),
            dict,
        ):
            return empty_state()

        return value

    except Exception:
        return empty_state()


STORE = load_state()


def save_state():
    directory = os.path.dirname(STATE_FILE)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temporary = STATE_FILE + ".tmp"

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            STORE,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temporary,
        STATE_FILE,
    )


# ============================================================
# SESSION
# ============================================================

def new_session():
    return {
        "revision": None,

        # Complete input object, including extra metadata.
        "inputs": None,

        # eventId -> canonical event JSON
        "eventIds": {},

        # Current execution state.
        "state": {
            node: None
            for node in DAG
        },

        # Successful immutable content-addressed evidence.
        #
        # cache[node][key] = {
        #     "artifactDigest": "...",
        #     "eventId": "..."
        # }
        #
        # Cache survives revision changes.
        "cache": {
            node: {}
            for node in DAG
        },
    }


def get_session(session_id):
    session = STORE["sessions"].get(
        session_id
    )

    if session is None:
        session = new_session()

    return session


# ============================================================
# REQUEST VALIDATION
# ============================================================

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

    if not nonempty_string(
        body["session"]
    ):
        return False

    if not safe_positive_int(
        body["revision"]
    ):
        return False

    if not isinstance(
        body["inputs"],
        dict,
    ):
        return False

    if not isinstance(
        body["events"],
        list,
    ):
        return False

    for name in INPUT_NAMES:
        if not nonempty_string(
            body["inputs"].get(name)
        ):
            return False

    # Extra metadata is allowed.
    return True


def valid_event_shape(event):
    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(
        EVENT_FIELDS
    ):
        return False

    if not nonempty_string(
        event["eventId"]
    ):
        return False

    if not safe_positive_int(
        event["revision"]
    ):
        return False

    if not nonempty_string(
        event["node"]
    ):
        return False

    if not safe_positive_int(
        event["attempt"]
    ):
        return False

    if event["status"] not in VALID_STATUSES:
        return False

    if not nonempty_string(
        event["key"]
    ):
        return False

    # Success requires a non-empty artifact.
    if event["status"] == "succeeded":
        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False
    else:
        # All non-success events require null artifact.
        if event["artifactDigest"] is not None:
            return False

    # Register/publish success requires exact receipt.
    if (
        event["status"] == "succeeded"
        and event["node"]
        in {"register", "publish"}
    ):
        expected_receipt = (
            "receipt:"
            + event["node"]
            + ":"
            + event["key"]
        )

        if event["receiptId"] != expected_receipt:
            return False

    else:
        # Every other event requires null receipt.
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


# ============================================================
# EXACT CONTENT-ADDRESSED DAG
# ============================================================

def dependency_array(
    node,
    inputs,
    reusable_artifacts,
):
    # verify_data
    # [generation, checksum]
    if node == "verify_data":
        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    # prepare
    # [canonicalData, prepareCode, prepareConfig]
    if node == "prepare":
        if "verify_data" not in reusable_artifacts:
            return None

        return [
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ]

    # train
    # [prepareArtifact, trainCode, trainConfig, runtime]
    if node == "train":
        if "prepare" not in reusable_artifacts:
            return None

        return [
            reusable_artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    # evaluate
    # [trainArtifact, canonicalData, evaluateCode, evaluateConfig]
    if node == "evaluate":
        if "train" not in reusable_artifacts:
            return None

        return [
            reusable_artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    # register
    # [evaluateArtifact, schemaDigest]
    if node == "register":
        if "evaluate" not in reusable_artifacts:
            return None

        return [
            reusable_artifacts["evaluate"],
            inputs["schemaDigest"],
        ]

    # publish
    # [registerArtifact, publishConfig]
    if node == "publish":
        if "register" not in reusable_artifacts:
            return None

        return [
            reusable_artifacts["register"],
            inputs["publishConfig"],
        ]

    return None


def recover_chain(
    session,
    inputs,
):
    """
    Recover the reusable successful prefix.

    Once a node has no reusable cache entry,
    all downstream keys are unavailable.
    """

    artifacts = {}
    keys = {}

    for node in DAG:

        deps = dependency_array(
            node,
            inputs,
            artifacts,
        )

        # Parent not reusable.
        if deps is None:
            keys[node] = None
            break

        key = cache_key(deps)
        keys[node] = key

        entry = (
            session["cache"][node]
            .get(key)
        )

        if entry is None:
            break

        artifacts[node] = entry[
            "artifactDigest"
        ]

    return artifacts, keys


# ============================================================
# RESPONSE DEPENDENCY DIGESTS
# ============================================================

def dependency_digests(
    node,
    inputs,
    artifacts,
    key,
):
    if node == "verify_data":
        return {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
            "cacheKey": key,
        }

    if node == "prepare":
        return {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "cacheKey": key,
        }

    if node == "train":
        return {
            "prepareArtifact": artifacts.get(
                "prepare"
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
            "cacheKey": key,
        }

    if node == "evaluate":
        return {
            "trainArtifact": artifacts.get(
                "train"
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
            "cacheKey": key,
        }

    if node == "register":
        return {
            "evaluateArtifact": artifacts.get(
                "evaluate"
            ),
            "schemaDigest": inputs["schemaDigest"],
            "cacheKey": key,
        }

    if node == "publish":
        return {
            "registerArtifact": artifacts.get(
                "register"
            ),
            "publishConfig": inputs["publishConfig"],
            "cacheKey": key,
        }

    return {
        "cacheKey": key
    }


# ============================================================
# PARENT REUSABILITY
# ============================================================

def parent_reusable(
    session,
    inputs,
    node,
):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = recover_chain(
        session,
        inputs,
    )

    return (
        parent in artifacts
        and keys.get(parent) is not None
    )


# ============================================================
# SUCCESSFUL EVIDENCE
# ============================================================

def commit_success(
    session,
    event,
):
    node = event["node"]
    key = event["key"]

    existing = (
        session["cache"][node]
        .get(key)
    )

    # First success permanently binds
    # key -> artifact + eventId.
    if existing is None:
        session["cache"][node][key] = {
            "artifactDigest":
                event["artifactDigest"],
            "eventId":
                event["eventId"],
        }

    # Current state becomes successful.
    session["state"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event["eventId"],
        "artifactDigest":
            event["artifactDigest"],
    }


# ============================================================
# EVENT TRANSITIONS
# ============================================================

def apply_event(
    session,
    event,
):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # --------------------------------------------------------
    # Existing immutable cache evidence.
    # --------------------------------------------------------

    cached = (
        session["cache"][node]
        .get(key)
    )

    if cached is not None:

        if status == "succeeded":

            # Different artifact for an already-bound key.
            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return (
                    "conflict",
                    "EVIDENCE_CONFLICT",
                )

            # IMPORTANT:
            # Exact replay is handled by eventIds
            # before apply_event().
            #
            # Therefore this is a NEW event ID and
            # must conflict.
            return (
                "conflict",
                "STATUS_CONFLICT",
            )

        # Cached successful state cannot transition again.
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # Current live state.
    # --------------------------------------------------------

    current = session["state"][node]

    # --------------------------------------------------------
    # No current state.
    # --------------------------------------------------------

    if current is None:

        # Only started(1) can begin.
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

        # Completion or attempt > 1 without start
        # is ignored.
        return "ignore", None

    # --------------------------------------------------------
    # Different key = stale event.
    # --------------------------------------------------------

    if current["key"] != key:
        return "ignore", None

    old_status = current["status"]
    old_attempt = current["attempt"]

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

    if old_status == "started":

        # Lower attempt is stale.
        if attempt < old_attempt:
            return "ignore", None

        # Same attempt may complete.
        if (
            attempt == old_attempt
            and status == "succeeded"
        ):
            commit_success(
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
                "eventId": event["eventId"],
            }

            return "accept", None

        # Any other transition conflicts.
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------

    if old_status == "retryable_failed":

        # Lower attempt is stale.
        if attempt < old_attempt:
            return "ignore", None

        # Only started(n+1) is legal.
        if (
            status == "started"
            and attempt == old_attempt + 1
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

    # --------------------------------------------------------
    # terminal_failed is final.
    # --------------------------------------------------------

    if old_status == "terminal_failed":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # succeeded is final.
    # --------------------------------------------------------

    if old_status == "succeeded":
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    return (
        "conflict",
        "STATUS_CONFLICT",
    )


# ============================================================
# RESPONSE NODES
# ============================================================

def build_nodes(
    session,
    inputs,
):
    artifacts, keys = recover_chain(
        session,
        inputs,
    )

    nodes = []

    blocked_by = None
    blocked_event_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # Parent-gated key unavailable.
        # ----------------------------------------------------

        if key is None:

            if blocked_by == "TERMINAL_FAILURE":
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            nodes.append({
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
                    list(blocked_event_ids),
            })

            continue

        # ----------------------------------------------------
        # Current key has successful immutable cache.
        # ----------------------------------------------------

        cached = (
            session["cache"][node]
            .get(key)
        )

        if cached is not None:

            nodes.append({
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

        # ----------------------------------------------------
        # Current key has no cache.
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

            trigger_ids = [
                current["eventId"]
            ]

        else:
            action = "rerun"
            reason = "CACHE_MISS"
            trigger_ids = []

        nodes.append({
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

        # ----------------------------------------------------
        # Everything downstream is blocked.
        # ----------------------------------------------------

        blocked_by = (
            "TERMINAL_FAILURE"
            if reason == "TERMINAL_FAILURE"
            else "PENDING"
        )

        blocked_event_ids = list(
            trigger_ids
        )

        for descendant in DAG[index + 1:]:

            descendant_reason = (
                "UPSTREAM_TERMINAL"
                if blocked_by == "TERMINAL_FAILURE"
                else "UPSTREAM_PENDING"
            )

            nodes.append({
                "node": descendant,
                "action": "block",
                "reasonCodes": [
                    descendant_reason
                ],
                "dependencyDigests":
                    dependency_digests(
                        descendant,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(blocked_event_ids),
            })

        break

    return nodes


# ============================================================
# PIPELINE ENDPOINT
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    # --------------------------------------------------------
    # Parse JSON.
    # --------------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return error_response(
            "INVALID_REQUEST"
        )

    # --------------------------------------------------------
    # Validate request.
    # --------------------------------------------------------

    if not valid_request(body):
        return error_response(
            "INVALID_REQUEST"
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    incoming_events = body["events"]

    # --------------------------------------------------------
    # Validate all event shapes before mutation.
    # --------------------------------------------------------

    for event in incoming_events:
        if not valid_event_shape(event):
            return error_response(
                "INVALID_EVENT"
            )

    with LOCK:

        existing = get_session(
            session_id
        )

        # ====================================================
        # OLDER REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision < existing["revision"]
        ):
            response_nodes = build_nodes(
                existing,
                existing["inputs"],
            )

            return {
                "revision":
                    existing["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": [
                    event["eventId"]
                    for event in incoming_events
                ],
                "nodes": response_nodes,
            }

        # ====================================================
        # SAME REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision == existing["revision"]
        ):

            # Entire input object must match exactly,
            # including extra metadata.
            if (
                existing["inputs"]
                != inputs
            ):
                return error_response(
                    "REVISION_CONFLICT"
                )

        # ====================================================
        # ATOMIC WORKING COPY
        # ====================================================

        working = copy.deepcopy(
            existing
        )

        # ====================================================
        # NEW REVISION
        #
        # Replace inputs.
        # Clear live state.
        # Preserve successful cache.
        # Preserve global event IDs.
        # ====================================================

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            preserved_cache = copy.deepcopy(
                working["cache"]
            )

            preserved_event_ids = copy.deepcopy(
                working["eventIds"]
            )

            working = new_session()

            working["revision"] = revision

            working["inputs"] = copy.deepcopy(
                inputs
            )

            working["cache"] = (
                preserved_cache
            )

            working["eventIds"] = (
                preserved_event_ids
            )

        else:
            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )

        accepted_ids = []
        ignored_ids = []

        # ====================================================
        # PROCESS EVENTS IN INPUT ORDER
        # ====================================================

        for event in incoming_events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # Wrong revision = ignored.
            # ------------------------------------------------

            if event["revision"] != revision:
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # Wrong/unknown node = ignored.
            # ------------------------------------------------

            if event["node"] not in DAG:
                ignored_ids.append(
                    event_id
                )
                continue

            canonical = canonical_event(
                event
            )

            # ------------------------------------------------
            # GLOBAL EVENT-ID RULE
            # ------------------------------------------------

            previous = working["eventIds"].get(
                event_id
            )

            if previous is not None:

                # Exact replay.
                if previous == canonical:
                    ignored_ids.append(
                        event_id
                    )
                    continue

                # Same ID but different content.
                return error_response(
                    "EVENT_ID_CONFLICT"
                )

            # ------------------------------------------------
            # Parent must be reusable.
            # ------------------------------------------------

            if not parent_reusable(
                working,
                inputs,
                event["node"],
            ):
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # Compute current expected key.
            # ------------------------------------------------

            artifacts, keys = recover_chain(
                working,
                inputs,
            )

            expected_key = keys.get(
                event["node"]
            )

            # Unavailable/stale key.
            if expected_key is None:
                ignored_ids.append(
                    event_id
                )
                continue

            # Wrong key = stale event.
            if event["key"] != expected_key:
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # Apply state transition.
            # ------------------------------------------------

            outcome, conflict_code = apply_event(
                working,
                event,
            )

            if outcome == "conflict":

                # Do NOT commit working.
                # Entire batch rolls back.
                return error_response(
                    conflict_code
                )

            if outcome == "accept":

                # Only accepted events consume IDs.
                working["eventIds"][
                    event_id
                ] = canonical

                accepted_ids.append(
                    event_id
                )

            else:
                # Ignored events do not consume IDs.
                ignored_ids.append(
                    event_id
                )

        # ====================================================
        # COMMIT
        # ====================================================

        STORE["sessions"][
            session_id
        ] = working

        save_state()

        # ====================================================
        # READ BACK COMMITTED STATE
        # ====================================================

        committed = STORE["sessions"][
            session_id
        ]

        return {
            "revision":
                committed["revision"],
            "acceptedEventIds":
                accepted_ids,
            "ignoredEventIds":
                ignored_ids,
            "nodes":
                build_nodes(
                    committed,
                    committed["inputs"],
                ),
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service":
            "content-addressed-ml-pipeline",
    }
