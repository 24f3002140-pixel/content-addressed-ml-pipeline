import copy
import hashlib
import json
import os
import threading
from typing import Any

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

# Render's filesystem is ephemeral, but this gives durable readback
# across requests while the instance is alive.
STATE_FILE = os.environ.get(
    "PIPELINE_STATE_FILE",
    "/tmp/content_addressed_ml_pipeline_state.json",
)


# ============================================================
# BASIC HELPERS
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def compact_json_preserve_order(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_array(values: list[Any]) -> str:
    raw = compact_json_preserve_order(values).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().lower()


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def positive_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= MAX_SAFE_INTEGER
    )


def error_response(code: str):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# PERSISTENCE
# ============================================================

def new_session() -> dict:
    return {
        "revision": None,
        "inputs": None,

        # eventId -> canonical compact event JSON
        "events": {},

        # Current execution state for the active revision.
        # node -> state or None
        "states": {
            node: None
            for node in DAG
        },

        # Successful immutable evidence.
        #
        # cache[node][key] = {
        #     "artifactDigest": str,
        #     "eventId": str
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


def new_store() -> dict:
    return {
        "sessions": {}
    }


def load_store() -> dict:
    if not os.path.exists(STATE_FILE):
        return new_store()

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return new_store()

        if not isinstance(data.get("sessions"), dict):
            return new_store()

        # Repair missing structures if an older deployment wrote
        # an incomplete state file.
        for sid, session in data["sessions"].items():
            if not isinstance(session, dict):
                data["sessions"][sid] = new_session()
                continue

            if not isinstance(session.get("events"), dict):
                session["events"] = {}

            if not isinstance(session.get("states"), dict):
                session["states"] = {
                    node: None
                    for node in DAG
                }

            if not isinstance(session.get("cache"), dict):
                session["cache"] = {
                    node: {}
                    for node in DAG
                }

            for node in DAG:
                if not isinstance(
                    session["cache"].get(node),
                    dict,
                ):
                    session["cache"][node] = {}

                if node not in session["states"]:
                    session["states"][node] = None

        return data

    except Exception:
        return new_store()


STORE = load_store()


def save_store() -> None:
    directory = os.path.dirname(STATE_FILE)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temp = STATE_FILE + ".tmp"

    with open(
        temp,
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
        temp,
        STATE_FILE,
    )


# ============================================================
# REQUEST VALIDATION
# ============================================================

def valid_request(body: Any) -> bool:
    if not isinstance(body, dict):
        return False

    if not nonempty_string(body.get("session")):
        return False

    if not positive_safe_integer(
        body.get("revision")
    ):
        return False

    inputs = body.get("inputs")

    if not isinstance(inputs, dict):
        return False

    for name in INPUTS:
        if not nonempty_string(
            inputs.get(name)
        ):
            return False

    events = body.get("events")

    if not isinstance(events, list):
        return False

    return True


def valid_event_shape(event: Any) -> bool:
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not positive_safe_integer(
        event["revision"]
    ):
        return False

    if not nonempty_string(event["node"]):
        return False

    if not positive_safe_integer(
        event["attempt"]
    ):
        return False

    if event["status"] not in STATUSES:
        return False

    if not nonempty_string(event["key"]):
        return False

    status = event["status"]
    node = event["node"]

    if status == "succeeded":
        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register/publish success needs exact receipt.
    if node in ("register", "publish"):
        if status == "succeeded":
            expected = (
                "receipt:"
                + node
                + ":"
                + event["key"]
            )

            if event["receiptId"] != expected:
                return False
        else:
            if event["receiptId"] is not None:
                return False
    else:
        if event["receiptId"] is not None:
            return False

    return True


def canonical_event(event: dict) -> str:
    # Event canonicalization uses all eight fields.
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


# ============================================================
# EXACT CONTENT-ADDRESSED DAG
# ============================================================

def dependency_array(
    node: str,
    inputs: dict,
    artifacts: dict,
):
    """
    EXACT arrays required by the specification.

    Important:
    A downstream node has no key until its parent has a
    reusable artifact.
    """

    if node == "verify_data":
        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    if node == "prepare":
        if artifacts.get("verify_data") is None:
            return None

        return [
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ]

    if node == "train":
        if artifacts.get("prepare") is None:
            return None

        return [
            artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    if node == "evaluate":
        if artifacts.get("train") is None:
            return None

        return [
            artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    if node == "register":
        if artifacts.get("evaluate") is None:
            return None

        return [
            artifacts["evaluate"],
            inputs["schemaDigest"],
        ]

    if node == "publish":
        if artifacts.get("register") is None:
            return None

        return [
            artifacts["register"],
            inputs["publishConfig"],
        ]

    return None


def calculate_key(
    node: str,
    inputs: dict,
    artifacts: dict,
):
    values = dependency_array(
        node,
        inputs,
        artifacts,
    )

    if values is None:
        return None

    return sha256_array(values)


# ============================================================
# CACHE WALK
# ============================================================

def reusable_prefix(
    session: dict,
    inputs: dict,
):
    """
    Walk from the root.

    A node contributes an artifact only if its exact key is
    already present in immutable successful cache.

    Once a node misses, every downstream key is null.
    """

    artifacts = {}
    keys = {}

    blocked = False

    for node in DAG:

        if blocked:
            keys[node] = None
            continue

        key = calculate_key(
            node,
            inputs,
            artifacts,
        )

        keys[node] = key

        if key is None:
            blocked = True
            continue

        entry = session["cache"][node].get(key)

        if entry is None:
            blocked = True
            continue

        artifacts[node] = entry[
            "artifactDigest"
        ]

    return artifacts, keys


def key_for_node(
    session: dict,
    inputs: dict,
    target: str,
):
    """
    Compute a target key only when all of its parents have
    reusable artifacts.
    """

    artifacts = {}

    for node in DAG:

        key = calculate_key(
            node,
            inputs,
            artifacts,
        )

        if node == target:
            return key

        if key is None:
            return None

        entry = session["cache"][node].get(key)

        if entry is None:
            return None

        artifacts[node] = entry[
            "artifactDigest"
        ]

    return None


def cached_artifact(
    session: dict,
    node: str,
    key: str,
):
    if key is None:
        return None

    return session["cache"][node].get(key)


# ============================================================
# RESPONSE DEPENDENCY OBJECTS
# ============================================================

def dependency_digests(
    node: str,
    inputs: dict,
    artifacts: dict,
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
# CURRENT STATE HELPERS
# ============================================================

def current_state(
    session: dict,
    node: str,
    key: str,
):
    state = session["states"].get(node)

    if state is None:
        return None

    if state.get("key") != key:
        return None

    return state


def parent_reusable(
    session: dict,
    inputs: dict,
    node: str,
):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = reusable_prefix(
        session,
        inputs,
    )

    return (
        keys.get(parent) is not None
        and parent in artifacts
    )


# ============================================================
# APPLY SUCCESS
# ============================================================

def store_success(
    session: dict,
    event: dict,
):
    node = event["node"]
    key = event["key"]

    existing = session["cache"][node].get(
        key
    )

    if existing is None:
        session["cache"][node][key] = {
            "artifactDigest":
                event["artifactDigest"],
            "eventId":
                event["eventId"],
        }

    session["states"][node] = {
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
    session: dict,
    event: dict,
):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # --------------------------------------------------------
    # Immutable successful evidence.
    # --------------------------------------------------------

    cached = session["cache"][node].get(
        key
    )

    if cached is not None:

        if status == "succeeded":

            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return "conflict", "EVIDENCE_CONFLICT"

            # Same immutable evidence is still not a new
            # valid state transition.
            return "conflict", "STATUS_CONFLICT"

        return "conflict", "STATUS_CONFLICT"

    state = current_state(
        session,
        node,
        key,
    )

    # --------------------------------------------------------
    # No state for this key.
    # --------------------------------------------------------

    if state is None:

        if (
            status == "started"
            and attempt == 1
        ):
            session["states"][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        # Completion / retry / terminal without
        # started(1) is ignored.
        return "ignore", None

    old_status = state["status"]
    old_attempt = state["attempt"]

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

    if old_status == "started":

        if attempt < old_attempt:
            return "ignore", None

        if (
            status == "succeeded"
            and attempt == old_attempt
        ):
            store_success(
                session,
                event,
            )
            return "accept", None

        if (
            status == "retryable_failed"
            and attempt == old_attempt
        ):
            session["states"][node] = {
                "status":
                    "retryable_failed",
                "attempt":
                    old_attempt,
                "key":
                    key,
                "eventId":
                    event["eventId"],
            }

            return "accept", None

        if (
            status == "terminal_failed"
            and attempt == old_attempt
        ):
            session["states"][node] = {
                "status":
                    "terminal_failed",
                "attempt":
                    old_attempt,
                "key":
                    key,
                "eventId":
                    event["eventId"],
            }

            return "accept", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # retryable_failed(n)
    # --------------------------------------------------------

    if old_status == "retryable_failed":

        if attempt < old_attempt:
            return "ignore", None

        if (
            status == "started"
            and attempt == old_attempt + 1
        ):
            session["states"][node] = {
                "status": "started",
                "attempt": attempt,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # terminal_failed
    # --------------------------------------------------------

    if old_status == "terminal_failed":
        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # succeeded should normally be represented by cache.
    # --------------------------------------------------------

    if old_status == "succeeded":
        return "conflict", "STATUS_CONFLICT"

    return "conflict", "STATUS_CONFLICT"


# ============================================================
# RESPONSE CONSTRUCTION
# ============================================================

def make_nodes(
    session: dict,
    inputs: dict,
):
    artifacts, keys = reusable_prefix(
        session,
        inputs,
    )

    result = []

    pending_node = None
    terminal_node = None
    trigger_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # If an earlier node is pending/terminal, all
        # descendants are blocked.
        # ----------------------------------------------------

        if pending_node is not None:
            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(trigger_ids),
            })
            continue

        if terminal_node is not None:
            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(trigger_ids),
            })
            continue

        # ----------------------------------------------------
        # No parent => root can always have a key.
        # ----------------------------------------------------

        if key is None:
            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds": [],
            })

            pending_node = node
            continue

        # ----------------------------------------------------
        # Successful immutable cache.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # No cache: inspect active state.
        # ----------------------------------------------------

        state = current_state(
            session,
            node,
            key,
        )

        if state is None:
            result.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": [
                    "CACHE_MISS"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [],
            })

            pending_node = node
            continue

        if state["status"] == "started":
            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "RUNNING"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [
                    state["eventId"]
                ],
            })

            pending_node = node
            trigger_ids = [
                state["eventId"]
            ]
            continue

        if state["status"] == "retryable_failed":
            result.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": [
                    "RETRYABLE_FAILURE"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [
                    state["eventId"]
                ],
            })

            pending_node = node
            trigger_ids = [
                state["eventId"]
            ]
            continue

        if state["status"] == "terminal_failed":
            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "TERMINAL_FAILURE"
                ],
                "dependencyDigests":
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        key,
                    ),
                "triggeringEventIds": [
                    state["eventId"]
                ],
            })

            terminal_node = node
            trigger_ids = [
                state["eventId"]
            ]
            continue

        result.append({
            "node": node,
            "action": "rerun",
            "reasonCodes": [
                "CACHE_MISS"
            ],
            "dependencyDigests":
                dependency_digests(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds": [],
        })

        pending_node = node

    return result


# ============================================================
# PIPELINE ENDPOINT
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return error_response(
            "INVALID_REQUEST"
        )

    if not valid_request(body):
        return error_response(
            "INVALID_REQUEST"
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Validate event shape before any mutation.
    for event in events:
        if not valid_event_shape(event):
            return error_response(
                "INVALID_EVENT"
            )

    with LOCK:

        if session_id in STORE["sessions"]:
            original = STORE["sessions"][session_id]
        else:
            original = new_session()

        # ----------------------------------------------------
        # Older revision:
        # valid events are ignored and do not consume IDs.
        # ----------------------------------------------------

        if (
            original["revision"] is not None
            and revision < original["revision"]
        ):
            ignored = [
                event["eventId"]
                for event in events
            ]

            return {
                "revision":
                    original["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds":
                    ignored,
                "nodes":
                    make_nodes(
                        original,
                        original["inputs"],
                    ),
            }

        # ----------------------------------------------------
        # Same revision:
        # all inputs, including extra metadata, must match.
        # ----------------------------------------------------

        if (
            original["revision"] is not None
            and revision == original["revision"]
        ):
            if (
                compact_json(original["inputs"])
                != compact_json(inputs)
            ):
                return error_response(
                    "REVISION_CONFLICT"
                )

        # ----------------------------------------------------
        # Transaction copy.
        # ----------------------------------------------------

        working = copy.deepcopy(
            original
        )

        # ----------------------------------------------------
        # New revision:
        #
        # KEEP successful cache.
        # KEEP global event IDs.
        # RESET active execution states.
        # ----------------------------------------------------

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            old_cache = copy.deepcopy(
                working["cache"]
            )

            old_events = copy.deepcopy(
                working["events"]
            )

            working = new_session()

            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )
            working["cache"] = old_cache
            working["events"] = old_events

        else:
            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )

        accepted = []
        ignored = []

        # ----------------------------------------------------
        # PROCESS INPUT EVENTS IN ORDER
        # ----------------------------------------------------

        for event in events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # Older/wrong revision event.
            # ------------------------------------------------

            if event["revision"] != revision:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Unknown node is explicitly an ignored event.
            # ------------------------------------------------

            if event["node"] not in DAG:
                ignored.append(event_id)
                continue

            canonical = canonical_event(
                event
            )

            # ------------------------------------------------
            # Global event ID.
            # ------------------------------------------------

            previous = working["events"].get(
                event_id
            )

            if previous is not None:

                if previous == canonical:
                    # Exact replay.
                    # Replay does not consume anything.
                    ignored.append(event_id)
                    continue

                return error_response(
                    "EVENT_ID_CONFLICT"
                )

            node = event["node"]

            # ------------------------------------------------
            # Parent gating.
            #
            # A child event is ignored unless its parent has
            # become reusable in the current content-addressed
            # state.
            # ------------------------------------------------

            if not parent_reusable(
                working,
                inputs,
                node,
            ):
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Current key.
            # ------------------------------------------------

            expected_key = key_for_node(
                working,
                inputs,
                node,
            )

            if expected_key is None:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Stale key.
            # ------------------------------------------------

            if event["key"] != expected_key:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Apply state transition.
            # ------------------------------------------------

            result, code = apply_event(
                working,
                event,
            )

            if result == "conflict":
                # Nothing has been committed because all work
                # has been performed against the transaction copy.
                return error_response(code)

            if result == "accept":

                working["events"][
                    event_id
                ] = canonical

                accepted.append(event_id)

            else:
                # Ignored events do not consume their IDs.
                ignored.append(event_id)

        # ----------------------------------------------------
        # ATOMIC COMMIT
        # ----------------------------------------------------

        STORE["sessions"][
            session_id
        ] = working

        save_store()

        committed = STORE["sessions"][
            session_id
        ]

        return {
            "revision":
                committed["revision"],
            "acceptedEventIds":
                accepted,
            "ignoredEventIds":
                ignored,
            "nodes":
                make_nodes(
                    committed,
                    committed["inputs"],
                ),
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service":
            "content-addressed-ml-pipeline",
    }
