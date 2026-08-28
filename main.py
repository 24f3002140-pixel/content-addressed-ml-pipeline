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

REQUIRED_INPUTS = [
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

STATE_FILE = os.environ.get(
    "PIPELINE_STATE_FILE",
    "/tmp/content_addressed_ml_pipeline_state.json",
)


# ============================================================
# JSON / HASH
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def hash_array(values: list[Any]) -> str:
    payload = compact_json(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().lower()


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def safe_positive_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def conflict(code: str):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# PERSISTENCE
# ============================================================

def empty_session() -> dict:
    return {
        "revision": None,
        "inputs": None,

        # Event IDs are global within this session.
        "eventIds": {},

        # Current revision execution state.
        "states": {
            node: None
            for node in DAG
        },

        # Successful immutable cache.
        #
        # cache[node][key] = {
        #     artifactDigest: "...",
        #     eventId: "..."
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


def empty_store() -> dict:
    return {
        "sessions": {}
    }


def load_store() -> dict:
    if not os.path.exists(STATE_FILE):
        return empty_store()

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return empty_store()

        if not isinstance(
            data.get("sessions"),
            dict,
        ):
            return empty_store()

        for session in data["sessions"].values():

            if not isinstance(session, dict):
                continue

            if not isinstance(
                session.get("eventIds"),
                dict,
            ):
                session["eventIds"] = {}

            if not isinstance(
                session.get("states"),
                dict,
            ):
                session["states"] = {
                    node: None
                    for node in DAG
                }

            if not isinstance(
                session.get("cache"),
                dict,
            ):
                session["cache"] = {
                    node: {}
                    for node in DAG
                }

            for node in DAG:

                if node not in session["states"]:
                    session["states"][node] = None

                if not isinstance(
                    session["cache"].get(node),
                    dict,
                ):
                    session["cache"][node] = {}

        return data

    except Exception:
        return empty_store()


STORE = load_store()


def save_store():
    directory = os.path.dirname(
        STATE_FILE
    )

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
# REQUEST VALIDATION
# ============================================================

def validate_request(body: Any) -> bool:

    if not isinstance(body, dict):
        return False

    if not nonempty_string(
        body.get("session")
    ):
        return False

    if not safe_positive_integer(
        body.get("revision")
    ):
        return False

    inputs = body.get("inputs")

    if not isinstance(inputs, dict):
        return False

    events = body.get("events")

    if not isinstance(events, list):
        return False

    for name in REQUIRED_INPUTS:

        if not nonempty_string(
            inputs.get(name)
        ):
            return False

    return True


def validate_event(event: Any) -> bool:

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

    if not safe_positive_integer(
        event["revision"]
    ):
        return False

    if not nonempty_string(
        event["node"]
    ):
        return False

    if not safe_positive_integer(
        event["attempt"]
    ):
        return False

    if event["status"] not in STATUSES:
        return False

    if not nonempty_string(
        event["key"]
    ):
        return False

    status = event["status"]
    node = event["node"]

    # Success requires artifact.
    if status == "succeeded":

        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False

    else:

        if event["artifactDigest"] is not None:
            return False

    # Receipt rules.
    if node in {
        "register",
        "publish",
    }:

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

    # Exact eight fields in specified order.
    ordered = {
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest":
            event["artifactDigest"],
        "receiptId":
            event["receiptId"],
    }

    return compact_json(ordered)


# ============================================================
# EXACT DAG DEPENDENCIES
# ============================================================

def dependency_array(
    node: str,
    inputs: dict,
    artifacts: dict,
):
    """
    EXACT arrays from the assignment.

    Downstream keys are unavailable until the parent
    has a reusable successful artifact.
    """

    if node == "verify_data":

        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    if node == "prepare":

        if "verify_data" not in artifacts:
            return None

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


def node_key(
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

    return hash_array(values)


# ============================================================
# CACHE RECOVERY
# ============================================================

def recover_cache(
    session: dict,
    inputs: dict,
):
    """
    Recover the reusable prefix.

    Example:

    verify_data cache hit
        -> prepare key can exist

    prepare cache miss
        -> train/evaluate/register/publish keys = null
    """

    artifacts = {}
    keys = {}

    stopped = False

    for node in DAG:

        if stopped:
            keys[node] = None
            continue

        key = node_key(
            node,
            inputs,
            artifacts,
        )

        if key is None:

            keys[node] = None
            stopped = True
            continue

        keys[node] = key

        cached = session[
            "cache"
        ][node].get(key)

        if cached is None:

            stopped = True
            continue

        artifacts[node] = cached[
            "artifactDigest"
        ]

    return artifacts, keys


def expected_key(
    session: dict,
    inputs: dict,
    target: str,
):
    """
    Calculate the current key for a target.

    Parent artifacts must come from successful
    immutable cache.
    """

    artifacts = {}

    for node in DAG:

        key = node_key(
            node,
            inputs,
            artifacts,
        )

        if node == target:
            return key

        if key is None:
            return None

        cached = session[
            "cache"
        ][node].get(key)

        if cached is None:
            return None

        artifacts[node] = cached[
            "artifactDigest"
        ]

    return None


# ============================================================
# CACHE / EVIDENCE
# ============================================================

def get_cache(
    session: dict,
    node: str,
    key: str,
):
    return session[
        "cache"
    ][node].get(key)


def commit_success(
    session: dict,
    event: dict,
):

    node = event["node"]
    key = event["key"]

    existing = get_cache(
        session,
        node,
        key,
    )

    if existing is None:

        session[
            "cache"
        ][node][key] = {
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
# STATE MACHINE
# ============================================================

def apply_transition(
    session: dict,
    event: dict,
):

    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    cached = get_cache(
        session,
        node,
        key,
    )

    # --------------------------------------------------------
    # IMMUTABLE SUCCESS
    # --------------------------------------------------------

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

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # CURRENT STATE
    # --------------------------------------------------------

    current = session[
        "states"
    ].get(node)

    # --------------------------------------------------------
    # NO STATE
    # --------------------------------------------------------

    if current is None:

        # Only started(1) establishes execution.
        if (
            status == "started"
            and attempt == 1
        ):

            session[
                "states"
            ][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        # Completion/retry without start.
        return (
            "ignore",
            None,
        )

    # --------------------------------------------------------
    # DIFFERENT KEY
    # --------------------------------------------------------

    if current["key"] != key:

        # The event key is stale for the current execution.
        return (
            "ignore",
            None,
        )

    old_status = current[
        "status"
    ]

    old_attempt = current[
        "attempt"
    ]

    # --------------------------------------------------------
    # STARTED(n)
    # --------------------------------------------------------

    if old_status == "started":

        if attempt < old_attempt:
            return (
                "ignore",
                None,
            )

        if (
            status == "succeeded"
            and attempt == old_attempt
        ):

            commit_success(
                session,
                event,
            )

            return (
                "accept",
                None,
            )

        if (
            status == "retryable_failed"
            and attempt == old_attempt
        ):

            session[
                "states"
            ][node] = {
                "status":
                    "retryable_failed",
                "attempt":
                    old_attempt,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        if (
            status == "terminal_failed"
            and attempt == old_attempt
        ):

            session[
                "states"
            ][node] = {
                "status":
                    "terminal_failed",
                "attempt":
                    old_attempt,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # RETRYABLE_FAILED(n)
    # --------------------------------------------------------

    if old_status == "retryable_failed":

        if attempt < old_attempt:
            return (
                "ignore",
                None,
            )

        if (
            status == "started"
            and attempt
            == old_attempt + 1
        ):

            session[
                "states"
            ][node] = {
                "status": "started",
                "attempt": attempt,
                "key": key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # TERMINAL
    # --------------------------------------------------------

    if old_status == "terminal_failed":

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # SUCCESS
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
# RESPONSE DEPENDENCIES
# ============================================================

def dependency_object(
    node: str,
    inputs: dict,
    artifacts: dict,
    key,
):

    if node == "verify_data":

        return {
            "generation":
                inputs["generation"],
            "checksum":
                inputs["checksum"],
            "cacheKey": key,
        }

    if node == "prepare":

        return {
            "canonicalData":
                inputs["canonicalData"],
            "prepareCode":
                inputs["prepareCode"],
            "prepareConfig":
                inputs["prepareConfig"],
            "cacheKey": key,
        }

    if node == "train":

        return {
            "prepareArtifact":
                artifacts.get("prepare"),
            "trainCode":
                inputs["trainCode"],
            "trainConfig":
                inputs["trainConfig"],
            "runtime":
                inputs["runtime"],
            "cacheKey": key,
        }

    if node == "evaluate":

        return {
            "trainArtifact":
                artifacts.get("train"),
            "canonicalData":
                inputs["canonicalData"],
            "evaluateCode":
                inputs["evaluateCode"],
            "evaluateConfig":
                inputs["evaluateConfig"],
            "cacheKey": key,
        }

    if node == "register":

        return {
            "evaluateArtifact":
                artifacts.get("evaluate"),
            "schemaDigest":
                inputs["schemaDigest"],
            "cacheKey": key,
        }

    if node == "publish":

        return {
            "registerArtifact":
                artifacts.get("register"),
            "publishConfig":
                inputs["publishConfig"],
            "cacheKey": key,
        }

    return {
        "cacheKey": key
    }


# ============================================================
# RESPONSE NODES
# ============================================================

def build_nodes(
    session: dict,
    inputs: dict,
):

    artifacts, keys = recover_cache(
        session,
        inputs,
    )

    result = []

    blocking_reason = None
    blocking_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # DESCENDANT OF PENDING / TERMINAL NODE
        # ----------------------------------------------------

        if blocking_reason is not None:

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    blocking_reason
                ],
                "dependencyDigests":
                    dependency_object(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(blocking_ids),
            })

            continue

        # ----------------------------------------------------
        # KEY UNAVAILABLE
        # ----------------------------------------------------

        if key is None:

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_object(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds": [],
            })

            blocking_reason = (
                "UPSTREAM_PENDING"
            )

            continue

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        cached = get_cache(
            session,
            node,
            key,
        )

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

        # ----------------------------------------------------
        # CURRENT STATE
        # ----------------------------------------------------

        state = session[
            "states"
        ].get(node)

        if (
            state is not None
            and state.get("key") == key
        ):

            if state["status"] == "started":

                result.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "RUNNING"
                    ],
                    "dependencyDigests":
                        dependency_object(
                            node,
                            inputs,
                            artifacts,
                            key,
                        ),
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                blocking_reason = "UPSTREAM_PENDING"
                blocking_ids = [
                    state["eventId"]
                ]

                continue

            if (
                state["status"]
                == "retryable_failed"
            ):

                result.append({
                    "node": node,
                    "action": "rerun",
                    "reasonCodes": [
                        "RETRYABLE_FAILURE"
                    ],
                    "dependencyDigests":
                        dependency_object(
                            node,
                            inputs,
                            artifacts,
                            key,
                        ),
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                blocking_reason = (
                    "UPSTREAM_PENDING"
                )

                blocking_ids = [
                    state["eventId"]
                ]

                continue

            if (
                state["status"]
                == "terminal_failed"
            ):

                result.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "TERMINAL_FAILURE"
                    ],
                    "dependencyDigests":
                        dependency_object(
                            node,
                            inputs,
                            artifacts,
                            key,
                        ),
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                blocking_reason = (
                    "UPSTREAM_TERMINAL"
                )

                blocking_ids = [
                    state["eventId"]
                ]

                continue

        # ----------------------------------------------------
        # READY CACHE MISS
        # ----------------------------------------------------

        result.append({
            "node": node,
            "action": "rerun",
            "reasonCodes": [
                "CACHE_MISS"
            ],
            "dependencyDigests":
                dependency_object(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds": [],
        })

        blocking_reason = (
            "UPSTREAM_PENDING"
        )

    return result


# ============================================================
# PIPELINE
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    # --------------------------------------------------------
    # JSON parsing
    # --------------------------------------------------------

    try:
        body = await request.json()

    except Exception:
        return conflict(
            "INVALID_REQUEST"
        )

    # --------------------------------------------------------
    # Request validation
    # --------------------------------------------------------

    if not validate_request(body):
        return conflict(
            "INVALID_REQUEST"
        )

    session_id = body[
        "session"
    ]

    revision = body[
        "revision"
    ]

    inputs = body[
        "inputs"
    ]

    events = body[
        "events"
    ]

    # Validate event shape before changing anything.
    for event in events:

        if not validate_event(event):
            return conflict(
                "INVALID_EVENT"
            )

    with LOCK:

        # ----------------------------------------------------
        # LOAD SESSION
        # ----------------------------------------------------

        if session_id in STORE[
            "sessions"
        ]:

            original = STORE[
                "sessions"
            ][session_id]

        else:

            original = empty_session()

        # ----------------------------------------------------
        # OLD REVISION
        # ----------------------------------------------------

        if (
            original["revision"]
            is not None
            and revision
            < original["revision"]
        ):

            return {
                "revision":
                    original["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": [
                    e["eventId"]
                    for e in events
                ],
                "nodes":
                    build_nodes(
                        original,
                        original["inputs"],
                    ),
            }

        # ----------------------------------------------------
        # SAME REVISION
        # ----------------------------------------------------

        if (
            original["revision"]
            is not None
            and revision
            == original["revision"]
        ):

            # Inputs are compared structurally.
            #
            # This includes extra metadata.
            if original["inputs"] != inputs:

                return conflict(
                    "REVISION_CONFLICT"
                )

        # ----------------------------------------------------
        # TRANSACTION COPY
        # ----------------------------------------------------

        working = copy.deepcopy(
            original
        )

        # ----------------------------------------------------
        # NEW REVISION
        #
        # Cache survives.
        # Active execution state resets.
        # Event IDs survive.
        # ----------------------------------------------------

        if (
            working["revision"]
            is None
            or revision
            > working["revision"]
        ):

            old_cache = copy.deepcopy(
                working["cache"]
            )

            old_event_ids = copy.deepcopy(
                working["eventIds"]
            )

            working = empty_session()

            working["revision"] = (
                revision
            )

            working["inputs"] = copy.deepcopy(
                inputs
            )

            working["cache"] = old_cache
            working["eventIds"] = (
                old_event_ids
            )

        else:

            working["revision"] = (
                revision
            )

            working["inputs"] = copy.deepcopy(
                inputs
            )

        accepted = []
        ignored = []

        # ----------------------------------------------------
        # PROCESS EVENTS IN ORDER
        # ----------------------------------------------------

        for event in events:

            event_id = event[
                "eventId"
            ]

            # ------------------------------------------------
            # WRONG REVISION
            # ------------------------------------------------

            if (
                event["revision"]
                != revision
            ):

                ignored.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # WRONG NODE
            # ------------------------------------------------

            if event["node"] not in DAG:

                ignored.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # EVENT ID
            # ------------------------------------------------

            canonical = canonical_event(
                event
            )

            previous = working[
                "eventIds"
            ].get(event_id)

            if previous is not None:

                if previous == canonical:

                    # Exact replay.
                    ignored.append(
                        event_id
                    )

                    continue

                return conflict(
                    "EVENT_ID_CONFLICT"
                )

            node = event[
                "node"
            ]

            # ------------------------------------------------
            # PARENT GATING
            # ------------------------------------------------

            parent = PARENT[node]

            if parent is not None:

                reusable_artifacts, reusable_keys = (
                    recover_cache(
                        working,
                        inputs,
                    )
                )

                if (
                    parent
                    not in reusable_artifacts
                ):

                    ignored.append(
                        event_id
                    )

                    continue

            # ------------------------------------------------
            # CURRENT KEY
            # ------------------------------------------------

            expected = expected_key(
                working,
                inputs,
                node,
            )

            if expected is None:

                ignored.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # WRONG / STALE KEY
            # ------------------------------------------------

            if event["key"] != expected:

                ignored.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # STATE TRANSITION
            # ------------------------------------------------

            outcome, code = (
                apply_transition(
                    working,
                    event,
                )
            )

            if outcome == "conflict":

                # IMPORTANT:
                # No STORE mutation has happened yet.
                # Entire batch rolls back.
                return conflict(code)

            if outcome == "accept":

                working[
                    "eventIds"
                ][event_id] = canonical

                accepted.append(
                    event_id
                )

            else:

                # Ignored events don't consume IDs.
                ignored.append(
                    event_id
                )

        # ----------------------------------------------------
        # ATOMIC COMMIT
        # ----------------------------------------------------

        STORE[
            "sessions"
        ][session_id] = working

        save_store()

        # ----------------------------------------------------
        # READ BACK
        # ----------------------------------------------------

        committed = STORE[
            "sessions"
        ][session_id]

        return {
            "revision":
                committed["revision"],
            "acceptedEventIds":
                accepted,
            "ignoredEventIds":
                ignored,
            "nodes":
                build_nodes(
                    committed,
                    committed["inputs"],
                ),
        }


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def health():

    return {
        "status": "ok",
        "service":
            "content-addressed-ml-pipeline",
    }
