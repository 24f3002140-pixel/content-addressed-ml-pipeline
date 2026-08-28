```python
import copy
import hashlib
import json
import os
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

def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_array(values):
    return hashlib.sha256(
        compact(values).encode("utf-8")
    ).hexdigest().lower()


def nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def conflict(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# PERSISTENCE
# ============================================================

def empty_store():
    return {
        "sessions": {}
    }


def load_store():
    try:
        if not os.path.exists(STATE_FILE):
            return empty_store()

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if (
            not isinstance(data, dict)
            or not isinstance(
                data.get("sessions"),
                dict,
            )
        ):
            return empty_store()

        return data

    except Exception:
        return empty_store()


STORE = load_store()


def save_store():
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
        "inputs": None,

        # eventId -> canonical event JSON
        # Global inside this session.
        "eventIds": {},

        # Current revision execution state.
        "state": {
            node: None
            for node in DAG
        },

        # Immutable successful evidence.
        #
        # node -> cacheKey -> {
        #     artifactDigest,
        #     eventId
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


def get_or_create_session(session_id):
    if session_id not in STORE["sessions"]:
        return new_session()

    return STORE["sessions"][session_id]


# ============================================================
# VALIDATION
# ============================================================

def valid_request(body):
    if not isinstance(body, dict):
        return False

    if not isinstance(body.get("session"), str):
        return False

    if len(body["session"]) == 0:
        return False

    if not safe_positive_integer(
        body.get("revision")
    ):
        return False

    if not isinstance(
        body.get("inputs"),
        dict,
    ):
        return False

    if not isinstance(
        body.get("events"),
        list,
    ):
        return False

    for name in REQUIRED_INPUTS:
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

    if event["status"] == "succeeded":

        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False

    else:

        if event["artifactDigest"] is not None:
            return False

    if event["node"] in {
        "register",
        "publish",
    }:

        if event["status"] == "succeeded":

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
        "artifactDigest":
            event["artifactDigest"],
        "receiptId":
            event["receiptId"],
    }

    return compact(ordered)


# ============================================================
# EXACT DAG DEPENDENCIES
# ============================================================

def dependency_values(
    node,
    inputs,
    artifacts,
):
    """
    Exact arrays required by the assignment.

    IMPORTANT:
    The parent artifact is used only where the specification
    explicitly places it in the array.
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


# ============================================================
# RECOVER CACHE PREFIX
# ============================================================

def recover_cache_prefix(
    session,
    inputs,
):
    """
    Returns:

        artifacts:
            reusable successful artifacts

        keys:
            key for every node whose parent is reusable

    Once a node has no cache hit, descendants receive
    null/unavailable keys.
    """

    artifacts = {}
    keys = {}

    stopped = False

    for node in DAG:

        if stopped:
            keys[node] = None
            continue

        values = dependency_values(
            node,
            inputs,
            artifacts,
        )

        if values is None:
            keys[node] = None
            stopped = True
            continue

        key = sha256_array(values)
        keys[node] = key

        entry = (
            session["cache"][node]
            .get(key)
        )

        if entry is None:
            stopped = True
            continue

        artifacts[node] = (
            entry["artifactDigest"]
        )

    return artifacts, keys


# ============================================================
# CURRENT NODE KEY
# ============================================================

def current_key(
    session,
    inputs,
    node,
):
    """
    Calculate a node key only when its parent is reusable.

    This is intentionally independent from the live state.
    """

    artifacts, keys = recover_cache_prefix(
        session,
        inputs,
    )

    return (
        keys.get(node),
        artifacts,
    )


# ============================================================
# RESPONSE DEPENDENCIES
# ============================================================

def response_dependencies(
    node,
    inputs,
    artifacts,
    key,
):
    if node == "verify_data":

        return {
            "generation":
                inputs["generation"],
            "checksum":
                inputs["checksum"],
            "cacheKey":
                key,
        }

    if node == "prepare":

        return {
            "canonicalData":
                inputs["canonicalData"],
            "prepareCode":
                inputs["prepareCode"],
            "prepareConfig":
                inputs["prepareConfig"],
            "cacheKey":
                key,
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
            "cacheKey":
                key,
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
            "cacheKey":
                key,
        }

    if node == "register":

        return {
            "evaluateArtifact":
                artifacts.get("evaluate"),
            "schemaDigest":
                inputs["schemaDigest"],
            "cacheKey":
                key,
        }

    if node == "publish":

        return {
            "registerArtifact":
                artifacts.get("register"),
            "publishConfig":
                inputs["publishConfig"],
            "cacheKey":
                key,
        }

    return {
        "cacheKey": key
    }


# ============================================================
# PARENT REUSABILITY
# ============================================================

def is_parent_reusable(
    session,
    inputs,
    node,
):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = recover_cache_prefix(
        session,
        inputs,
    )

    return (
        parent in artifacts
        and keys.get(parent) is not None
    )


# ============================================================
# SUCCESSFUL IMMUTABLE EVIDENCE
# ============================================================

def store_success(
    session,
    event,
):
    node = event["node"]
    key = event["key"]

    existing = (
        session["cache"][node]
        .get(key)
    )

    if existing is None:

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


# ============================================================
# EVENT TRANSITIONS
# ============================================================

def transition(
    session,
    event,
):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # --------------------------------------------------------
    # IMMUTABLE CACHE
    # --------------------------------------------------------

    cached = (
        session["cache"][node]
        .get(key)
    )

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

            # Same evidence under a NEW event ID is not
            # a new state transition.
            return (
                "conflict",
                "STATUS_CONFLICT",
            )

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # LIVE STATE
    # --------------------------------------------------------

    current = session["state"][node]

    # --------------------------------------------------------
    # NO STATE
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
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        # Completion or attempt > 1 without start
        # is ignored.
        return (
            "ignore",
            None,
        )

    # --------------------------------------------------------
    # DIFFERENT KEY = STALE
    # --------------------------------------------------------

    if current["key"] != key:
        return (
            "ignore",
            None,
        )

    old_status = current["status"]
    old_attempt = current["attempt"]

    # --------------------------------------------------------
    # STARTED
    # --------------------------------------------------------

    if old_status == "started":

        if attempt < old_attempt:
            return (
                "ignore",
                None,
            )

        if (
            attempt == old_attempt
            and status == "succeeded"
        ):

            store_success(
                session,
                event,
            )

            return (
                "accept",
                None,
            )

        if (
            attempt == old_attempt
            and status
            == "retryable_failed"
        ):

            session["state"][node] = {
                "status":
                    "retryable_failed",
                "attempt":
                    old_attempt,
                "key":
                    key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        if (
            attempt == old_attempt
            and status
            == "terminal_failed"
        ):

            session["state"][node] = {
                "status":
                    "terminal_failed",
                "attempt":
                    old_attempt,
                "key":
                    key,
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
    # RETRYABLE FAILURE
    # --------------------------------------------------------

    if old_status == "retryable_failed":

        if attempt < old_attempt:
            return (
                "ignore",
                None,
            )

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
# RESPONSE BUILDER
# ============================================================

def build_nodes(
    session,
    inputs,
):
    artifacts, keys = recover_cache_prefix(
        session,
        inputs,
    )

    result = []

    upstream_block = None
    upstream_events = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # PARENT NOT REUSABLE
        # ----------------------------------------------------

        if key is None:

            if (
                upstream_block
                == "TERMINAL_FAILURE"
            ):
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            result.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    reason
                ],
                "dependencyDigests":
                    response_dependencies(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(upstream_events),
            })

            continue

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        cached = (
            session["cache"][node]
            .get(key)
        )

        if cached is not None:

            result.append({
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
                    cached["eventId"]
                ],
            })

            continue

        # ----------------------------------------------------
        # CACHE MISS / LIVE STATE
        # ----------------------------------------------------

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

            triggers = [
                state["eventId"]
            ]

        else:

            action = "rerun"
            reason = "CACHE_MISS"
            triggers = []

        result.append({
            "node": node,
            "action": action,
            "reasonCodes": [
                reason
            ],
            "dependencyDigests":
                response_dependencies(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds":
                triggers,
        })

        # ----------------------------------------------------
        # BLOCK ALL DESCENDANTS
        # ----------------------------------------------------

        upstream_block = (
            "TERMINAL_FAILURE"
            if reason == "TERMINAL_FAILURE"
            else "PENDING"
        )

        upstream_events = list(triggers)

        for descendant in DAG[
            index + 1:
        ]:

            descendant_reason = (
                "UPSTREAM_TERMINAL"
                if upstream_block
                == "TERMINAL_FAILURE"
                else "UPSTREAM_PENDING"
            )

            result.append({
                "node": descendant,
                "action": "block",
                "reasonCodes": [
                    descendant_reason
                ],
                "dependencyDigests":
                    response_dependencies(
                        descendant,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(upstream_events),
            })

        break

    return result


# ============================================================
# POST /pipeline
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    try:
        body = await request.json()
    except Exception:
        return conflict("INVALID_REQUEST")

    if not valid_request(body):
        return conflict("INVALID_REQUEST")

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # Validate complete batch before mutation.
    for event in events:

        if not valid_event(event):
            return conflict("INVALID_EVENT")

    with LOCK:

        existing = get_or_create_session(
            session_id
        )

        # ====================================================
        # OLDER REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision
            < existing["revision"]
        ):

            return {
                "revision":
                    existing["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": [
                    e["eventId"]
                    for e in events
                ],
                "nodes":
                    build_nodes(
                        existing,
                        existing["inputs"],
                    ),
            }

        # ====================================================
        # SAME REVISION INPUT CHECK
        # ====================================================

        if (
            existing["revision"] is not None
            and revision
            == existing["revision"]
        ):

            if existing["inputs"] != inputs:

                return conflict(
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
        # Preserve:
        #   cache
        #   event IDs
        #
        # Reset:
        #   live state
        # ====================================================

        if (
            working["revision"] is None
            or revision
            > working["revision"]
        ):

            old_cache = copy.deepcopy(
                working["cache"]
            )

            old_event_ids = copy.deepcopy(
                working["eventIds"]
            )

            working = new_session()

            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )

            working["cache"] = old_cache
            working["eventIds"] = old_event_ids

        else:

            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )

        accepted = []
        ignored = []

        # ====================================================
        # INPUT EVENT ORDER
        # ====================================================

        for event in events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # OLD REVISION
            # ------------------------------------------------

            if event["revision"] != revision:

                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # UNKNOWN NODE
            # ------------------------------------------------

            if event["node"] not in DAG:

                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # EVENT ID
            # ------------------------------------------------

            canonical = canonical_event(
                event
            )

            previous = (
                working["eventIds"]
                .get(event_id)
            )

            if previous is not None:

                if previous == canonical:

                    # Exact replay.
                    ignored.append(event_id)
                    continue

                return conflict(
                    "EVENT_ID_CONFLICT"
                )

            # ------------------------------------------------
            # PARENT GATING
            # ------------------------------------------------

            if not is_parent_reusable(
                working,
                inputs,
                event["node"],
            ):

                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # CURRENT EXPECTED KEY
            # ------------------------------------------------

            expected_key, _ = current_key(
                working,
                inputs,
                event["node"],
            )

            if expected_key is None:

                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # STALE KEY
            # ------------------------------------------------

            if event["key"] != expected_key:

                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # APPLY TRANSITION
            # ------------------------------------------------

            outcome, code = transition(
                working,
                event,
            )

            if outcome == "conflict":

                # No STORE mutation has happened.
                # Entire batch rolls back.
                return conflict(code)

            if outcome == "accept":

                working["eventIds"][
                    event_id
                ] = canonical

                accepted.append(event_id)

            else:

                # Ignored events do not consume IDs.
                ignored.append(event_id)

        # ====================================================
        # ATOMIC COMMIT
        # ====================================================

        STORE["sessions"][
            session_id
        ] = working

        save_store()

        # ====================================================
        # READBACK
        # ====================================================

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
```
