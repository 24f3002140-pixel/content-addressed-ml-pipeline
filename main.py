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
# BASIC HELPERS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_compact_array(values):
    return hashlib.sha256(
        compact_json(values).encode("utf-8")
    ).hexdigest().lower()


def is_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def is_safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= MAX_SAFE_INTEGER
    )


def error_response(code):
    return JSONResponse(
        status_code=409,
        content={"error": code},
    )


# ============================================================
# PERSISTENCE
# ============================================================

def make_empty_session():
    return {
        "revision": None,
        "inputs": None,

        # Event IDs already consumed by accepted events.
        "eventIds": {},

        # Current revision execution state.
        "state": {
            node: None
            for node in DAG
        },

        # Immutable content-addressed successful evidence.
        #
        # {
        #   node: {
        #       cache_key: {
        #           "artifactDigest": "...",
        #           "eventId": "..."
        #       }
        #   }
        # }
        "cache": {
            node: {}
            for node in DAG
        },
    }


def make_empty_store():
    return {
        "sessions": {}
    }


def load_store():
    if not os.path.exists(STATE_FILE):
        return make_empty_store()

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return make_empty_store()

        if not isinstance(
            data.get("sessions"),
            dict,
        ):
            return make_empty_store()

        return data

    except Exception:
        return make_empty_store()


STORE = load_store()


def save_store():
    directory = os.path.dirname(STATE_FILE)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temp_file = STATE_FILE + ".tmp"

    with open(
        temp_file,
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
        temp_file,
        STATE_FILE,
    )


# ============================================================
# REQUEST VALIDATION
# ============================================================

def validate_request(body):
    if not isinstance(body, dict):
        return False

    if not is_nonempty_string(
        body.get("session")
    ):
        return False

    if not is_safe_positive_integer(
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
        if not is_nonempty_string(
            inputs.get(name)
        ):
            return False

    return True


def validate_event_shape(event):
    if not isinstance(event, dict):
        return False

    # Exactly the eight required fields.
    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not is_nonempty_string(
        event["eventId"]
    ):
        return False

    if not is_safe_positive_integer(
        event["revision"]
    ):
        return False

    if not is_nonempty_string(
        event["node"]
    ):
        return False

    if not is_safe_positive_integer(
        event["attempt"]
    ):
        return False

    if event["status"] not in VALID_STATUSES:
        return False

    if not is_nonempty_string(
        event["key"]
    ):
        return False

    if event["status"] == "succeeded":
        if not is_nonempty_string(
            event["artifactDigest"]
        ):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register and publish success require
    # receipt:<node>:<key>.
    if event["node"] in {
        "register",
        "publish",
    }:
        if event["status"] == "succeeded":
            expected_receipt = (
                "receipt:"
                + event["node"]
                + ":"
                + event["key"]
            )

            if event["receiptId"] != expected_receipt:
                return False
        else:
            if event["receiptId"] is not None:
                return False
    else:
        if event["receiptId"] is not None:
            return False

    return True


def canonical_event_json(event):
    # Canonical compact JSON in the exact field order.
    canonical = {
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest": event["artifactDigest"],
        "receiptId": event["receiptId"],
    }

    return compact_json(canonical)


# ============================================================
# CONTENT-ADDRESSED DAG KEYS
# ============================================================

def dependency_array(
    node,
    inputs,
    artifacts,
):
    """
    EXACT required dependency arrays.

    A node's dependency array is unavailable until the
    parent artifact is reusable.
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


def calculate_key(
    node,
    inputs,
    artifacts,
):
    values = dependency_array(
        node,
        inputs,
        artifacts,
    )

    if values is None:
        return None

    return sha256_compact_array(values)


# ============================================================
# RECOVER REUSABLE CACHE
# ============================================================

def recover_reusable_prefix(
    session,
    inputs,
):
    """
    Walk the fixed DAG in order.

    A child can only obtain a key after its parent has a
    reusable successful artifact.

    Once a node misses cache, downstream keys become None.
    """

    artifacts = {}
    keys = {}

    cache_open = True

    for node in DAG:

        if not cache_open:
            keys[node] = None
            continue

        key = calculate_key(
            node,
            inputs,
            artifacts,
        )

        if key is None:
            keys[node] = None
            cache_open = False
            continue

        keys[node] = key

        entry = (
            session["cache"][node]
            .get(key)
        )

        if entry is None:
            cache_open = False
            continue

        artifacts[node] = (
            entry["artifactDigest"]
        )

    return artifacts, keys


# ============================================================
# CURRENT EXPECTED KEY
# ============================================================

def expected_key_for_node(
    session,
    inputs,
    node,
):
    artifacts = {}

    for current_node in DAG:

        key = calculate_key(
            current_node,
            inputs,
            artifacts,
        )

        if current_node == node:
            return key

        if key is None:
            return None

        cached = (
            session["cache"][current_node]
            .get(key)
        )

        if cached is None:
            return None

        artifacts[current_node] = (
            cached["artifactDigest"]
        )

    return None


def parent_is_reusable(
    session,
    inputs,
    node,
):
    parent = PARENT[node]

    if parent is None:
        return True

    artifacts, keys = recover_reusable_prefix(
        session,
        inputs,
    )

    return (
        parent in artifacts
        and keys.get(parent) is not None
    )


# ============================================================
# RESPONSE DEPENDENCY OBJECT
# ============================================================

def response_dependency_digests(
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
# IMMUTABLE SUCCESS EVIDENCE
# ============================================================

def get_cache_entry(
    session,
    node,
    key,
):
    return (
        session["cache"][node]
        .get(key)
    )


def save_success_evidence(
    session,
    event,
):
    node = event["node"]
    key = event["key"]

    existing = get_cache_entry(
        session,
        node,
        key,
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
# EVENT STATE MACHINE
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
    # IMMUTABLE CACHE EVIDENCE
    # --------------------------------------------------------

    cached = get_cache_entry(
        session,
        node,
        key,
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

            return (
                "conflict",
                "STATUS_CONFLICT",
            )

        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # CURRENT LIVE STATE
    # --------------------------------------------------------

    current = session["state"][node]

    # --------------------------------------------------------
    # NO PREVIOUS STATE
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

            return (
                "accept",
                None,
            )

        # Completion without start or attempt > 1.
        return (
            "ignore",
            None,
        )

    # --------------------------------------------------------
    # DIFFERENT CURRENT KEY
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
            status == "succeeded"
            and attempt == old_attempt
        ):
            save_success_evidence(
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
            status == "terminal_failed"
            and attempt == old_attempt
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
    # TERMINAL FAILURE
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
# NODE RESPONSE
# ============================================================

def build_node_response(
    session,
    inputs,
):
    artifacts, keys = recover_reusable_prefix(
        session,
        inputs,
    )

    nodes = []

    blocked_reason = None
    blocked_trigger_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # BLOCKED BY UPSTREAM
        # ----------------------------------------------------

        if key is None:

            if (
                blocked_reason
                == "TERMINAL_FAILURE"
            ):
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    reason
                ],
                "dependencyDigests":
                    response_dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(blocked_trigger_ids),
            })

            continue

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        cached = get_cache_entry(
            session,
            node,
            key,
        )

        if cached is not None:

            nodes.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    response_dependency_digests(
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
        # CURRENT EXECUTION STATE
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

            trigger_ids = [
                state["eventId"]
            ]

        else:

            action = "rerun"
            reason = "CACHE_MISS"
            trigger_ids = []

        nodes.append({
            "node": node,
            "action": action,
            "reasonCodes": [
                reason
            ],
            "dependencyDigests":
                response_dependency_digests(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds":
                trigger_ids,
        })

        # ----------------------------------------------------
        # ALL DESCENDANTS ARE BLOCKED
        # ----------------------------------------------------

        blocked_reason = (
            "TERMINAL_FAILURE"
            if reason == "TERMINAL_FAILURE"
            else "PENDING"
        )

        blocked_trigger_ids = list(
            trigger_ids
        )

        for descendant in DAG[
            index + 1:
        ]:

            descendant_reason = (
                "UPSTREAM_TERMINAL"
                if blocked_reason
                == "TERMINAL_FAILURE"
                else "UPSTREAM_PENDING"
            )

            nodes.append({
                "node": descendant,
                "action": "block",
                "reasonCodes": [
                    descendant_reason
                ],
                "dependencyDigests":
                    response_dependency_digests(
                        descendant,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(blocked_trigger_ids),
            })

        break

    return nodes


# ============================================================
# POST /pipeline
# ============================================================

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

    # --------------------------------------------------------
    # Validate event shape BEFORE any mutation.
    # --------------------------------------------------------

    for event in events:
        if not validate_event_shape(event):
            return error_response(
                "INVALID_EVENT"
            )

    with LOCK:

        if session_id not in STORE["sessions"]:
            existing = make_empty_session()
        else:
            existing = STORE["sessions"][
                session_id
            ]

        # ====================================================
        # OLDER REQUEST REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision < existing["revision"]
        ):
            ignored = [
                event["eventId"]
                for event in events
            ]

            return {
                "revision":
                    existing["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": ignored,
                "nodes":
                    build_node_response(
                        existing,
                        existing["inputs"],
                    ),
            }

        # ====================================================
        # SAME REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision == existing["revision"]
        ):
            if existing["inputs"] != inputs:
                return error_response(
                    "REVISION_CONFLICT"
                )

        # ====================================================
        # WORKING COPY FOR ATOMIC BATCH
        # ====================================================

        working = copy.deepcopy(
            existing
        )

        # ====================================================
        # NEW REVISION
        #
        # Inputs change.
        # Live attempts/terminal state are cleared.
        #
        # Successful cache entries survive.
        # Event ID history also survives because IDs are
        # global within a session.
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

            working = make_empty_session()

            working["revision"] = revision
            working["inputs"] = copy.deepcopy(
                inputs
            )

            working["cache"] = preserved_cache
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

        for event in events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # EVENT FROM OLDER / DIFFERENT REVISION
            # ------------------------------------------------

            if event["revision"] != revision:
                ignored_ids.append(event_id)
                continue

            # ------------------------------------------------
            # UNKNOWN NODE
            # ------------------------------------------------

            if event["node"] not in DAG:
                ignored_ids.append(event_id)
                continue

            # ------------------------------------------------
            # EVENT ID REPLAY / CONFLICT
            # ------------------------------------------------

            canonical = canonical_event_json(
                event
            )

            previous_canonical = (
                working["eventIds"]
                .get(event_id)
            )

            if previous_canonical is not None:

                if (
                    previous_canonical
                    == canonical
                ):
                    ignored_ids.append(
                        event_id
                    )
                    continue

                return error_response(
                    "EVENT_ID_CONFLICT"
                )

            # ------------------------------------------------
            # PARENT MUST BE REUSABLE
            # ------------------------------------------------

            if not parent_is_reusable(
                working,
                inputs,
                event["node"],
            ):
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # CURRENT CONTENT-ADDRESSED KEY
            # ------------------------------------------------

            expected_key = expected_key_for_node(
                working,
                inputs,
                event["node"],
            )

            if expected_key is None:
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # STALE / WRONG KEY
            # ------------------------------------------------

            if event["key"] != expected_key:
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # STATE TRANSITION
            # ------------------------------------------------

            result, code = apply_event(
                working,
                event,
            )

            if result == "conflict":
                # Nothing from this batch is committed.
                return error_response(code)

            if result == "accept":

                working["eventIds"][
                    event_id
                ] = canonical

                accepted_ids.append(
                    event_id
                )

            else:

                # Ignored events do NOT consume IDs.
                ignored_ids.append(
                    event_id
                )

        # ====================================================
        # ATOMIC COMMIT
        # ====================================================

        STORE["sessions"][
            session_id
        ] = working

        save_store()

        committed = STORE["sessions"][
            session_id
        ]

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "revision":
                committed["revision"],
            "acceptedEventIds":
                accepted_ids,
            "ignoredEventIds":
                ignored_ids,
            "nodes":
                build_node_response(
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
