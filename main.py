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


def cache_key(values):
    """
    Lowercase SHA-256 over UTF-8 compact JSON array.
    """
    raw = compact_json(values).encode("utf-8")

    return hashlib.sha256(raw).hexdigest().lower()


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
        content={
            "error": code
        },
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
            value = json.load(f)

        if not isinstance(value, dict):
            return empty_store()

        if not isinstance(
            value.get("sessions"),
            dict,
        ):
            return empty_store()

        return value

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

    temp_file = (
        STATE_FILE + ".tmp"
    )

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
# SESSION STATE
# ============================================================

def new_session():
    return {
        "revision": None,

        # Complete input object.
        # Extra metadata is preserved because same revision
        # must conflict if ANY input metadata changes.
        "inputs": None,

        # Global within this session.
        #
        # eventId -> canonical compact event JSON
        #
        # Accepted event IDs are stored here.
        # Ignored event IDs are NOT stored.
        "eventIds": {},

        # Current revision's live execution state.
        #
        # node -> None OR:
        #
        # {
        #   status,
        #   attempt,
        #   key,
        #   eventId,
        #   artifactDigest
        # }
        "state": {
            node: None
            for node in DAG
        },

        # Immutable successful evidence.
        #
        # cache[node][key] = {
        #     artifactDigest,
        #     eventId
        # }
        #
        # This survives revision changes.
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

    if not required.issubset(
        body.keys()
    ):
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

    # All 12 required inputs must be
    # non-empty strings.
    for name in INPUT_NAMES:
        if not nonempty_string(
            body["inputs"].get(name)
        ):
            return False

    # Extra input metadata is allowed.
    return True


def valid_event_shape(event):
    if not isinstance(event, dict):
        return False

    # Event must contain exactly eight fields.
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

    # Success requires artifact digest.
    if event["status"] == "succeeded":
        if not nonempty_string(
            event["artifactDigest"]
        ):
            return False

    # Every other status requires null artifact.
    else:
        if event["artifactDigest"] is not None:
            return False

    # Register and publish successful events
    # require their receipt.
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

        if (
            event["receiptId"]
            != expected_receipt
        ):
            return False

    # Every other event requires null receipt.
    else:
        if event["receiptId"] is not None:
            return False

    return True


def canonical_event(event):
    """
    Canonical compact JSON in exact event-field order.
    """

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
# CONTENT-ADDRESSED DEPENDENCY ARRAYS
# ============================================================

def dependency_array(
    node,
    inputs,
    reusable_artifacts,
):
    """
    EXACT assignment dependency arrays.

    A downstream dependency array is unavailable until
    its parent has a reusable successful artifact.
    """

    # --------------------------------------------------------
    # verify_data
    #
    # [generation, checksum]
    # --------------------------------------------------------

    if node == "verify_data":
        return [
            inputs["generation"],
            inputs["checksum"],
        ]

    # --------------------------------------------------------
    # prepare
    #
    # [canonicalData, prepareCode, prepareConfig]
    # --------------------------------------------------------

    if node == "prepare":

        if (
            "verify_data"
            not in reusable_artifacts
        ):
            return None

        return [
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ]

    # --------------------------------------------------------
    # train
    #
    # [prepareArtifact, trainCode, trainConfig, runtime]
    # --------------------------------------------------------

    if node == "train":

        if (
            "prepare"
            not in reusable_artifacts
        ):
            return None

        return [
            reusable_artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ]

    # --------------------------------------------------------
    # evaluate
    #
    # [trainArtifact, canonicalData,
    #  evaluateCode, evaluateConfig]
    # --------------------------------------------------------

    if node == "evaluate":

        if (
            "train"
            not in reusable_artifacts
        ):
            return None

        return [
            reusable_artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ]

    # --------------------------------------------------------
    # register
    #
    # [evaluateArtifact, schemaDigest]
    # --------------------------------------------------------

    if node == "register":

        if (
            "evaluate"
            not in reusable_artifacts
        ):
            return None

        return [
            reusable_artifacts["evaluate"],
            inputs["schemaDigest"],
        ]

    # --------------------------------------------------------
    # publish
    #
    # [registerArtifact, publishConfig]
    # --------------------------------------------------------

    if node == "publish":

        if (
            "register"
            not in reusable_artifacts
        ):
            return None

        return [
            reusable_artifacts["register"],
            inputs["publishConfig"],
        ]

    return None


# ============================================================
# CACHE RECOVERY
# ============================================================

def recover_chain(
    session,
    inputs,
):
    """
    Recover reusable cache prefix.

    Example with empty cache:

        verify_data = key
        prepare = null
        train = null
        evaluate = null
        register = null
        publish = null

    Example after verify_data is cached:

        verify_data = key
        prepare = key
        train = null
        evaluate = null
        register = null
        publish = null

    A downstream key is never computed from a
    non-reusable parent artifact.
    """

    artifacts = {}
    keys = {}

    for node in DAG:

        deps = dependency_array(
            node,
            inputs,
            artifacts,
        )

        # Parent is not reusable.
        if deps is None:
            break

        key = cache_key(deps)

        keys[node] = key

        entry = (
            session["cache"][node]
            .get(key)
        )

        # Current node is not cached.
        # Therefore descendants cannot obtain
        # reusable parent artifacts.
        if entry is None:
            break

        artifacts[node] = (
            entry["artifactDigest"]
        )

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
# IMMUTABLE SUCCESS EVIDENCE
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

    # First successful artifact permanently binds
    # this content-addressed key.
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
        "attempt":
            event["attempt"],
        "key": key,
        "eventId":
            event["eventId"],
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

    cached = (
        session["cache"][node]
        .get(key)
    )

    if cached is not None:

        if status == "succeeded":

            # Same key, different artifact:
            # immutable evidence conflict.
            if (
                event["artifactDigest"]
                != cached["artifactDigest"]
            ):
                return (
                    "conflict",
                    "EVIDENCE_CONFLICT",
                )

            # IMPORTANT:
            # Exact replay is handled before this function
            # using eventId.
            #
            # Therefore reaching here means this is a
            # NEW event ID.
            #
            # A successful cached node cannot accept
            # another new success.
            return (
                "conflict",
                "STATUS_CONFLICT",
            )

        # Any other event against successful cache
        # is a status conflict.
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # CURRENT LIVE STATE
    # --------------------------------------------------------

    current = session["state"][node]

    # --------------------------------------------------------
    # NO CURRENT STATE
    #
    # Only started(1) is accepted.
    # Completion without start is ignored.
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

        return (
            "ignore",
            None,
        )

    # --------------------------------------------------------
    # DIFFERENT KEY
    #
    # This is stale/non-current state.
    # --------------------------------------------------------

    if current["key"] != key:
        return (
            "ignore",
            None,
        )

    old_status = current["status"]
    old_attempt = current["attempt"]

    # --------------------------------------------------------
    # started(n)
    # --------------------------------------------------------

    if old_status == "started":

        # Lower attempt is stale.
        if attempt < old_attempt:
            return (
                "ignore",
                None,
            )

        # Exact attempt may succeed.
        if (
            attempt == old_attempt
            and status == "succeeded"
        ):
            commit_success(
                session,
                event,
            )

            return (
                "accept",
                None,
            )

        # Exact attempt may retryably fail.
        if (
            attempt == old_attempt
            and status
            == "retryable_failed"
        ):
            session["state"][node] = {
                "status":
                    "retryable_failed",
                "attempt":
                    attempt,
                "key":
                    key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        # Exact attempt may terminally fail.
        if (
            attempt == old_attempt
            and status
            == "terminal_failed"
        ):
            session["state"][node] = {
                "status":
                    "terminal_failed",
                "attempt":
                    attempt,
                "key":
                    key,
                "eventId":
                    event["eventId"],
            }

            return (
                "accept",
                None,
            )

        # Any other transition conflicts.
        return (
            "conflict",
            "STATUS_CONFLICT",
        )

    # --------------------------------------------------------
    # retryable_failed(n)
    #
    # Only started(n+1) is valid.
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
            session["state"][node] = {
                "status":
                    "started",
                "attempt":
                    attempt,
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
# BUILD RESPONSE NODES
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

    # Once a node blocks, all descendants are blocked.
    blocked_reason = None
    blocked_trigger_ids = []

    for index, node in enumerate(DAG):

        key = keys.get(node)

        # ----------------------------------------------------
        # KEY UNAVAILABLE
        # ----------------------------------------------------

        if key is None:

            if blocked_reason == "TERMINAL_FAILURE":
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
                    dependency_digests(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(
                        blocked_trigger_ids
                    ),
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
        # CURRENT NON-CACHED STATE
        # ----------------------------------------------------

        current = (
            session["state"][node]
        )

        if (
            current is not None
            and current["key"] == key
        ):

            if (
                current["status"]
                == "started"
            ):
                action = "block"
                reason = "RUNNING"

            elif (
                current["status"]
                == "retryable_failed"
            ):
                action = "rerun"
                reason = (
                    "RETRYABLE_FAILURE"
                )

            elif (
                current["status"]
                == "terminal_failed"
            ):
                action = "block"
                reason = (
                    "TERMINAL_FAILURE"
                )

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
            "reasonCodes": [
                reason
            ],
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
        # BLOCK ALL DESCENDANTS
        # ----------------------------------------------------

        if reason == "TERMINAL_FAILURE":
            blocked_reason = (
                "TERMINAL_FAILURE"
            )
        else:
            blocked_reason = "PENDING"

        blocked_trigger_ids = list(
            trigger_ids
        )

        for descendant in DAG[
            index + 1:
        ]:

            if (
                blocked_reason
                == "TERMINAL_FAILURE"
            ):
                descendant_reason = (
                    "UPSTREAM_TERMINAL"
                )
            else:
                descendant_reason = (
                    "UPSTREAM_PENDING"
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
                    list(
                        blocked_trigger_ids
                    ),
            })

        break

    return nodes


# ============================================================
# POST /pipeline
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    # --------------------------------------------------------
    # JSON parsing
    # --------------------------------------------------------

    try:
        body = await request.json()

    except Exception:
        return error_response(
            "INVALID_REQUEST"
        )

    # --------------------------------------------------------
    # Request validation
    # --------------------------------------------------------

    if not valid_request(body):
        return error_response(
            "INVALID_REQUEST"
        )

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # --------------------------------------------------------
    # Validate every event BEFORE mutating anything.
    # --------------------------------------------------------

    for event in events:

        if not valid_event_shape(event):
            return error_response(
                "INVALID_EVENT"
            )

    with LOCK:

        existing = get_session(
            session_id
        )

        # ====================================================
        # OLDER REQUEST REVISION
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
                    event["eventId"]
                    for event in events
                ],

                "nodes":
                    build_nodes(
                        existing,
                        existing["inputs"],
                    ),
            }

        # ====================================================
        # SAME REVISION
        # ====================================================

        if (
            existing["revision"] is not None
            and revision
            == existing["revision"]
        ):

            # Exact complete input object must match.
            # This includes extra metadata.
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
        # Inputs change.
        # Current state is cleared.
        # Successful cache remains.
        # Event-ID history remains session-global.
        # ====================================================

        if (
            working["revision"] is None
            or revision
            > working["revision"]
        ):

            preserved_cache = (
                copy.deepcopy(
                    working["cache"]
                )
            )

            preserved_event_ids = (
                copy.deepcopy(
                    working["eventIds"]
                )
            )

            working = new_session()

            working["revision"] = (
                revision
            )

            working["inputs"] = (
                copy.deepcopy(inputs)
            )

            working["cache"] = (
                preserved_cache
            )

            working["eventIds"] = (
                preserved_event_ids
            )

        else:

            working["revision"] = (
                revision
            )

            working["inputs"] = (
                copy.deepcopy(inputs)
            )

        accepted_ids = []
        ignored_ids = []

        # ====================================================
        # PROCESS EVENTS IN INPUT ORDER
        # ====================================================

        for event in events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # Wrong revision
            # ------------------------------------------------

            if (
                event["revision"]
                != revision
            ):
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # Wrong node
            # ------------------------------------------------

            if event["node"] not in DAG:
                ignored_ids.append(
                    event_id
                )
                continue

            # ------------------------------------------------
            # Canonical event
            # ------------------------------------------------

            canonical = canonical_event(
                event
            )

            # ------------------------------------------------
            # Event ID is global within session.
            # ------------------------------------------------

            previous = (
                working["eventIds"].get(
                    event_id
                )
            )

            if previous is not None:

                # Exact replay.
                if previous == canonical:

                    ignored_ids.append(
                        event_id
                    )

                    continue

                # Same ID, different content.
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
            # Recover current keys.
            # ------------------------------------------------

            artifacts, keys = recover_chain(
                working,
                inputs,
            )

            expected_key = keys.get(
                event["node"]
            )

            # No usable key.
            if expected_key is None:

                ignored_ids.append(
                    event_id
                )

                continue

            # Wrong key = stale event.
            if (
                event["key"]
                != expected_key
            ):

                ignored_ids.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # Apply event transition.
            # ------------------------------------------------

            outcome, code = apply_event(
                working,
                event,
            )

            # ------------------------------------------------
            # Any conflict rolls back entire batch.
            # ------------------------------------------------

            if outcome == "conflict":

                return error_response(
                    code
                )

            # ------------------------------------------------
            # Accepted event consumes its ID.
            # ------------------------------------------------

            if outcome == "accept":

                working["eventIds"][
                    event_id
                ] = canonical

                accepted_ids.append(
                    event_id
                )

            # ------------------------------------------------
            # Ignored event does NOT consume ID.
            # ------------------------------------------------

            else:

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

        # ====================================================
        # DURABLE READBACK
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
