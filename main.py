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
# JSON / HASH HELPERS
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_json_array(values: list[Any]) -> str:
    """
    Lowercase SHA-256 of compact UTF-8 JSON array.
    """
    payload = compact_json(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().lower()


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


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
# STATE
# ============================================================

def empty_session() -> dict:
    return {
        "revision": None,
        "inputs": None,

        # Global event ID ledger for this session.
        "eventIds": {},

        # Current revision execution states.
        "states": {
            node: None
            for node in DAG
        },

        # Successful immutable content-addressed evidence.
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
            value = json.load(f)

        if not isinstance(value, dict):
            return empty_store()

        if not isinstance(value.get("sessions"), dict):
            return empty_store()

        return value

    except Exception:
        return empty_store()


STORE = load_store()


def save_store() -> None:
    directory = os.path.dirname(STATE_FILE)

    if directory:
        os.makedirs(directory, exist_ok=True)

    temp_path = STATE_FILE + ".tmp"

    with open(
        temp_path,
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

    os.replace(temp_path, STATE_FILE)


# ============================================================
# REQUEST VALIDATION
# ============================================================

def validate_request(body: Any) -> bool:
    if not isinstance(body, dict):
        return False

    if not nonempty_string(body.get("session")):
        return False

    if not safe_positive_integer(body.get("revision")):
        return False

    inputs = body.get("inputs")

    if not isinstance(inputs, dict):
        return False

    events = body.get("events")

    if not isinstance(events, list):
        return False

    for name in REQUIRED_INPUTS:
        if not nonempty_string(inputs.get(name)):
            return False

    return True


def validate_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False

    # Exactly eight fields.
    if set(event.keys()) != set(EVENT_FIELDS):
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not safe_positive_integer(event["revision"]):
        return False

    if not nonempty_string(event["node"]):
        return False

    if not safe_positive_integer(event["attempt"]):
        return False

    if event["status"] not in STATUSES:
        return False

    if not nonempty_string(event["key"]):
        return False

    status = event["status"]
    node = event["node"]

    if status == "succeeded":
        if not nonempty_string(event["artifactDigest"]):
            return False
    else:
        if event["artifactDigest"] is not None:
            return False

    if node in {"register", "publish"}:
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
    """
    Canonical compact JSON with exactly the required
    eight fields in request-schema order.
    """
    value = {
        "eventId": event["eventId"],
        "revision": event["revision"],
        "node": event["node"],
        "attempt": event["attempt"],
        "status": event["status"],
        "key": event["key"],
        "artifactDigest": event["artifactDigest"],
        "receiptId": event["receiptId"],
    }

    return compact_json(value)


# ============================================================
# CONTENT-ADDRESSED DEPENDENCY ARRAYS
# ============================================================

def dependency_values(
    node: str,
    inputs: dict,
    artifacts: dict,
):
    """
    EXACT DAG arrays from the specification.

    A downstream array is unavailable until its parent
    artifact is reusable.
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
    values = dependency_values(
        node,
        inputs,
        artifacts,
    )

    if values is None:
        return None

    return sha256_json_array(values)


# ============================================================
# RECONSTRUCT REUSABLE PIPELINE PREFIX
# ============================================================

def reusable_pipeline(
    session: dict,
    inputs: dict,
):
    """
    Walk the DAG in strict order.

    For each node:
      1. calculate its dependency array only if its parent
         has a reusable artifact;
      2. calculate its key;
      3. check immutable cache;
      4. if cache misses, stop the reusable prefix.

    Therefore downstream keys are NULL until their parent
    becomes reusable.
    """

    artifacts = {}
    keys = {}

    blocked = False

    for node in DAG:

        if blocked:
            keys[node] = None
            continue

        key = node_key(
            node,
            inputs,
            artifacts,
        )

        if key is None:
            keys[node] = None
            blocked = True
            continue

        keys[node] = key

        entry = session["cache"][node].get(key)

        if entry is None:
            blocked = True
            continue

        artifacts[node] = entry["artifactDigest"]

    return artifacts, keys


def expected_node_key(
    session: dict,
    inputs: dict,
    target: str,
):
    """
    Calculate target key using only reusable parent artifacts.
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

        cached = session["cache"][node].get(key)

        if cached is None:
            return None

        artifacts[node] = cached["artifactDigest"]

    return None


# ============================================================
# CACHE / EVIDENCE
# ============================================================

def cache_entry(
    session: dict,
    node: str,
    key: str,
):
    return session["cache"][node].get(key)


def commit_success(
    session: dict,
    event: dict,
):
    node = event["node"]
    key = event["key"]

    existing = cache_entry(
        session,
        node,
        key,
    )

    if existing is None:
        session["cache"][node][key] = {
            "artifactDigest": event["artifactDigest"],
            "eventId": event["eventId"],
        }

    session["states"][node] = {
        "status": "succeeded",
        "attempt": event["attempt"],
        "key": key,
        "eventId": event["eventId"],
        "artifactDigest": event["artifactDigest"],
    }


# ============================================================
# EVENT STATE MACHINE
# ============================================================

def transition(
    session: dict,
    event: dict,
):
    node = event["node"]
    key = event["key"]
    status = event["status"]
    attempt = event["attempt"]

    # --------------------------------------------------------
    # IMMUTABLE SUCCESSFUL CACHE
    # --------------------------------------------------------

    cached = cache_entry(
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
                return "conflict", "EVIDENCE_CONFLICT"

            return "conflict", "STATUS_CONFLICT"

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # CURRENT STATE
    # --------------------------------------------------------

    current = session["states"][node]

    # --------------------------------------------------------
    # NO CURRENT STATE
    # --------------------------------------------------------

    if current is None:

        # Only started(1) can establish a fresh execution.
        if status == "started" and attempt == 1:

            session["states"][node] = {
                "status": "started",
                "attempt": 1,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        # Completion without initial start is ignored.
        return "ignore", None

    # --------------------------------------------------------
    # DIFFERENT KEY
    # --------------------------------------------------------

    if current["key"] != key:
        return "ignore", None

    old_status = current["status"]
    old_attempt = current["attempt"]

    # --------------------------------------------------------
    # STARTED(n)
    # --------------------------------------------------------

    if old_status == "started":

        if attempt < old_attempt:
            return "ignore", None

        if (
            status == "succeeded"
            and attempt == old_attempt
        ):
            commit_success(
                session,
                event,
            )

            return "accept", None

        if (
            status == "retryable_failed"
            and attempt == old_attempt
        ):
            session["states"][node] = {
                "status": "retryable_failed",
                "attempt": old_attempt,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        if (
            status == "terminal_failed"
            and attempt == old_attempt
        ):
            session["states"][node] = {
                "status": "terminal_failed",
                "attempt": old_attempt,
                "key": key,
                "eventId": event["eventId"],
            }

            return "accept", None

        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # RETRYABLE_FAILURE(n)
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
    # TERMINAL_FAILURE
    # --------------------------------------------------------

    if old_status == "terminal_failed":
        return "conflict", "STATUS_CONFLICT"

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if old_status == "succeeded":
        return "conflict", "STATUS_CONFLICT"

    return "conflict", "STATUS_CONFLICT"


# ============================================================
# RESPONSE DEPENDENCIES
# ============================================================

def dependency_digest_object(
    node: str,
    inputs: dict,
    artifacts: dict,
    key,
):
    """
    Preserve the exact named dependency order from the spec.
    """

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
            "prepareArtifact": artifacts.get("prepare"),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
            "cacheKey": key,
        }

    if node == "evaluate":
        return {
            "trainArtifact": artifacts.get("train"),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
            "cacheKey": key,
        }

    if node == "register":
        return {
            "evaluateArtifact": artifacts.get("evaluate"),
            "schemaDigest": inputs["schemaDigest"],
            "cacheKey": key,
        }

    if node == "publish":
        return {
            "registerArtifact": artifacts.get("register"),
            "publishConfig": inputs["publishConfig"],
            "cacheKey": key,
        }

    return {
        "cacheKey": key,
    }


# ============================================================
# RESPONSE BUILDING
# ============================================================

def build_nodes(
    session: dict,
    inputs: dict,
):
    artifacts, keys = reusable_pipeline(
        session,
        inputs,
    )

    output = []

    upstream_terminal = False
    upstream_pending = False
    upstream_trigger_ids = []

    for node in DAG:

        key = keys.get(node)

        # ----------------------------------------------------
        # DOWNSTREAM KEY NOT AVAILABLE
        # ----------------------------------------------------

        if key is None:

            if upstream_terminal:
                reason = "UPSTREAM_TERMINAL"
            else:
                reason = "UPSTREAM_PENDING"

            output.append({
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests":
                    dependency_digest_object(
                        node,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(upstream_trigger_ids),
            })

            continue

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        cached = cache_entry(
            session,
            node,
            key,
        )

        if cached is not None:

            output.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    dependency_digest_object(
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

        state = session["states"][node]

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

        output.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests":
                dependency_digest_object(
                    node,
                    inputs,
                    artifacts,
                    key,
                ),
            "triggeringEventIds":
                trigger_ids,
        })

        # ----------------------------------------------------
        # SET DESCENDANT BLOCKING MODE
        # ----------------------------------------------------

        if reason == "TERMINAL_FAILURE":
            upstream_terminal = True
            upstream_pending = False
        else:
            upstream_pending = True
            upstream_terminal = False

        upstream_trigger_ids = list(
            trigger_ids
        )

        # Remaining descendants are blocked.
        for descendant in DAG[
            DAG.index(node) + 1:
        ]:

            if upstream_terminal:
                descendant_reason = "UPSTREAM_TERMINAL"
            else:
                descendant_reason = "UPSTREAM_PENDING"

            output.append({
                "node": descendant,
                "action": "block",
                "reasonCodes": [
                    descendant_reason
                ],
                "dependencyDigests":
                    dependency_digest_object(
                        descendant,
                        inputs,
                        artifacts,
                        None,
                    ),
                "triggeringEventIds":
                    list(upstream_trigger_ids),
            })

        break

    return output


# ============================================================
# PIPELINE ENDPOINT
# ============================================================

@app.post("/pipeline")
async def pipeline(request: Request):

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return conflict("INVALID_REQUEST")

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not validate_request(body):
        return conflict("INVALID_REQUEST")

    session_id = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    # --------------------------------------------------------
    # Validate every event's shape before mutation.
    # --------------------------------------------------------

    for event in events:
        if not validate_event(event):
            return conflict("INVALID_EVENT")

    with LOCK:

        # ----------------------------------------------------
        # Get session.
        # ----------------------------------------------------

        if session_id in STORE["sessions"]:
            original = STORE["sessions"][session_id]
        else:
            original = empty_session()

        # ----------------------------------------------------
        # Older request revision.
        #
        # Well-formed events are ignored.
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
                "revision": original["revision"],
                "acceptedEventIds": [],
                "ignoredEventIds": ignored,
                "nodes": build_nodes(
                    original,
                    original["inputs"],
                ),
            }

        # ----------------------------------------------------
        # Same revision requires byte-equivalent JSON values.
        #
        # Extra metadata therefore participates in revision
        # identity even though it does not participate in keys.
        # ----------------------------------------------------

        if (
            original["revision"] is not None
            and revision == original["revision"]
        ):

            if compact_json(original["inputs"]) != compact_json(inputs):
                return conflict("REVISION_CONFLICT")

        # ----------------------------------------------------
        # Work on a deep copy.
        #
        # This guarantees a conflicting event batch does not
        # partially mutate persistent state.
        # ----------------------------------------------------

        working = copy.deepcopy(original)

        # ----------------------------------------------------
        # New revision.
        #
        # Preserve successful cache and global event ledger.
        # Clear current execution state.
        # ----------------------------------------------------

        if (
            working["revision"] is None
            or revision > working["revision"]
        ):

            old_cache = copy.deepcopy(
                working["cache"]
            )

            old_event_ids = copy.deepcopy(
                working["eventIds"]
            )

            working = empty_session()

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

        # ----------------------------------------------------
        # PROCESS EVENTS IN INPUT ORDER
        # ----------------------------------------------------

        for event in events:

            event_id = event["eventId"]

            # ------------------------------------------------
            # Wrong revision = ignore.
            # ------------------------------------------------

            if event["revision"] != revision:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Wrong node = ignore.
            # ------------------------------------------------

            if event["node"] not in DAG:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Global event ID handling.
            # ------------------------------------------------

            canonical = canonical_event(event)

            previous = working["eventIds"].get(
                event_id
            )

            if previous is not None:

                if previous == canonical:
                    ignored.append(event_id)
                    continue

                return conflict(
                    "EVENT_ID_CONFLICT"
                )

            # ------------------------------------------------
            # Parent must currently be reusable.
            # ------------------------------------------------

            parent = PARENT[event["node"]]

            if parent is not None:

                reusable_artifacts, reusable_keys = (
                    reusable_pipeline(
                        working,
                        inputs,
                    )
                )

                if (
                    parent not in reusable_artifacts
                    or reusable_keys.get(parent) is None
                ):
                    ignored.append(event_id)
                    continue

            # ------------------------------------------------
            # Calculate current content-addressed key.
            # ------------------------------------------------

            expected = expected_node_key(
                working,
                inputs,
                event["node"],
            )

            if expected is None:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Stale / wrong key = ignore.
            # ------------------------------------------------

            if event["key"] != expected:
                ignored.append(event_id)
                continue

            # ------------------------------------------------
            # Apply transition.
            # ------------------------------------------------

            result, code = transition(
                working,
                event,
            )

            if result == "conflict":
                return conflict(code)

            if result == "accept":

                working["eventIds"][
                    event_id
                ] = canonical

                accepted.append(event_id)

            else:
                # Ignored events do not consume IDs.
                ignored.append(event_id)

        # ----------------------------------------------------
        # ATOMIC COMMIT
        # ----------------------------------------------------

        STORE["sessions"][session_id] = working

        save_store()

        committed = STORE["sessions"][session_id]

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "revision": committed["revision"],
            "acceptedEventIds": accepted,
            "ignoredEventIds": ignored,
            "nodes": build_nodes(
                committed,
                committed["inputs"],
            ),
        }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "content-addressed-ml-pipeline",
    }
