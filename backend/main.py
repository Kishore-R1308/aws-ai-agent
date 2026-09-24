import uuid
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from backend.agent import run_agent
from backend.aws_auth import AWS_SESSIONS, connect_aws
from backend.database import Base, engine, get_db
from backend.models import ChatMessage
from backend.schemas import (
    AWSConnectRequest,
    AWSConnectResponse,
    ChatRequest,
    ChatResponse,
)

from backend.action_schemas import (
    AWSActionRequest,
    AWSActionApprovalRequest,
    RCARecommendedActionBatch,
    RCABatchApprovalRequest,
)

from backend.action_store import (
    create_pending_action,
    get_pending_action,
    approve_pending_action,
    create_action_batch,
    get_action_batch,
    batch_belongs_to_session,
    approve_action_batch,
    update_batch_status,
    update_batch_action_status,
)

from backend.action_validation import validate_action
from backend.aws_action_executor import execute_aws_action
from backend.aws_tools import (
    get_ec2_instances,
    get_s3_buckets,
    get_rds_instances,
    get_lambda_functions,
    get_vpcs,
    get_subnets,
    get_security_groups,
    get_iam_resources,
)


# =====================================================
# DATABASE INITIALIZATION / MIGRATION
# =====================================================

Base.metadata.create_all(bind=engine)


def migrate_chat_messages():
    """
    Safely migrate existing chat_messages databases.

    Adds columns required by the current ChatMessage model when they are
    missing from an older SQLite database. Existing rows are preserved.
    """

    inspector = inspect(engine)

    if "chat_messages" not in inspector.get_table_names():
        return

    columns = {
        column["name"]
        for column in inspector.get_columns("chat_messages")
    }

    # Each tuple contains the column name and the SQLite ALTER TABLE
    # statement needed to add it.
    migrations = []

    if "conversation_id" not in columns:
        migrations.append(
            (
                "conversation_id",
                "ALTER TABLE chat_messages "
                "ADD COLUMN conversation_id VARCHAR(100)",
            )
        )

    if "account_id" not in columns:
        migrations.append(
            (
                "account_id",
                "ALTER TABLE chat_messages "
                "ADD COLUMN account_id VARCHAR(50)",
            )
        )

    if "rca" not in columns:
        migrations.append(
            (
                "rca",
                "ALTER TABLE chat_messages "
                "ADD COLUMN rca TEXT",
            )
        )

    if "recommendations" not in columns:
        migrations.append(
            (
                "recommendations",
                "ALTER TABLE chat_messages "
                "ADD COLUMN recommendations TEXT",
            )
        )

    if not migrations:
        return

    with engine.begin() as connection:
        for _, sql in migrations:
            connection.execute(text(sql))

        # Existing records were created before these fields existed.
        # Use session_id as a safe fallback for account_id and
        # conversation_id so old chat history remains accessible.
        if "conversation_id" not in columns:
            connection.execute(
                text(
                    "UPDATE chat_messages "
                    "SET conversation_id = session_id "
                    "WHERE conversation_id IS NULL"
                )
            )

        if "account_id" not in columns:
            connection.execute(
                text(
                    "UPDATE chat_messages "
                    "SET account_id = session_id "
                    "WHERE account_id IS NULL"
                )
            )


migrate_chat_messages()


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="AWS AI Agent",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# HELPERS
# =====================================================

def validate_aws_session(session_id: str) -> Dict[str, Any]:
    """
    Validate that an AWS session exists.
    """

    aws_session = AWS_SESSIONS.get(session_id)

    if not aws_session:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid session_id. "
                "Please connect to AWS first."
            ),
        )

    if not aws_session.get("account_id"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Account ID not found for "
                "the given session_id."
            ),
        )

    return aws_session


def model_to_dict(model: Any) -> Dict[str, Any]:
    """
    Support Pydantic v2 and v1.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
    return {
        "message": "AWS AI Agent API is running"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =====================================================
# AWS CONNECTION
# =====================================================

@app.post(
    "/aws/connect",
    response_model=AWSConnectResponse,
)
def aws_connect(
    request: AWSConnectRequest,
):
    session_id = str(uuid.uuid4())

    try:
        return connect_aws(
            session_id=session_id,
            access_key=request.access_key,
            secret_key=request.secret_key,
            region=request.region,
            role_arn=request.role_arn,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"AWS connection failed: {exc}",
        )


# =====================================================
# CHAT
# =====================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    aws_session = validate_aws_session(
        request.session_id
    )

    account_id = aws_session.get("account_id")

    previous_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == request.conversation_id,
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    history = [
        {
            "user": record.user_message,
            "assistant": record.assistant_message,
        }
        for record in previous_messages
    ]

    try:
        result = run_agent(
            session_id=request.session_id,
            query=request.message,
            history=history,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    record = ChatMessage(
        account_id=account_id,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        user_message=request.message,
        assistant_message=result["answer"],
        intent=result["intent"],
        service=result.get("service"),
        rca=result.get("rca"),
        recommendations=result.get(
            "recommendations"
        ),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    # ACTION responses are planned by the agent and stored in the existing
    # pending-action store. No AWS execution happens here.
    action_plan = result.get("action_plan") or {}
    requires_confirmation = bool(
        result.get("intent") == "ACTION"
        and action_plan.get("validated") is True
        and action_plan.get("action_id")
    )

    return {
        **result,
        "action_plan": action_plan,
        "action_validation": result.get("action_validation"),
        "requires_confirmation": requires_confirmation,
    }


# =====================================================
# ALL CONVERSATIONS
# =====================================================

@app.get(
    "/conversations/{account_id}"
)
def conversations(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id
        )
        .order_by(
            ChatMessage.created_at.desc()
        )
        .all()
    )

    conversations_data = {}
    ordered_ids = []

    for record in records:
        conversation_id = (
            record.conversation_id
            or record.session_id
        )

        if conversation_id not in conversations_data:
            conversations_data[
                conversation_id
            ] = {
                "conversation_id": conversation_id,
                "title": record.user_message,
                "created_at": str(
                    record.created_at
                ),
                "updated_at": str(
                    record.created_at
                ),
            }

            ordered_ids.append(conversation_id)

        else:
            conversations_data[
                conversation_id
            ]["updated_at"] = str(
                record.created_at
            )

    return [
        conversations_data[conversation_id]
        for conversation_id in ordered_ids
    ]


# =====================================================
# CONVERSATION HISTORY
# =====================================================

@app.get(
    "/history/{account_id}/{conversation_id}"
)
def conversation_history(
    account_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == conversation_id,
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "intent": record.intent,
            "service": record.service,
            "rca": record.rca,
            "recommendations": record.recommendations,
            "created_at": str(record.created_at),
        }
        for record in records
    ]


# =====================================================
# DELETE CONVERSATION
# =====================================================

@app.delete(
    "/history/{account_id}/{conversation_id}"
)
def delete_conversation(
    account_id: str,
    conversation_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id,
            ChatMessage.conversation_id
            == conversation_id,
        )
        .all()
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    for record in records:
        db.delete(record)

    db.commit()

    return {
        "success": True,
        "message": (
            "Conversation deleted successfully."
        ),
        "conversation_id": conversation_id,
    }


# =====================================================
# LEGACY ACCOUNT HISTORY
# =====================================================

@app.get(
    "/history/{account_id}"
)
def history(
    account_id: str,
    db: Session = Depends(get_db),
):
    records = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.account_id == account_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "user_message": record.user_message,
            "assistant_message": record.assistant_message,
            "intent": record.intent,
            "service": record.service,
            "rca": record.rca,
            "recommendations": record.recommendations,
            "created_at": str(record.created_at),
        }
        for record in records
    ]


# =====================================================
# LIVE AWS RESOURCE LISTING
# =====================================================

@app.get("/aws/resources/{session_id}")
def list_aws_resources(session_id: str, service: str):
    """Return live resources for the Resource Management page."""

    validate_aws_session(session_id)

    loaders = {
        "ec2": get_ec2_instances,
        "s3": get_s3_buckets,
        "rds": get_rds_instances,
        "lambda": get_lambda_functions,
        "vpc": get_vpcs,
        "subnet": get_subnets,
        "security_group": get_security_groups,
        "iam": get_iam_resources,
    }

    loader = loaders.get(service)
    if loader is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported resource listing service: {service}",
        )

    try:
        raw_resources = loader(session_id)
        resources = []

        for item in raw_resources or []:
            if service == "ec2":
                resource_id = item.get("instance_id")
            elif service == "s3":
                resource_id = item.get("name")
            elif service == "rds":
                resource_id = item.get("identifier")
            elif service == "lambda":
                resource_id = item.get("function_name") or item.get("name")
            elif service == "vpc":
                resource_id = item.get("vpc_id")
            elif service == "subnet":
                resource_id = item.get("subnet_id")
            elif service == "iam":
                # IAM resources return their identifier in the generic `id` field.
                # IAM is global and its resource kinds (user/role/group) are
                # distinguished in the resource details.
                resource_id = item.get("id") or item.get("name")
            else:
                resource_id = item.get("group_id")

            if not resource_id:
                continue

            resources.append({
                "id": resource_id,
                "name": item.get("name") or resource_id,
                "details": item,
            })

        return {"service": service, "resources": resources}

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not load live AWS resources: {exc}",
        )


# =====================================================
# INDIVIDUAL AWS ACTION PLANNING
# =====================================================

@app.post("/aws/action/plan")
def plan_aws_action(
    request: AWSActionRequest,
):
    """
    Validate and store one pending AWS action.
    This endpoint does not execute the action.
    """

    validate_aws_session(request.session_id)

    is_valid, message = validate_action(
        request.action,
        request.resource_type,
        request.parameters,
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    action = create_pending_action(
        session_id=request.session_id,
        action=request.action,
        resource_type=request.resource_type,
        parameters=request.parameters,
        explanation=(
            f"Proposed {request.action} operation "
            f"for {request.resource_type}."
        ),
    )

    return {
        "message": (
            "Action created and awaiting confirmation"
        ),
        "action": action,
        "requires_confirmation": True,
    }


# =====================================================
# INDIVIDUAL AWS ACTION CONFIRMATION
# =====================================================

REQUIRED_CONFIRMATION_PHRASE = (
    "I CONFIRM THIS AWS ACTION"
)


@app.post("/aws/action/confirm")
def confirm_aws_action(
    request: AWSActionApprovalRequest,
):
    """
    Confirm and execute one pending AWS action.
    """

    validate_aws_session(request.session_id)

    action = get_pending_action(
        request.action_id
    )

    if not action:
        raise HTTPException(
            status_code=404,
            detail="Action not found",
        )

    if action["session_id"] != request.session_id:
        raise HTTPException(
            status_code=403,
            detail=(
                "Action does not belong to "
                "this session"
            ),
        )

    if action["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Action is already "
                f"{action['status']}"
            ),
        )

    if (
        request.confirmation_phrase
        != REQUIRED_CONFIRMATION_PHRASE
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid confirmation phrase. "
                "Type exactly: "
                f"{REQUIRED_CONFIRMATION_PHRASE}"
            ),
        )

    try:
        result = execute_aws_action(
            session_id=request.session_id,
            action=action["action"],
            resource_type=action["resource_type"],
            parameters=action["parameters"],
        )

        approved_action = approve_pending_action(
            request.action_id
        )

        approved_action["status"] = "executed"

        return {
            "message": (
                "AWS action executed successfully"
            ),
            "action": approved_action,
            "execution_status": "completed",
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "AWS action execution failed: "
                f"{str(exc)}"
            ),
        )


# =====================================================
# RCA RECOMMENDED ACTION BATCH PLANNING
# =====================================================

@app.post("/aws/rca/batch/plan")
def plan_rca_action_batch(
    request: RCARecommendedActionBatch,
):
    """
    Create a batch of RCA remediation actions.

    All actions are validated before being stored.
    Nothing is executed at this stage.
    """

    validate_aws_session(request.session_id)

    if not request.actions:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one remediation action "
                "is required."
            ),
        )

    validated_actions = []

    for action in request.actions:
        action_data = model_to_dict(action)

        is_valid, message = validate_action(
            action_data["action"],
            action_data["resource_type"],
            action_data["parameters"],
        )

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid action "
                    f"{action_data.get('action_id')}: "
                    f"{message}"
                ),
            )

        validated_actions.append(
            action_data
        )

    batch = create_action_batch(
        session_id=request.session_id,
        issue_id=request.issue_id,
        issue_title=request.issue_title,
        issue_description=request.issue_description,
        actions=validated_actions,
    )

    return {
        "message": (
            "RCA remediation batch created. "
            "One confirmation is required "
            "for the complete batch."
        ),
        "batch": batch,
        "requires_confirmation": True,
        "confirmation_phrase": (
            REQUIRED_CONFIRMATION_PHRASE
        ),
    }


# =====================================================
# GET RCA ACTION BATCH STATUS
# =====================================================

@app.get("/aws/rca/batch/{batch_id}")
def get_rca_action_batch(
    batch_id: str,
    session_id: str,
):
    """
    Return the current status of an RCA action batch.
    """

    validate_aws_session(session_id)

    batch = get_action_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=404,
            detail="RCA action batch not found.",
        )

    if not batch_belongs_to_session(
        batch_id,
        session_id,
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "This action batch does not "
                "belong to the current session."
            ),
        )

    return {
        "batch": batch
    }


# =====================================================
# RCA ACTION BATCH CONFIRMATION AND EXECUTION
# =====================================================

@app.post("/aws/rca/batch/confirm")
def confirm_rca_action_batch(
    request: RCABatchApprovalRequest,
):
    """
    Confirm and execute all RCA remediation actions.

    Important behavior:
    - Only one confirmation is required.
    - Actions execute sequentially.
    - Each action status is updated.
    - Execution stops if one action fails.
    - Remaining actions are marked as skipped.
    """

    validate_aws_session(request.session_id)

    batch = get_action_batch(
        request.batch_id
    )

    if not batch:
        raise HTTPException(
            status_code=404,
            detail="RCA action batch not found.",
        )

    if not batch_belongs_to_session(
        request.batch_id,
        request.session_id,
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "This action batch does not "
                "belong to the current session."
            ),
        )

    if batch["status"] not in [
        "pending",
        "approved",
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch is already "
                f"{batch['status']}."
            ),
        )

    if (
        request.confirmation_phrase
        != REQUIRED_CONFIRMATION_PHRASE
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid confirmation phrase. "
                "Type exactly: "
                f"{REQUIRED_CONFIRMATION_PHRASE}"
            ),
        )

    # Validate every action again before execution.
    for action in batch["actions"]:
        is_valid, message = validate_action(
            action["action"],
            action["resource_type"],
            action["parameters"],
        )

        if not is_valid:
            update_batch_status(
                request.batch_id,
                "failed",
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "Batch validation failed: "
                    f"{message}"
                ),
            )

    approve_action_batch(
        request.batch_id
    )

    update_batch_status(
        request.batch_id,
        "running",
    )

    execution_results = []

    for action in batch["actions"]:
        action_id = action.get(
            "action_id"
        )

        update_batch_action_status(
            batch_id=request.batch_id,
            action_id=action_id,
            status="running",
        )

        try:
            result = execute_aws_action(
                session_id=request.session_id,
                action=action["action"],
                resource_type=action[
                    "resource_type"
                ],
                parameters=action[
                    "parameters"
                ],
            )

            update_batch_action_status(
                batch_id=request.batch_id,
                action_id=action_id,
                status="completed",
                result=result,
            )

            execution_results.append(
                {
                    "action_id": action_id,
                    "resource_type": action[
                        "resource_type"
                    ],
                    "status": "completed",
                    "result": result,
                }
            )

        except Exception as exc:
            error_message = str(exc)

            update_batch_action_status(
                batch_id=request.batch_id,
                action_id=action_id,
                status="failed",
                error=error_message,
            )

            execution_results.append(
                {
                    "action_id": action_id,
                    "resource_type": action[
                        "resource_type"
                    ],
                    "status": "failed",
                    "error": error_message,
                }
            )

            # Stop execution after the first failure.
            remaining_actions = batch["actions"]

            current_index = next(
                (
                    index
                    for index, item
                    in enumerate(remaining_actions)
                    if item.get("action_id")
                    == action_id
                ),
                -1,
            )

            if current_index >= 0:
                for remaining_action in (
                    remaining_actions[
                        current_index + 1:
                    ]
                ):
                    remaining_action_id = (
                        remaining_action.get(
                            "action_id"
                        )
                    )

                    update_batch_action_status(
                        batch_id=request.batch_id,
                        action_id=remaining_action_id,
                        status="skipped",
                        error=(
                            "Skipped because a "
                            "previous action failed."
                        ),
                    )

                    execution_results.append(
                        {
                            "action_id": (
                                remaining_action_id
                            ),
                            "resource_type": (
                                remaining_action[
                                    "resource_type"
                                ]
                            ),
                            "status": "skipped",
                            "error": (
                                "Skipped because a "
                                "previous action failed."
                            ),
                        }
                    )

            update_batch_status(
                request.batch_id,
                "failed",
            )

            return {
                "message": (
                    "RCA remediation stopped "
                    "because an action failed."
                ),
                "batch_id": request.batch_id,
                "execution_status": "failed",
                "results": execution_results,
                "batch": get_action_batch(
                    request.batch_id
                ),
            }

    update_batch_status(
        request.batch_id,
        "completed",
    )

    return {
        "message": (
            "All RCA remediation actions "
            "executed successfully."
        ),
        "batch_id": request.batch_id,
        "execution_status": "completed",
        "results": execution_results,
        "batch": get_action_batch(
            request.batch_id
        ),
    }
# =====================================================
# LIVE RESOURCE DETAIL AND CONTENT APIs
# =====================================================

import base64
from datetime import datetime
from typing import Optional

from fastapi import Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.aws_auth import get_aws_client


class S3UploadRequest(BaseModel):
    bucket_name: str = Field(min_length=3, max_length=63)
    object_key: str = Field(min_length=1, max_length=1024)
    content_base64: str = Field(min_length=1)
    content_type: str = Field(default="application/octet-stream", max_length=255)


class S3DownloadRequest(BaseModel):
    bucket_name: str = Field(min_length=3, max_length=63)
    object_key: str = Field(min_length=1, max_length=1024)


def _resource_client(session_id: str, service_name: str):
    validate_aws_session(session_id)
    return get_aws_client(session_id=session_id, service_name=service_name)


@app.get("/aws/resource/details/{session_id}")
def get_resource_details(
    session_id: str,
    service: str = Query(...),
    resource_id: str = Query(..., min_length=1),
):
    """Return fresh technical details for one selected AWS resource."""
    try:
        if service == "ec2":
            client = _resource_client(session_id, "ec2")
            response = client.describe_instances(InstanceIds=[resource_id])
            instances = [
                instance
                for reservation in response.get("Reservations", [])
                for instance in reservation.get("Instances", [])
            ]
            if not instances:
                raise HTTPException(status_code=404, detail="EC2 instance not found")
            return {"service": service, "resource_id": resource_id, "details": instances[0]}

        if service == "s3":
            client = _resource_client(session_id, "s3")
            location = client.get_bucket_location(Bucket=resource_id).get("LocationConstraint")
            return {
                "service": service,
                "resource_id": resource_id,
                "details": {
                    "bucket_name": resource_id,
                    "region": location or "us-east-1",
                    "head": client.head_bucket(Bucket=resource_id),
                },
            }

        if service == "rds":
            client = _resource_client(session_id, "rds")
            response = client.describe_db_instances(DBInstanceIdentifier=resource_id)
            instances = response.get("DBInstances", [])
            if not instances:
                raise HTTPException(status_code=404, detail="RDS instance not found")
            return {"service": service, "resource_id": resource_id, "details": instances[0]}

        if service == "lambda":
            client = _resource_client(session_id, "lambda")
            response = client.get_function_configuration(FunctionName=resource_id)
            return {"service": service, "resource_id": resource_id, "details": response}

        if service == "vpc":
            client = _resource_client(session_id, "ec2")
            response = client.describe_vpcs(VpcIds=[resource_id])
            vpcs = response.get("Vpcs", [])
            if not vpcs:
                raise HTTPException(status_code=404, detail="VPC not found")
            return {"service": service, "resource_id": resource_id, "details": vpcs[0]}

        if service == "subnet":
            client = _resource_client(session_id, "ec2")
            response = client.describe_subnets(SubnetIds=[resource_id])
            subnets = response.get("Subnets", [])
            if not subnets:
                raise HTTPException(status_code=404, detail="Subnet not found")
            return {"service": service, "resource_id": resource_id, "details": subnets[0]}

        if service == "security_group":
            client = _resource_client(session_id, "ec2")
            response = client.describe_security_groups(GroupIds=[resource_id])
            groups = response.get("SecurityGroups", [])
            if not groups:
                raise HTTPException(status_code=404, detail="Security group not found")
            return {"service": service, "resource_id": resource_id, "details": groups[0]}

        raise HTTPException(status_code=400, detail=f"Unsupported service: {service}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load resource details: {exc}")


@app.get("/aws/s3/objects/{session_id}")
def list_s3_objects(
    session_id: str,
    bucket_name: str = Query(..., min_length=3, max_length=63),
    prefix: str = Query(default="", max_length=1024),
    max_keys: int = Query(default=100, ge=1, le=1000),
):
    """List objects inside a selected S3 bucket."""
    try:
        client = _resource_client(session_id, "s3")
        response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys)
        raw_objects = response.get("Contents", [])
        # Normalize AWS's PascalCase response fields for the frontend.
        objects = [
            {
                "key": item.get("Key"),
                "size": item.get("Size", 0),
                "etag": item.get("ETag"),
                "last_modified": (
                    item.get("LastModified").isoformat()
                    if item.get("LastModified") is not None
                    else None
                ),
                "storage_class": item.get("StorageClass"),
            }
            for item in raw_objects
            if item.get("Key")
        ]
        return {
            "bucket_name": bucket_name,
            "prefix": prefix,
            "is_truncated": response.get("IsTruncated", False),
            "objects": objects,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not list S3 objects: {exc}")


@app.post("/aws/s3/upload/{session_id}")
def upload_s3_object(session_id: str, request: S3UploadRequest):
    """Upload a base64-encoded file to an S3 bucket."""
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload is limited to 10 MB")

        client = _resource_client(session_id, "s3")
        response = client.put_object(
            Bucket=request.bucket_name,
            Key=request.object_key,
            Body=content,
            ContentType=request.content_type,
        )
        return {
            "success": True,
            "bucket_name": request.bucket_name,
            "object_key": request.object_key,
            "etag": response.get("ETag"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not upload S3 object: {exc}")


@app.post("/aws/s3/download/{session_id}")
def download_s3_object(session_id: str, request: S3DownloadRequest):
    """Return an S3 object as base64 so the Streamlit frontend can save it."""
    try:
        client = _resource_client(session_id, "s3")
        response = client.get_object(Bucket=request.bucket_name, Key=request.object_key)
        content = response["Body"].read()
        return {
            "bucket_name": request.bucket_name,
            "object_key": request.object_key,
            "content_type": response.get("ContentType") or "application/octet-stream",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "content_length": len(content),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not download S3 object: {exc}")


@app.get("/aws/cloudwatch/logs/{session_id}")
def get_cloudwatch_logs(
    session_id: str,
    log_group_name: str = Query(..., min_length=1, max_length=512),
    log_stream_name: Optional[str] = Query(default=None, max_length=512),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Read recent CloudWatch log events for a supplied log group/stream."""
    try:
        client = _resource_client(session_id, "logs")
        if not log_stream_name:
            streams_response = client.describe_log_streams(
                logGroupName=log_group_name,
                orderBy="LastEventTime",
                descending=True,
                limit=1,
            )
            streams = streams_response.get("logStreams", [])
            if not streams:
                return {"log_group_name": log_group_name, "events": [], "message": "No log streams found"}
            log_stream_name = streams[0].get("logStreamName")

        response = client.get_log_events(
            logGroupName=log_group_name,
            logStreamName=log_stream_name,
            limit=limit,
            startFromHead=False,
        )
        return {
            "log_group_name": log_group_name,
            "log_stream_name": log_stream_name,
            "events": response.get("events", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load CloudWatch logs: {exc}")

# =====================================================
# S3 OBJECT DELETE
# =====================================================

@app.delete("/aws/s3/objects/{session_id}")
def delete_s3_object(
    session_id: str,
    bucket_name: str = Query(..., min_length=3, max_length=63),
    object_key: str = Query(..., min_length=1, max_length=1024),
):
    """Delete one object from an S3 bucket after frontend confirmation."""
    try:
        client = _resource_client(session_id, "s3")

        client.delete_object(
            Bucket=bucket_name,
            Key=object_key,
        )

        return {
            "success": True,
            "bucket_name": bucket_name,
            "object_key": object_key,
            "message": "S3 object deleted successfully.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not delete S3 object: {exc}",
        )

