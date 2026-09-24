from typing import Any, Dict, Tuple


ALLOWED_ACTIONS = {
    "create",
    "delete",
    "start",
    "stop",
    "reboot",
    "enable",
    "disable",
    "upload",
    "download",
}


RESOURCE_REQUIRED_FIELDS = {
    "s3_bucket": {
        "create": ["bucket_name"],
        "delete": ["bucket_name", "delete_confirmation"],
    },
    "s3_object": {
        "upload": ["bucket_name", "object_key", "file_path"],
        "download": ["bucket_name", "object_key", "file_path"],
        "delete": ["bucket_name", "object_key", "delete_confirmation"],
    },
    "ec2_instance": {
        "start": ["instance_id"],
        "stop": ["instance_id"],
        "reboot": ["instance_id"],
        "create": [
            "resource_name",
            "ami_id",
            "instance_type",
            "key_name",
        ],
        "delete": ["instance_id", "delete_confirmation"],
    },
    "rds_instance": {
        "start": ["db_instance_identifier"],
        "stop": ["db_instance_identifier"],
        "create": [
            "db_instance_identifier",
            "db_instance_class",
            "engine",
            "master_username",
            "master_password",
        ],
        "delete": ["db_instance_identifier", "delete_confirmation"],
    },
    "lambda_function": {
        "enable": ["function_name"],
        "disable": ["function_name"],
        "create": [
            "function_name",
            "runtime",
            "role_arn",
            "handler",
            "zip_file",
        ],
        "delete": ["function_name", "delete_confirmation"],
    },
    "security_group": {
        "create": ["group_name", "description", "vpc_id"],
        "delete": ["group_id", "delete_confirmation"],
    },
    "vpc": {
        "create": ["resource_name", "cidr_block"],
        "delete": ["vpc_id", "delete_confirmation"],
    },
    "subnet": {
        "create": [
            "resource_name",
            "vpc_id",
            "cidr_block",
            "availability_zone",
        ],
        "delete": ["subnet_id", "delete_confirmation"],
    },
    "iam_resource": {
        "create": ["resource_name", "resource_kind"],
        "delete": [
            "resource_name",
            "resource_kind",
            "delete_confirmation",
        ],
    },
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _get_delete_identifier(
    resource_type: str,
    parameters: Dict[str, Any],
) -> str:
    if resource_type == "s3_object":
        bucket_name = str(parameters.get("bucket_name", "")).strip()
        object_key = str(parameters.get("object_key", "")).strip()
        if not bucket_name or not object_key:
            return ""
        return f"{bucket_name}/{object_key}"

    identifier_fields = {
        "s3_bucket": "bucket_name",
        "ec2_instance": "instance_id",
        "rds_instance": "db_instance_identifier",
        "lambda_function": "function_name",
        "security_group": "group_id",
        "vpc": "vpc_id",
        "subnet": "subnet_id",
        "iam_resource": "resource_name",
    }
    identifier_field = identifier_fields.get(resource_type)
    if not identifier_field:
        return ""
    return str(parameters.get(identifier_field, "")).strip()


def validate_action(
    action: str,
    resource_type: str,
    parameters: Dict[str, Any],
) -> Tuple[bool, str]:
    if not isinstance(action, str) or not action.strip():
        return False, "Action must be a non-empty string"
    if not isinstance(resource_type, str) or not resource_type.strip():
        return False, "Resource type must be a non-empty string"

    action = action.strip().lower()
    resource_type = resource_type.strip().lower()

    if action not in ALLOWED_ACTIONS:
        return False, f"Unsupported action: {action}"

    if resource_type not in RESOURCE_REQUIRED_FIELDS:
        return False, f"Unsupported resource type: {resource_type}"

    if not isinstance(parameters, dict):
        return False, "Parameters must be a dictionary"

    resource_actions = RESOURCE_REQUIRED_FIELDS[resource_type]
    if action not in resource_actions:
        return False, (
            f"Action '{action}' is not supported for resource type "
            f"'{resource_type}'"
        )

    required_fields = resource_actions[action]
    missing_fields = [
        field
        for field in required_fields
        if field not in parameters or _is_missing(parameters[field])
    ]

    if missing_fields:
        return False, "Missing required fields: " + ", ".join(missing_fields)

    if action == "delete":
        resource_identifier = _get_delete_identifier(
            resource_type,
            parameters,
        )
        if not resource_identifier:
            return False, (
                "Unable to determine the resource identifier "
                "for deletion confirmation"
            )

        delete_confirmation = str(
            parameters.get("delete_confirmation", "")
        ).strip()

        if delete_confirmation != resource_identifier:
            return False, (
                "delete_confirmation must exactly match "
                f"'{resource_identifier}'"
            )

    return True, "Action is valid"
