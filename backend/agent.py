import json
import os
import inspect
import re
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from backend.aws_tools import (
    get_ec2_instances,
    get_rds_instances,
    get_s3_buckets,
    get_s3_storage_summary,
    get_s3_objects,
    get_vpcs,
    get_subnets,
    get_internet_gateways,
    get_route_tables,
    get_security_groups,
    get_cost_by_service,
    get_cost_summary,
    get_patch_status,
    get_lambda_functions,
    get_cloudwatch_metrics,
    get_cloudtrail_events,
    get_inspector_findings,
    get_resource_tags,
    get_ec2_tags,
    get_s3_tags,
    get_lambda_tags,
    get_cloudwatch_alarms,
    get_cloudwatch_logs
)

from backend.rag import retrieve_context


# =====================================================
# TOOL MAP
# =====================================================

TOOL_MAP = {
    "get_ec2_instances": get_ec2_instances,
    "get_s3_buckets": get_s3_buckets,
    "get_s3_storage_summary": get_s3_storage_summary,
    "get_s3_objects": get_s3_objects,
    "get_rds_instances": get_rds_instances,
    "get_vpcs": get_vpcs,
    "get_subnets": get_subnets,
    "get_internet_gateways": get_internet_gateways,
    "get_route_tables": get_route_tables,
    "get_security_groups": get_security_groups,
    "get_cost_summary": get_cost_summary,
    "get_cost_by_service": get_cost_by_service,
    "get_patch_status": get_patch_status,
    "get_lambda_functions": get_lambda_functions,
    "get_cloudwatch_metrics": get_cloudwatch_metrics,
    "get_cloudtrail_events": get_cloudtrail_events,
    "get_inspector_findings": get_inspector_findings,
    "get_resource_tags": get_resource_tags,
    "get_ec2_tags": get_ec2_tags,
    "get_s3_tags": get_s3_tags,
    "get_lambda_tags": get_lambda_tags,
    "get_cloudwatch_alarms": get_cloudwatch_alarms,
    "get_cloudwatch_logs": get_cloudwatch_logs
}


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict, total=False):
    session_id: str
    query: str

    # Conversation memory
    history: list[dict[str, str]]
    resolved_query: str

    # Existing agent state
    intent: str
    tools: list[str]
    tool_parameters: dict
    context: str
    service: str
    tool_result: str
    rca: str
    recommendations: str
    answer: str

    # ACTION intent state
    action_plan: dict
    action_validation: str


# =====================================================
# CONVERSATION CONTEXT
# =====================================================

def _get_history_value(message, *keys):
    """Read a history field while supporting the existing DB/API formats."""
    for key in keys:
        value = message.get(key)
        if value:
            return str(value).strip()
    return ""


def get_last_turn(history: list[dict[str, str]]) -> tuple[str, str]:
    """Return only the immediately previous user/assistant turn."""
    if not history:
        return "", ""

    previous = history[-1]
    user_message = _get_history_value(
        previous, "user", "user_message", "question"
    )
    assistant_message = _get_history_value(
        previous, "assistant", "assistant_message", "answer"
    )

    return user_message, assistant_message


def format_last_turn(history: list[dict[str, str]]) -> str:
    """Format only the immediately previous turn for context resolution."""
    user_message, assistant_message = get_last_turn(history)

    if not user_message and not assistant_message:
        return "No previous conversation."

    if len(user_message) > 3000:
        user_message = user_message[:3000] + "..."

    if len(assistant_message) > 5000:
        assistant_message = assistant_message[:5000] + "..."

    return (
        f"Previous user question:\n{user_message}\n\n"
        f"Previous assistant answer:\n{assistant_message}"
    )


def format_relevant_context(history: list[dict[str, str]]) -> str:
    """
    Return only the previous turn.

    The planner decides whether the current query is actually a follow-up.
    We intentionally do NOT inject the complete chat history into every LLM
    prompt, because that causes unrelated old topics to leak into answers.
    """
    return format_last_turn(history)


# =====================================================
# LLM
# =====================================================

# =====================================================
# MISTRAL LLM
# =====================================================
# Set these in .env / Railway Variables:
# MISTRAL_API_KEY=your_mistral_api_key
# MISTRAL_MODEL=ministral-3b-2512

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "ministral-3b-2512"
).strip()

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not configured. "
        "Add it to your .env file or Railway Variables."
    )

# Main LLM for RCA, recommendations and final answers.
llm = ChatOpenAI(
    model=MISTRAL_MODEL,
    temperature=0,
    max_tokens=4096,
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

# Planner only needs to return a small JSON object.
# Keeping this at 1024 avoids unnecessarily reserving output tokens.
planner_llm = ChatOpenAI(
    model=MISTRAL_MODEL,
    temperature=0,
    max_tokens=1024,
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)


# =====================================================
# PLANNER NODE
# =====================================================


def _clean_json_text(raw: str) -> str:
    """Remove common Markdown wrappers around an LLM JSON response."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_object(raw: str):
    """Parse the first JSON object from an LLM response."""
    text = _clean_json_text(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM did not return a valid JSON object")


def _invoke_json(prompt: str, purpose: str = "planner") -> dict:
    """Invoke the planner and repair malformed JSON once through the LLM."""
    response = planner_llm.invoke(prompt)
    raw = str(response.content or "").strip()

    try:
        parsed = _parse_json_object(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as first_error:
        repair_prompt = f"""
You are a strict JSON repair layer for an AWS planning system.

The previous model attempted to answer the {purpose} request but returned
invalid JSON. Repair it without changing its meaning.
Do not invent AWS resource values, IDs, credentials, files, or parameters.
Return ONLY one valid JSON object. No Markdown and no explanation.

ORIGINAL REQUEST/PROMPT:
{prompt}

INVALID MODEL OUTPUT:
{raw}

Return the corrected JSON object now.
"""
        repair_response = planner_llm.invoke(repair_prompt)
        repaired_raw = str(repair_response.content or "").strip()
        try:
            repaired = _parse_json_object(repaired_raw)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            raise ValueError(
                f"{purpose.capitalize()} planner returned invalid JSON."
            ) from first_error

    raise ValueError(f"{purpose.capitalize()} planner returned invalid JSON.")


def _required_tool_parameters(tool_names: list[str]) -> dict[str, list[str]]:
    """Discover required tool parameters from the actual Python call signatures."""
    requirements = {}
    for tool_name in tool_names:
        tool_function = TOOL_MAP.get(tool_name)
        if tool_function is None:
            continue
        try:
            signature = inspect.signature(tool_function)
        except (TypeError, ValueError):
            continue

        required = []
        for parameter in signature.parameters.values():
            if parameter.name == "session_id":
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
        if required:
            requirements[tool_name] = required
    return requirements


def _repair_tool_parameters(
    query: str,
    resolved_query: str,
    history_text: str,
    tools: list[str],
    current_parameters: dict,
) -> dict:
    """Ask the LLM to fill only missing live-tool parameters from user context."""
    requirements = _required_tool_parameters(tools)
    if not requirements:
        return current_parameters

    missing = {}
    for tool_name, fields in requirements.items():
        supplied = current_parameters.get(tool_name, {})
        if not isinstance(supplied, dict):
            supplied = {}
        missing_fields = [
            field for field in fields
            if field not in supplied or supplied[field] in (None, "")
        ]
        if missing_fields:
            missing[tool_name] = missing_fields

    if not missing:
        return current_parameters

    repair_prompt = f"""
You are the parameter-resolution layer of an AWS monitoring planner.

Resolve ONLY parameters that are explicitly supported by the current user
request or the immediately previous turn. Never invent values.
The selected tools and their required parameters are:
{json.dumps(missing, indent=2)}

CURRENT USER REQUEST:
{query}

RESOLVED QUERY:
{resolved_query}

PREVIOUS TURN:
{history_text}

PARAMETERS ALREADY SUPPLIED:
{json.dumps(current_parameters, indent=2, default=str)}

Return ONLY this JSON shape:
{{
  "tool_parameters": {{
    "tool_name": {{"parameter": "value"}}
  }}
}}

For example, if the request says "files in bucket abc", return:
{{
  "tool_parameters": {{
    "get_s3_objects": {{"bucket_name": "abc"}}
  }}
}}
If a required value is not actually present in the request/context, leave
that field absent. Do not guess.
"""

    repaired = _invoke_json(repair_prompt, "monitoring parameter")
    additions = repaired.get("tool_parameters", {})
    if not isinstance(additions, dict):
        return current_parameters

    merged = dict(current_parameters)
    for tool_name, params in additions.items():
        if tool_name not in TOOL_MAP or not isinstance(params, dict):
            continue
        existing = merged.get(tool_name, {})
        if not isinstance(existing, dict):
            existing = {}
        existing = dict(existing)
        for key, value in params.items():
            if value is not None and str(value).strip() != "":
                existing[str(key)] = value
        merged[tool_name] = existing
    return merged

def planner_node(state):

    history_text = format_relevant_context(state.get("history", []))

    prompt = f"""
You are an AWS planning agent.

You are given the current user query and the previous conversation.

Use ONLY the immediately previous user/assistant turn when deciding whether
the current message is a follow-up.

IMPORTANT CONTEXT RULES:
1. A new standalone question must be treated as a new question.
2. Only use the previous turn when the current message clearly refers to it.
3. Resolve references such as "it", "this", "that", "these", "those",
   "the instance", "the metrics", "explain it", "explain this",
   "why", "how", "more details", etc. when they clearly refer to the
   immediately previous turn.
4. Do NOT carry information from the previous turn into a new standalone
   question merely because the previous topic was AWS.
5. If the current question is complete by itself, keep it unchanged.
6. Do not use old conversation turns to change the subject of a standalone
   current question.
7. Do not invent information that is not supported by the current question
   or the immediately previous turn.
8. IMPORTANT ACTION FOLLOW-UP RULE: if the immediately previous assistant
   message says an AWS action is missing a specific required field, and the
   current message is a short value that can supply that field, treat the
   current message as the answer to that missing field. Do not block it merely
   because the short value is not independently an AWS question.
9. Example: if the previous assistant says it still needs `bucket_name`
   for `create` + `s3_bucket`, and the current user says `kishore949840`,
   resolve the request as `Create an S3 bucket named kishore949840`.
10. This rule is context resolution, not keyword-based intent routing: use
    the previous assistant response and the current value together.

Your responsibilities:

1. Determine whether the query is AWS related.
2. Block non-AWS queries.
3. Block requests for credentials, passwords, secrets, tokens or keys but allow logs of CloudWatch and CloudTrail.
4. For valid AWS queries, determine the intent:
- KNOWLEDGE
- MONITORING
- RCA
- ACTION
5. Allow educational security questions.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUERY
=====================================================

{state["query"]}

=====================================================
RESOLVED QUERY
=====================================================

Create a standalone version of the current user request.

The resolved query must include context from the conversation when necessary.

If the current query is already complete, keep its meaning unchanged.

If the current query depends on previous conversation, resolve the missing context using the conversation history.

ACTION FOLLOW-UP EXAMPLE:
Previous conversation:
User: create s3
Assistant: I can prepare `create` for `s3_bucket`, but I still need: bucket_name.
Current query:
kishore949840
Resolved query:
Create an S3 bucket named kishore949840

When the previous assistant explicitly asks for one missing action parameter,
a short current value should be used as that parameter when it is the natural
continuation. Do not classify such a value as BLOCKED just because it is not
AWS-related on its own.

Examples:

Previous conversation:

User: What is Amazon EC2?

Assistant: Amazon EC2 is a virtual server service provided by AWS.

Current query:

What are its benefits?

The resolved query should represent:

What are the benefits of Amazon EC2?

Another example:

Previous conversation:

User: Show my EC2 instances.

Assistant: Here are the EC2 instances in the AWS account.

Current query:

Which one is running?

The resolved query should represent:

Which of the previously listed EC2 instances are currently running?

Another example:

Previous conversation:

User: What is VPC peering?

Assistant: VPC peering allows two VPCs to communicate privately.

Current query:

How does it work?

The resolved query should represent:

How does Amazon VPC peering work?

Do not invent information that is not available in the conversation.

IMPORTANT FOLLOW-UP EXAMPLES:

Previous conversation:
User: Which EC2 instance is tagged to monitor?
Assistant: The EC2 instance i-05f91e488b8956a1f is tagged Monitoring: MyMonitor.

Current query:
Explain it in 5 lines.

Resolved query:
Explain the monitoring-tagged EC2 instance i-05f91e488b8956a1f in 5 lines.

---

Previous conversation:
User: Which EC2 instance is tagged to monitor?
Assistant: The EC2 instance i-05f91e488b8956a1f is tagged Monitoring: MyMonitor.

Current query:
What is EC2?

Resolved query:
What is EC2?

IMPORTANT: "What is EC2?" is a complete standalone question.
Do NOT resolve it using the previous monitoring-instance answer.

---

Previous conversation:
User: What are the NetworkIn and NetworkOut metrics for my EC2 instances?
Assistant: NetworkIn measures data received and NetworkOut measures data sent.

Current query:
Explain them in 3 lines.

Resolved query:
Explain the NetworkIn and NetworkOut metrics for the user's EC2 instances in 3 lines.

---

Previous conversation:
User: What is VPC peering?
Assistant: VPC peering connects two VPCs privately.

Current query:
What is S3?

Resolved query:
What is S3?

IMPORTANT: This is a new standalone AWS question.

=====================================================
OUTPUT FORMAT
=====================================================

Return ONLY a valid JSON object.

For allowed AWS queries, return:

{{
    "allowed": true,
    "reason": "",
    "intent": "KNOWLEDGE | MONITORING | RCA | ACTION",
    "tools": [],
    "services": [],
    "resolved_query": "",
    "tool_parameters": {{}},
    "action": "",
    "resource_type": "",
    "parameters": {{}}
}}

For blocked queries, return:

{{
    "allowed": false,
    "reason": "Clear explanation of why the request was blocked.",
    "intent": "BLOCKED",
    "tools": [],
    "services": [],
    "resolved_query": ""
}}

=====================================================
AVAILABLE TOOLS
=====================================================

get_ec2_instances
get_s3_buckets
get_s3_storage_summary
get_s3_objects
get_rds_instances
get_vpcs
get_subnets
get_internet_gateways
get_route_tables
get_security_groups
get_cost_summary
get_cost_by_service
get_patch_status
get_lambda_functions
get_cloudwatch_metrics
get_cloudtrail_events
get_inspector_findings
get_resource_tags
get_ec2_tags
get_s3_tags
get_lambda_tags
get_cloudwatch_alarms
get_cloudwatch_logs

=====================================================
INTENT DEFINITIONS
=====================================================

KNOWLEDGE:

AWS concepts
Explanations
Documentation
Architecture
Configuration
Best practices
General AWS Learning


MONITORING:

Questions requiring live data from the user's AWS account.

IMPORTANT CLASSIFICATION RULE:
- Requests to list, show, find, inspect, count, or retrieve actual AWS resources MUST be MONITORING, not KNOWLEDGE.
- "List all VPCs and their CIDR blocks" MUST use MONITORING with tools ["get_vpcs"].
- "List all EC2 instances" MUST use MONITORING with tools ["get_ec2_instances"].
- "List all S3 buckets" MUST use MONITORING with tools ["get_s3_buckets"].
- "What files are in bucket X" MUST use MONITORING with tool get_s3_objects and pass the bucket name as that tool's parameter.
- When a monitoring tool requires a resource identifier, put the identifier in tool_parameters; do not encode it only in prose.
- The tool_parameters object MUST map each selected tool to its arguments.
- Example: for "What files are in bucket abcdegg123456kishore", return:
  "tools": ["get_s3_objects"],
  "tool_parameters": {{"get_s3_objects": {{"bucket_name": "abcdegg123456kishore"}}}}
- Do not omit tool_parameters when a selected tool has required arguments.
- KNOWLEDGE is only for conceptual or documentation questions.

This includes:
- Resource inventory
- Resource counts
- Resource status
- Resource usage
- Resource configuration
- AWS costs
- CloudWatch metrics
- CPU utilization
- NetworkIn
- NetworkOut
- Disk metrics
- Status check metrics
- CloudWatch alarms
- CloudWatch logs

If the user asks for metrics, measurements, usage,
performance data, or CloudWatch data from their AWS account,
classify the request as MONITORING unless they are explicitly
asking why a problem occurred. If they ask why a problem
occurred, classify the request as RCA.

ACTION-REQUEST CLASSIFICATION RULES:
- Requests that ask the assistant to CREATE, DELETE, START, STOP, REBOOT,
  ENABLE, DISABLE, UPLOAD, or DOWNLOAD an AWS resource/object must be ACTION.
- Requests to plan or prepare one of those executable AWS operations are ACTION.
- ACTION is for an operation the user wants performed, not for merely
  explaining how an operation works.
- A request that only lists, shows, checks, or reports status remains MONITORING.
- RCA is reserved for diagnosis, causes, failures, anomalies, and incident analysis.
- Never execute an action during planning. The action must be validated,
  presented for approval, and explicitly confirmed through the API before execution.
- Explicit examples: "create S3 bucket" is ACTION; "create EC2 instance" is ACTION;
  "upload a file to bucket X" is ACTION; "download file Y from bucket X" is ACTION.
- Missing action parameters must remain missing. Never create defaults such as a bucket name,
  AMI ID, VPC CIDR, instance type, key name, or resource name unless the user supplied them.
- Supported ACTION resource types are: s3_bucket, s3_object, ec2_instance, rds_instance,
  lambda_function, security_group, vpc, subnet, iam_resource.

TAG / ACCOUNT-DATA RULES:
- Requests for live data from the user's AWS account are MONITORING.
- Requests to list, show, find, inspect, or count actual AWS resource tags in the user's account are MONITORING.
- Phrases such as "in my account", "my AWS resources", "my instances", "my buckets", or "currently" indicate account-specific data when the user asks for actual resources or configuration.
- For account-wide tag questions, select get_resource_tags.
- For EC2 tag questions, select get_ec2_tags.
- For S3 bucket tag questions, select get_s3_tags.
- For Lambda tag questions, select get_lambda_tags.
- Conceptual questions such as "What are AWS tags?" or "How do AWS tags work?" are KNOWLEDGE.



ROOT CAUSE ANALYSIS:

Root cause analysis
Incident investigation
Failure diagnosis
Performance degradation analysis
Security finding analysis
Questions asking for causes, impact, failures or anomalies
Troubleshooting
Diagnosis of AWS problems


=====================================================
EXAMPLE
=====================================================

"Why is my EC2 instance experiencing high CPU?"

must be:

{{
    "allowed": true,
    "intent": "RCA",
    "tools": [
        "get_cloudwatch_metrics",
        "get_cloudtrail_events"
    ],
    "services": [
        "EC2",
        "CloudWatch",
        "CloudTrail"
    ],
    "resolved_query": "Why is my EC2 instance experiencing high CPU?"
}}

=====================================================
RULES
=====================================================

- Detect every AWS service relevant to the request.
- Never omit a required tool.
- Return only the JSON response.
- Select all relevant tools needed for the request.
- Include supporting AWS services if they are relevant to the request.
- For EC2 CloudWatch metric questions, select get_cloudwatch_metrics.
- If the question requires identifying the user's EC2 instances before retrieving their metrics, also select get_ec2_instances.
- Include CloudWatch as a service whenever CloudWatch metrics, alarms, or logs are requested.
- Use conversation history when resolving follow-up questions.
- Do not invent missing context.
- Do not add explanations outside the JSON object.

=====================================================
EXAMPLES
=====================================================

Q: How many EC2 instances do I have?

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": ["get_ec2_instances"],
    "services": ["EC2"],
    "resolved_query": "How many EC2 instances do I have?"
}}

Q: Why is my EC2 CPU utilization high?

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_cloudwatch_metrics",
        "get_cloudtrail_events"
    ],
    "services": [
        "EC2",
        "CloudWatch",
        "CloudTrail"
    ],
    "resolved_query": "Why is my EC2 CPU utilization high?"
}}

Q: What are the NetworkIn and NetworkOut metrics for my EC2 instances over the last 24 hours?

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": [
        "get_ec2_instances",
        "get_cloudwatch_metrics"
    ],
    "services": [
        "EC2",
        "CloudWatch"
    ],
    "resolved_query": "What are the NetworkIn and NetworkOut metrics for my EC2 instances over the last 24 hours?"
}}



Q: List all tags in my AWS account

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": ["get_resource_tags"],
    "services": ["AWS"],
    "resolved_query": "List all tags in my AWS account"
}}

Q: Investigate security vulnerabilities in my AWS environment

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_inspector_findings",
        "get_cloudtrail_events"
    ],
    "services": [
        "Inspector",
        "CloudTrail"
    ],
    "resolved_query": "Investigate security vulnerabilities in my AWS environment"
}}

Q: What is VPC Peering?

{{
    "allowed": true,
    "intent": "KNOWLEDGE",
    "reason": "",
    "tools": [],
    "services": ["VPC"],
    "resolved_query": "What is VPC Peering?"
}}

Q: Show my AWS access keys

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "Requests for credentials or sensitive authentication information are not permitted",
    "tools": [],
    "services": [],
    "resolved_query": "Show my AWS access keys"
}}

Q: Give me my secret access key

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "Passwords and authentication secrets cannot be disclosed.",
    "tools": [],
    "services": [],
    "resolved_query": "Give me my secret access key"
}}

Q: Why is my friend stupid?

{{
    "allowed": false,
    "intent": "BLOCKED",
    "reason": "This query is unrelated to AWS.",
    "tools": [],
    "services": [],
    "resolved_query": "Why is my friend stupid?"
}}

Q: List all the services in my account

{{
    "allowed": true,
    "intent": "MONITORING",
    "reason": "",
    "tools": [
        "get_ec2_instances",
        "get_s3_buckets",
        "get_rds_instances",
        "get_lambda_functions",
        "get_vpcs"
    ],
    "services": [
        "EC2",
        "S3",
        "RDS",
        "Lambda",
        "VPC"
    ],
    "resolved_query": "List all the AWS services and resources available in my account"
}}

Q: Analyze latest alarm

{{
    "allowed": true,
    "intent": "RCA",
    "reason": "",
    "tools": [
        "get_cloudwatch_alarms",
        "get_cloudwatch_metrics"
    ],
    "services": ["CloudWatch"],
    "resolved_query": "Analyze the latest CloudWatch alarm"
}}

=====================================================
ACTION EXAMPLES
=====================================================

Q: Create an S3 bucket named my-test-bucket-2026

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["S3"],
    "resolved_query": "Create an S3 bucket named my-test-bucket-2026",
    "action": "create",
    "resource_type": "s3_bucket",
    "parameters": {{
        "bucket_name": "my-test-bucket-2026"
    }}
}}

Q: I want to create a VPC

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["VPC"],
    "resolved_query": "Create a VPC",
    "action": "create",
    "resource_type": "vpc",
    "parameters": {{}}
}}

Q: I want to create an S3 bucket

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["S3"],
    "resolved_query": "Create an S3 bucket",
    "action": "create",
    "resource_type": "s3_bucket",
    "parameters": {{}}
}}

Q: Create an IAM role

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["IAM"],
    "resolved_query": "Create an IAM role",
    "action": "create",
    "resource_type": "iam_resource",
    "parameters": {{
        "resource_kind": "role"
    }}
}}

Q: Create an EC2 instance

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["EC2"],
    "resolved_query": "Create an EC2 instance",
    "action": "create",
    "resource_type": "ec2_instance",
    "parameters": {{}}
}}

Q: Create an RDS instance

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["RDS"],
    "resolved_query": "Create an RDS instance",
    "action": "create",
    "resource_type": "rds_instance",
    "parameters": {{}}
}}

Q: Create a Lambda function

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["Lambda"],
    "resolved_query": "Create a Lambda function",
    "action": "create",
    "resource_type": "lambda_function",
    "parameters": {{}}
}}

Q: Create a security group

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["EC2"],
    "resolved_query": "Create a security group",
    "action": "create",
    "resource_type": "security_group",
    "parameters": {{}}
}}

Q: Create a subnet

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["VPC"],
    "resolved_query": "Create a subnet",
    "action": "create",
    "resource_type": "subnet",
    "parameters": {{}}
}}

Q: Create a VPC named production-vpc with CIDR 10.0.0.0/16

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["VPC"],
    "resolved_query": "Create a VPC named production-vpc with CIDR 10.0.0.0/16",
    "action": "create",
    "resource_type": "vpc",
    "parameters": {{
        "resource_name": "production-vpc",
        "cidr_block": "10.0.0.0/16"
    }}
}}

Q: Delete my S3 bucket my-test-bucket-2026

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["S3"],
    "resolved_query": "Delete my S3 bucket my-test-bucket-2026",
    "action": "delete",
    "resource_type": "s3_bucket",
    "parameters": {{
        "bucket_name": "my-test-bucket-2026"
    }}
}}

Q: Start EC2 instance i-0123456789abcdef0

{{
    "allowed": true,
    "intent": "ACTION",
    "reason": "",
    "tools": [],
    "services": ["EC2"],
    "resolved_query": "Start EC2 instance i-0123456789abcdef0",
    "action": "start",
    "resource_type": "ec2_instance",
    "parameters": {{
        "instance_id": "i-0123456789abcdef0"
    }}
}}

IMPORTANT ACTION UNDERSTANDING RULE:
- Any clear request to create, delete, start, stop, reboot, enable, disable, upload, or download a supported AWS resource is ACTION.
- Do not reinterpret a state-changing request as KNOWLEDGE just because the user did not provide all fields yet.
- For create requests, identify the AWS resource from the user's language and return ACTION with the fields the user actually supplied.
- If the user says only "create an IAM role", return resource_type=iam_resource and resource_kind=role with no invented resource name.
- If the user says only "create a VPC", return resource_type=vpc with no invented name or CIDR.
- If the user says only "create an S3 bucket", return resource_type=s3_bucket with no invented bucket name.
- The ACTION node will use the same resource/action requirements as the Create Resource UI to determine missing fields.

IMPORTANT ACTION RULE:
- A state-changing AWS request is still an allowed AWS request.
- Classify it as ACTION even when required parameters are missing.
- Do NOT block an ACTION merely because a parameter is missing.
- Missing parameters must be handled by the ACTION node's validation layer.
- Do NOT invent missing parameter values.
- Do NOT execute an action during planning.
- The user must explicitly confirm through the existing confirmation API.

=====================================================
FINAL INSTRUCTION
=====================================================

Return ONLY the JSON object.
"""

    try:

        plan = _invoke_json(prompt, "planner")

        allowed = plan.get(
            "allowed",
            False
        )

        if not allowed:

            reason = plan.get(
                "reason",
                "Request Blocked"
            )

            print("Request Blocked")
            print("Reason:", reason)

            return {
                "intent": "BLOCKED",
                "answer": f"Sorry: {reason}"
            }

        intent = plan.get(
            "intent",
            ""
        ).strip().upper()

        tools = [
            tool
            for tool in plan.get(
                "tools",
                []
            )
            if tool in TOOL_MAP
        ]

        services = plan.get(
            "services",
            []
        )

        tool_parameters = plan.get("tool_parameters", {})
        if not isinstance(tool_parameters, dict):
            tool_parameters = {}

        resolved_query = plan.get(
            "resolved_query",
            state["query"]
        )

        if not isinstance(
            resolved_query,
            str
        ):
            resolved_query = state["query"]

        resolved_query = resolved_query.strip()

        if not resolved_query:
            resolved_query = state["query"]

        tool_parameters = _repair_tool_parameters(
            query=state["query"],
            resolved_query=resolved_query,
            history_text=history_text,
            tools=tools,
            current_parameters=tool_parameters,
        )

        # The LLM planner is the source of truth for intent, tools, services, and resolved query.
        # Python only validates that selected tools exist in TOOL_MAP; it does not
        # override natural-language classification with keyword rules.

        print("\n===== PLANNER OUTPUT =====")
        print("Intent:", intent)
        print("Tools:", tools)
        print("Services:", ", ".join(services))
        print("Resolved Query:", resolved_query)
        print("==========================\n")

        return {
            "intent": intent,
            "tools": tools,
            "services": ", ".join(services),
            "resolved_query": resolved_query,
            "tool_parameters": tool_parameters
        }

    except Exception as e:

        print(
            "Planner Error:",
            str(e)
        )

        return {
            "intent": "BLOCKED",
            "answer": "Sorry. Unable to classify the request."
        }


# =====================================================
# ACTION NODE
# =====================================================

def action_node(state):
    """
    Convert an ACTION request into a structured pending action.

    This node NEVER executes an AWS action. It:
    1. asks the LLM to structure the requested action,
    2. validates the parameters,
    3. creates a pending action when valid,
    4. returns the pending action id for explicit confirmation.

    Actual AWS execution remains in the existing confirmation API.
    """

    query = state.get("resolved_query", state["query"])
    history_text = format_last_turn(state.get("history", []))

    prompt = f"""
You are the AWS action-planning layer.

Create a structured action request from the user's request.
Do NOT execute anything.
Do NOT invent resource IDs, names, ARNs, credentials, passwords, or files.
Use the current request and the immediately previous turn only when the
current request clearly refers to it.
If the previous assistant asked for one missing required action parameter and
the current request is a short value, use that value to complete the action.
Example: previous assistant asks for `bucket_name` after `create s3`, current
request is `kishore949840` -> create `s3_bucket` with `bucket_name` set to
`kishore949840`. Do not invent any other fields.

SUPPORTED ACTIONS:
- create
- delete
- start
- stop
- reboot
- enable
- disable
- upload
- download

SUPPORTED RESOURCE TYPES:
- s3_bucket
- s3_object
- ec2_instance
- rds_instance
- lambda_function
- security_group
- vpc
- subnet
- iam_resource

ACTION FIELD REQUIREMENTS:
- s3_bucket create: bucket_name
- s3_bucket delete: bucket_name, delete_confirmation
- s3_object upload: bucket_name, object_key, file_path
- s3_object download: bucket_name, object_key, file_path
- s3_object delete: bucket_name, object_key, delete_confirmation
- ec2_instance start/stop/reboot: instance_id
- ec2_instance create: resource_name, ami_id, instance_type, key_name
- ec2_instance delete: instance_id, delete_confirmation
- rds_instance start/stop: db_instance_identifier
- rds_instance create: db_instance_identifier, db_instance_class, engine, master_username, master_password
- rds_instance delete: db_instance_identifier, delete_confirmation
- lambda_function enable/disable: function_name
- lambda_function create: function_name, runtime, role_arn, handler, zip_file
- lambda_function delete: function_name, delete_confirmation
- security_group create: group_name, description, vpc_id
- security_group delete: group_id, delete_confirmation
- vpc create: resource_name, cidr_block
- vpc delete: vpc_id, delete_confirmation
- subnet create: resource_name, vpc_id, cidr_block, availability_zone
- subnet delete: subnet_id, delete_confirmation
- iam_resource create: resource_name, resource_kind
- iam_resource delete: resource_name, resource_kind, delete_confirmation

For delete_confirmation, only provide it when the user explicitly supplied
that exact resource identifier as the deletion target. Do not fabricate it.

Q: What files are in bucket aws-123-unique1308

{{
  "action": "",
  "resource_type": "",
  "parameters": {{}},
  "explanation": "S3 object inventory is monitoring, not an ACTION."
}}

Q: Delete file reports/report.csv from bucket aws-123-unique1308

{{
  "action": "delete",
  "resource_type": "s3_object",
  "parameters": {{
    "bucket_name": "aws-123-unique1308",
    "object_key": "reports/report.csv",
    "delete_confirmation": "aws-123-unique1308/reports/report.csv"
  }},
  "explanation": "Delete the specified S3 object after explicit confirmation."
}}

Q: Download file reports/report.csv from bucket aws-123-unique1308 to C:/Users/me/Downloads/report.csv

{{
  "action": "download",
  "resource_type": "s3_object",
  "parameters": {{
    "bucket_name": "aws-123-unique1308",
    "object_key": "reports/report.csv",
    "file_path": "C:/Users/me/Downloads/report.csv"
  }},
  "explanation": "Download the specified S3 object to the supplied local path after explicit confirmation."
}}

Q: Upload C:/Users/me/Documents/report.csv to bucket aws-123-unique1308 as reports/report.csv

{{
  "action": "upload",
  "resource_type": "s3_object",
  "parameters": {{
    "bucket_name": "aws-123-unique1308",
    "object_key": "reports/report.csv",
    "file_path": "C:/Users/me/Documents/report.csv"
  }},
  "explanation": "Upload the specified local file to the S3 bucket after explicit confirmation."
}}

Q: Create an EC2 instance

{{
  "action": "create",
  "resource_type": "ec2_instance",
  "parameters": {{}},
  "explanation": "The user requested EC2 creation but supplied no launch parameters."
}}

Q: Create an S3 bucket

{{
  "action": "create",
  "resource_type": "s3_bucket",
  "parameters": {{}},
  "explanation": "The user requested S3 bucket creation but supplied no bucket name."
}}

Q: Create a VPC

{{
  "action": "create",
  "resource_type": "vpc",
  "parameters": {{}},
  "explanation": "The user requested VPC creation but supplied no name or CIDR block."
}}

Q: Create a security group

{{
  "action": "create",
  "resource_type": "security_group",
  "parameters": {{}},
  "explanation": "The user requested security-group creation but supplied no fields."
}}

PREVIOUS TURN:
{history_text}

CURRENT REQUEST:
{query}

Return ONLY JSON:
{{
  "action": "create|delete|start|stop|reboot|enable|disable|upload|download",
  "resource_type": "",
  "parameters": {{}},
  "explanation": "",
  "missing_fields": []
}}
"""

    try:
        parsed = _invoke_json(prompt, "action planner")

        if not isinstance(parsed, dict):
            raise ValueError("Action planner returned a non-object")

        action = str(parsed.get("action", "")).strip().lower()
        resource_type = str(
            parsed.get("resource_type", "")
        ).strip().lower()

        parameters = parsed.get("parameters", {})
        explanation = str(
            parsed.get("explanation", "")
        ).strip()

        if not isinstance(parameters, dict):
            parameters = {}

        # Security/provenance guard:
        # For a standalone destructive request such as "delete rds", do not
        # accept identifiers that the LLM may have copied from the previous
        # turn or invented from context. Previous-turn values are allowed only
        # when the current message is clearly a short answer to a missing-field
        # request (for example: "create s3" -> "my-bucket").
        raw_current_query = str(state.get("query", "")).strip()
        normalized_query = raw_current_query.lower()
        previous_text = history_text.lower()

        follow_up_to_missing_field = (
            len(raw_current_query.split()) <= 6
            and any(
                marker in previous_text
                for marker in (
                    "still need",
                    "i still need",
                    "need:",
                    "missing required",
                    "missing:",
                )
            )
        )

        standalone_action_with_no_target = (
            bool(re.search(
                r"\b(delete|start|stop|reboot|enable|disable)\b",
                normalized_query,
            ))
            and not follow_up_to_missing_field
        )

        if standalone_action_with_no_target:
            # A target must be present in the current request itself.
            # These are validation guards, not intent routing.
            target_words = {
                "delete", "start", "stop", "reboot", "enable", "disable",
                "create", "upload", "download",
                "aws", "amazon", "s3", "bucket", "object", "file",
                "ec2", "instance", "instances",
                "rds", "database", "db",
                "lambda", "function", "functions",
                "security", "group",
                "vpc", "subnet", "iam", "resource",
                "the", "a", "an", "this", "that", "my", "please",
                "from", "to", "in", "on", "of",
            }

            current_tokens = re.findall(
                r"[A-Za-z0-9_.:/\\-]+",
                normalized_query,
            )
            explicit_target_tokens = [
                token
                for token in current_tokens
                if token not in target_words
            ]

            if not explicit_target_tokens:
                parameters = {}
                explanation = (
                    explanation
                    if explanation and "previous" not in explanation.lower()
                    else ""
                )

        # Keep only supplied action parameters.
        parameters = {
            str(k): v
            for k, v in parameters.items()
            if v is not None and str(v).strip() != ""
        }

        from backend.action_validation import validate_action
        from backend.action_store import create_pending_action

        valid, message = validate_action(
            action,
            resource_type,
            parameters,
        )

        if valid:
            validation_message = "Action parameters are complete."

            # Create a pending action only.
            # The existing confirmation endpoint remains responsible for
            # approval and AWS execution.
            pending_action = create_pending_action(
                session_id=state["session_id"],
                action=action,
                resource_type=resource_type,
                parameters=parameters,
                explanation=(
                    explanation
                    or f"Proposed {action} operation for {resource_type}."
                ),
            )

            action_id = pending_action.get("action_id")

            action_plan = {
                "action": action,
                "resource_type": resource_type,
                "parameters": parameters,
                "explanation": explanation,
                "missing_fields": [],
                "validated": True,
                "action_id": action_id,
                "status": pending_action.get("status", "pending"),
            }

            parameter_lines = []
            for key, value in parameters.items():
                label = str(key).replace("_", " ").title()
                if key.lower() in {
                    "master_password",
                    "secret_key",
                    "access_key",
                    "password",
                    "token",
                    "zip_file",
                }:
                    display_value = "••••••••"
                else:
                    display_value = str(value)
                parameter_lines.append(
                    f"- **{label}:** {display_value}"
                )

            parameter_text = (
                "\n".join(parameter_lines)
                if parameter_lines
                else "- No additional parameters"
            )

            answer = (
                "## AWS Action Plan\n\n"
                f"**Action:** {action.title()}\n\n"
                f"**Resource:** {resource_type.replace('_', ' ').title()}\n\n"
                "**Parameters:**\n"
                f"{parameter_text}\n\n"
                f"**Action ID:** `{action_id}`\n\n"
                "The action has been prepared and is awaiting your "
                "explicit confirmation. No AWS action has been executed.\n\n"
                "Type exactly `I CONFIRM THIS AWS ACTION` to approve this action."
            )

        else:
            validation_message = message
            missing_fields = []

            if "Missing required field:" in message:
                missing_fields = [
                    message.split(
                        "Missing required field:", 1
                    )[1].strip()
                ]
            elif "Missing required fields:" in message:
                missing_fields = [
                    item.strip()
                    for item in message.split(
                        "Missing required fields:", 1
                    )[1].split(",")
                    if item.strip()
                ]

            action_plan = {
                "action": action,
                "resource_type": resource_type,
                "parameters": parameters,
                "explanation": explanation,
                "missing_fields": missing_fields,
                "validated": False,
                "action_id": None,
                "status": "not_created",
            }

            fields = (
                ", ".join(missing_fields)
                if missing_fields
                else validation_message
            )

            answer = (
                "## AWS Action\n\n"
                f"I can prepare `{action or 'the requested'}` for "
                f"`{resource_type or 'the AWS resource'}`, but I still "
                f"need: **{fields}**.\n\n"
                "Nothing has been executed and no pending action was created."
            )

        return {
            "action_plan": action_plan,
            "action_validation": validation_message,
            "answer": answer,
        }

    except Exception as exc:
        return {
            "action_plan": {
                "action": "",
                "resource_type": "",
                "parameters": {},
                "explanation": "",
                "missing_fields": [],
                "validated": False,
                "action_id": None,
                "status": "not_created",
            },
            "action_validation": str(exc),
            "answer": (
                "I could not safely prepare the AWS action. "
                "No AWS action was executed."
            ),
        }

# =====================================================
# KNOWLEDGE NODE
# =====================================================

def knowledge_node(state):

    query = state.get(
        "resolved_query",
        state["query"]
    )

    context = retrieve_context(
        query
    )

    return {
        "context": context
    }


# =====================================================
# MONITORING NODE
# =====================================================

def monitoring_node(state):

    session_id = state["session_id"]

    tools = state.get(
        "tools",
        []
    )

    print(
        "\nSelected Tools:",
        tools
    )

    results = {}

    for tool_name in tools:

        try:

            print(
                f"Executing {tool_name}"
            )

            tool_function = TOOL_MAP[
                tool_name
            ]

            tool_parameters = state.get("tool_parameters", {})
            parameters = tool_parameters.get(tool_name, {})
            if not isinstance(parameters, dict):
                parameters = {}

            required_fields = _required_tool_parameters([tool_name]).get(
                tool_name, []
            )
            missing_fields = [
                field
                for field in required_fields
                if field not in parameters
                or parameters[field] in (None, "")
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required parameter(s) for {tool_name}: "
                    + ", ".join(missing_fields)
                )

            results[tool_name] = tool_function(
                session_id,
                **parameters
            )

        except Exception as e:

            print(
                f"Error executing {tool_name}: {str(e)}"
            )

            results[tool_name] = {
                "error": str(e)
            }

    return {
        "tool_result": json.dumps(
            results,
            indent=2,
            default=str
        )
    }


# =====================================================
# RCA NODE
# =====================================================

def rca_node(state):

    if state["intent"] != "RCA":

        return {
            "rca": "RCA was not required for this query."
        }

    history_text = format_last_turn(state.get("history", []))

    resolved_query = state.get(
        "resolved_query",
        state["query"]
    )

    prompt = f"""
You are an AWS Root Cause Analysis engine.

Analyze the AWS evidence supplied below.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUERY
=====================================================

{state["query"]}

=====================================================
RESOLVED QUERY
=====================================================

{resolved_query}

=====================================================
AWS EVIDENCE
=====================================================

{state.get("tool_result", "")}

Your job:

1. Identify abnormal behavior.
2. Correlate evidence across AWS services.
3. Determine the most likely root cause.
4. Do not invent missing information.
5. Clearly distinguish confirmed observations, supported inferences, and unknowns.
6. If metrics are empty or an instance is stopped, do NOT claim a cause, intentional stopping, agent involvement, monitoring absence, or user action.
7. Never call a cause High or Medium confidence unless the supplied evidence directly supports it.
8. If evidence is insufficient, explicitly write: "No root cause confirmed from the available evidence."
9. Do not infer that the AI agent, CloudWatch Agent, or any person stopped or changed an instance without explicit CloudTrail evidence.
10. Assign confidence:
   - High
   - Medium
   - Low

Return:

Problem:
<problem>

Evidence:
<important evidence>

Root Cause:
<root cause>

Confidence:
<High/Medium/Low>

Impact:
<impact>
"""

    response = llm.invoke(
        prompt
    )

    return {
        "rca": str(
            response.content
        )
    }


# =====================================================
# RECOMMENDATION NODE
# =====================================================

def recommendation_node(state):
    """Generate safe, structured RCA recommendations."""

    if state.get("intent") != "RCA":
        return {"recommendations": "Recommendations not required"}

    history_text = format_last_turn(state.get("history", []))
    resolved_query = state.get("resolved_query", state["query"])

    prompt = f"""
You are an AWS RCA remediation planner.

Use only the current query, previous turn, AWS evidence, and RCA.
Do not invent resource IDs, names, or parameter values.
Do not execute anything.
Only create executable actions when every required parameter is explicitly available.
The backend will validate actions and execute an approved batch sequentially.
The user gives one confirmation for the complete batch, not one confirmation per action.

Previous turn:
{history_text}

Current query:
{state["query"]}

Resolved query:
{resolved_query}

AWS evidence:
{state.get("tool_result", "")}

Root cause analysis:
{state.get("rca", "")}

Supported executable operations:
- action: create, delete, start, stop, or reboot
- EC2 start/stop/reboot use resource_type ec2_instance; RDS start/stop use rds_instance; Lambda enable/disable use lambda_function
- create/delete resource types: s3_bucket, ec2_instance, rds_instance, lambda_function,
  security_group, vpc, subnet, iam_resource

Return ONLY valid JSON in this format:
{{
  "issue_title": "Short issue title",
  "issue_description": "Confirmed issue description",
  "recommendations": [
    {{
      "priority": "high|medium|low",
      "action": "Recommended action",
      "reason": "Why it is recommended",
      "expected_result": "Expected result",
      "requires_approval": true
    }}
  ],
  "actions": [
    {{
      "action_id": "unique-id",
      "action": "create|delete|start|stop|reboot",
      "resource_type": "supported resource type",
      "parameters": {{}},
      "explanation": "Why this action is required",
      "status": "pending"
    }}
  ]
}}

Rules:
- Never recommend associate-vpc-cidr-block-association as a way to enable CloudWatch or CPU monitoring.
- EC2 CPUUtilization is AWS-provided; CloudWatch Agent is for OS-level metrics such as memory, processes, and disk space.
- If an instance is stopped, state that current metrics may be unavailable and do not recommend installing software until it is running.
- A terminated instance cannot be remediated as a running instance.
- Never claim the AI agent caused a termination without explicit CloudTrail evidence.
- Create executable actions only when supported and all required parameters are present.
- Supported executable actions include EC2 start/stop/reboot, RDS start/stop, Lambda enable/disable, and the supported create/delete operations. Never invent install-agent or monitoring actions.
- For EC2 start/stop/reboot, use live instance state: start only stopped instances, stop/reboot only running instances. Never create an action when the current state makes it invalid.
- If the current request is only a monitoring/status question, return empty recommendations and empty actions.
- Maximum 5 recommendations.
- Use an empty actions list when required parameters are missing.
- Never include speculative or destructive actions.
- Do not include Markdown or text outside the JSON object.
"""

    try:
        response = llm.invoke(prompt)
        raw = str(response.content).strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid recommendation structure")

        recommendations = parsed.get("recommendations", [])
        actions = parsed.get("actions", [])

        if not isinstance(recommendations, list):
            recommendations = []
        if not isinstance(actions, list):
            actions = []

        return {
            "recommendations": json.dumps(
                {
                    "issue_title": str(parsed.get("issue_title", "RCA Issue")),
                    "issue_description": str(parsed.get("issue_description", "")),
                    "recommendations": recommendations[:5],
                    "actions": actions,
                },
                ensure_ascii=False,
            )
        }

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Recommendation parsing error: {exc}")
        return {
            "recommendations": json.dumps(
                {
                    "issue_title": "RCA Recommendations",
                    "issue_description": "Structured remediation actions could not be generated safely.",
                    "recommendations": [],
                    "actions": [],
                },
                ensure_ascii=False,
            )
        }


# =====================================================
# OUTPUT SAFETY HELPERS
# =====================================================

def _remove_duplicate_recommendations(text: str) -> str:
    """Keep structured recommendations in the UI, not duplicated in the LLM answer."""
    if not text:
        return text
    import re
    cleaned = re.split(r"(?im)^\s*#+\s*recommendations\s*$", text, maxsplit=1)[0]
    return cleaned.rstrip()


def _apply_evidence_guardrails(text: str, state) -> str:
    """Prevent unsupported RCA certainty when AWS evidence is incomplete."""
    if state.get("intent") != "RCA":
        return text
    evidence = str(state.get("tool_result", "")).lower()
    if ("stopped" in evidence and ("metric" in evidence or "cloudwatch" in evidence)
            and ("empty" in evidence or "no data" in evidence or "null" in evidence)):
        unsafe = (
            "intentionally stopped", "ai agent", "cloudwatch agent is not installed",
            "cloudwatch agent was absent", "the user stopped", "routine checks", "no monitoring", "monitoring is not enabled"
        )
        if any(term in text.lower() for term in unsafe):
            return (
                "## Root Cause Analysis\n\n"
                "**Confirmed observations:** The supplied AWS data indicates that one or more "
                "instances are stopped and the available monitoring data is empty or insufficient.\n\n"
                "**Conclusion:** No root cause confirmed from the available evidence. Current CPU and "
                "other runtime metrics cannot be evaluated for a stopped instance. The evidence does not "
                "establish who or what stopped the instance, whether the stop was intentional, or whether "
                "an AI agent or monitoring agent caused a change.\n\n"
                "**Confidence:** Low / insufficient evidence\n\n"
                "**Impact:** Runtime performance cannot be assessed until the instance is running and "
                "relevant historical evidence is available."
            )
    return text


# =====================================================
# FINAL ANSWER NODE
# =====================================================

def final_answer_node(state):

    if state["intent"] == "BLOCKED":

        return {
            "answer": state["answer"]
        }

    history_text = format_last_turn(state.get("history", []))

    resolved_query = state.get(
        "resolved_query",
        state["query"]
    )

    # =================================================
    # KNOWLEDGE
    # =================================================

    if state["intent"] == "KNOWLEDGE":

        prompt = f"""
You are an AWS technical assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
KNOWLEDGE BASE
=====================================================

{state.get("context", "")}

Rules:

- Use only the resolved question.
- The resolved question already contains the required context when this is a follow-up.
- Do not answer an older question.
- Do not carry unrelated information from the previous turn.
- Do not invent information.
- Use only information supported by the Knowledge Base and the resolved question.
- Keep the answer concise unless the user explicitly asks for details.
"""

    # =================================================
    # MONITORING
    # =================================================

    elif state["intent"] == "MONITORING":

        prompt = f"""
You are an AWS monitoring assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
TOOLS EXECUTED
=====================================================

{state.get("tools")}

=====================================================
AWS RESULTS
=====================================================

{state.get("tool_result")}

Generate a monitoring report.

Use only the resolved question to identify the requested resource or result.

Rules:

- Answer the resolved question.
- Use the AWS results as the source of truth for account-specific information.
- Do not invent AWS resources or values.
- If the service is to be listed, provide it in tabular format.
- If get_s3_objects returned object records, display the object keys/names, sizes, last-modified values, and storage class when available. Do not replace an object listing with only a storage summary.
- If no objects are returned, clearly say the bucket contains no objects (or no matching objects for the requested prefix).

Format:

## Summary

## AWS Findings
"""

    # =================================================
    # ACTION
    # =================================================

    elif state["intent"] == "ACTION":
        return {
            "answer": state.get(
                "answer",
                "No AWS action was executed."
            )
        }

    # =================================================
    # RCA
    # =================================================

    else:

        prompt = f"""
You are an AWS monitoring and operations assistant.

=====================================================
CONVERSATION HISTORY
=====================================================

{history_text}

=====================================================
CURRENT USER QUESTION
=====================================================

{state["query"]}

=====================================================
RESOLVED QUESTION
=====================================================

{resolved_query}

=====================================================
TOOLS EXECUTED
=====================================================

{state.get("tools")}

=====================================================
AWS RESULTS
=====================================================

{state.get("tool_result")}

=====================================================
ROOT CAUSE ANALYSIS
=====================================================

{state.get("rca", "")}

Generate a professional response.

Answer only the resolved question. Do not introduce unrelated information from previous turns.

Format:

## Summary

## AWS Findings

## Root Cause Analysis

## Recommendations

## Impact

Rules:

1. Use only AWS Results for account-specific findings.
2. Clearly distinguish confirmed facts from likely causes.
3. If RCA confidence is low, say additional investigation is required.
4. Do not generate a separate Recommendations section; structured recommendations are rendered separately by the frontend.
5. Do not invent AWS resources or evidence.
6. State that no action is executed automatically; ask the user to review and explicitly confirm any proposed batch.
7. Do not claim execution success or failure unless execution results are present.
"""

    response = llm.invoke(
        prompt
    )

    answer_text = str(response.content).replace(
        "associate-vpc-cidr-block-association",
        "Do not use VPC CIDR association commands to enable CloudWatch monitoring"
    )
    answer_text = _remove_duplicate_recommendations(answer_text)
    answer_text = _apply_evidence_guardrails(answer_text, state)
    return {
        "answer": answer_text
    }


# =====================================================
# ROUTER
# =====================================================

def route_after_planner(state):

    if state["intent"] == "BLOCKED":

        return "final"

    if state["intent"] == "KNOWLEDGE":

        return "knowledge"

    if state["intent"] == "ACTION":

        return "action"

    if state["intent"] == "RCA":

        return "monitoring"

    return "monitoring"


def route_after_monitoring(state):

    if state["intent"] == "RCA":

        return "rca"

    return "final"


# =====================================================
# GRAPH
# =====================================================

builder = StateGraph(
    AgentState
)


builder.add_node(
    "planner",
    planner_node,
)


builder.add_node(
    "knowledge",
    knowledge_node,
)


builder.add_node(
    "action",
    action_node,
)


builder.add_node(
    "rca",
    rca_node,
)


builder.add_node(
    "recommendation",
    recommendation_node,
)


builder.add_node(
    "monitoring",
    monitoring_node,
)


builder.add_node(
    "final",
    final_answer_node,
)


builder.add_edge(
    START,
    "planner",
)


builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "knowledge": "knowledge",
        "action": "action",
        "monitoring": "monitoring",
        "final": "final"
    },
)


builder.add_edge(
    "knowledge",
    "final",
)


builder.add_edge(
    "action",
    "final",
)


builder.add_conditional_edges(
    "monitoring",
    route_after_monitoring,
    {
        "rca": "rca",
        "final": "final",
    }
)


builder.add_edge(
    "rca",
    "recommendation",
)


builder.add_edge(
    "recommendation",
    "final",
)


builder.add_edge(
    "final",
    END,
)


graph = builder.compile()


# =====================================================
# RUN AGENT
# =====================================================

def run_agent(
    session_id,
    query,
    history=None
):

    if history is None:
        history = []

    result = graph.invoke(
        {
            "session_id": session_id,
            "query": query,
            "history": history,
        }
    )

    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "service": result.get("service"),
        "tools": result.get(
            "tools",
            []
        ),
        "rca": result.get("rca"),
        "recommendations": result.get(
            "recommendations"
        ),
        "action_plan": result.get(
            "action_plan",
            {}
        ),
        "action_validation": result.get(
            "action_validation",
            ""
        )
    }
