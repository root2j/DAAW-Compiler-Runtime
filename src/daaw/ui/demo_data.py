"""Pre-loaded demo data for the DAAW UI — works without any API keys."""

from __future__ import annotations

from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec, WorkflowSpec
from daaw.schemas.results import AgentResult, TaskResult


# ---------------------------------------------------------------------------
# Demo Scenario: E-commerce Order Processing Pipeline
#
# Diamond DAG:
#   task_001 (Receive Order)
#       |-- task_002 (Validate Payment)
#       \-- task_003 (Check Inventory)
#              |-- (both) --> task_004 (Process Fulfillment)
#                                 |-- task_005 (Send Confirmation)
#                                 \-- task_006 (Update Analytics)
# ---------------------------------------------------------------------------

DEMO_WORKFLOW_SPEC = WorkflowSpec(
    id="demo-ecommerce-001",
    name="E-commerce Order Processing Pipeline",
    description=(
        "End-to-end order processing workflow: receive an order, validate "
        "payment and check inventory in parallel, then fulfill, confirm, "
        "and update analytics."
    ),
    tasks=[
        TaskSpec(
            id="task_001",
            name="Receive Order",
            description="Collect order details from the customer including items, quantities, shipping address, and payment method.",
            agent=AgentSpec(role="user_proxy", tools_allowed=["web_search"]),
            dependencies=[],
            success_criteria="Order details captured with at least: items list, shipping address, payment method.",
            timeout_seconds=120,
            max_retries=1,
        ),
        TaskSpec(
            id="task_002",
            name="Validate Payment",
            description="Verify the customer's payment method, check for sufficient funds, and authorize the transaction.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[DependencySpec(task_id="task_001")],
            success_criteria="Payment authorization confirmed with transaction ID.",
            timeout_seconds=60,
            max_retries=2,
        ),
        TaskSpec(
            id="task_003",
            name="Check Inventory",
            description="Verify that all ordered items are in stock and reserve them for this order.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["file_read"]),
            dependencies=[DependencySpec(task_id="task_001")],
            success_criteria="All items confirmed in stock with reservation IDs.",
            timeout_seconds=60,
            max_retries=2,
        ),
        TaskSpec(
            id="task_004",
            name="Process Fulfillment",
            description="Create shipping label, assign to warehouse, and initiate the pick-pack-ship process.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["file_write"]),
            dependencies=[
                DependencySpec(task_id="task_002"),
                DependencySpec(task_id="task_003"),
            ],
            success_criteria="Shipping label created and warehouse assignment confirmed.",
            timeout_seconds=180,
            max_retries=2,
        ),
        TaskSpec(
            id="task_005",
            name="Send Confirmation",
            description="Send order confirmation email to the customer with order summary and tracking number.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[DependencySpec(task_id="task_004")],
            success_criteria="Confirmation email sent successfully with tracking number.",
            timeout_seconds=30,
            max_retries=2,
        ),
        TaskSpec(
            id="task_006",
            name="Update Analytics",
            description="Record order metrics: revenue, items sold, fulfillment time, and customer segment data.",
            agent=AgentSpec(role="breakdown", tools_allowed=["file_write"]),
            dependencies=[DependencySpec(task_id="task_004")],
            success_criteria="Analytics dashboard updated with order metrics.",
            timeout_seconds=60,
            max_retries=1,
        ),
    ],
    metadata={
        "compiled_by": "DAAW Compiler v1.0",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
)


# ---------------------------------------------------------------------------
# Demo Results -- realistic outputs with tool call metadata
# ---------------------------------------------------------------------------

DEMO_RESULTS: dict[str, TaskResult] = {
    "task_001": TaskResult(
        task_id="task_001",
        agent_result=AgentResult(
            output={
                "order_id": "ORD-2025-8842",
                "customer": "Acme Corp",
                "items": [
                    {"sku": "WIDGET-A", "qty": 50, "unit_price": 12.99},
                    {"sku": "GADGET-B", "qty": 20, "unit_price": 34.50},
                ],
                "shipping_address": "123 Innovation Blvd, San Francisco, CA 94105",
                "payment_method": "corporate_card_ending_4291",
            },
            status="success",
            metadata={"source": "user_proxy_questionnaire"},
        ),
        attempt=1,
        elapsed_seconds=2.3,
    ),
    "task_002": TaskResult(
        task_id="task_002",
        agent_result=AgentResult(
            output={
                "authorized": True,
                "transaction_id": "TXN-9901-ABCD",
                "amount": 1339.50,
                "currency": "USD",
                "fraud_score": 0.02,
            },
            status="success",
            metadata={
                "model": "llama-3.3-70b-versatile",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "payment gateway authorize corporate_card_ending_4291"},
                        "result": "Authorization successful. Transaction ID: TXN-9901-ABCD. Amount: $1,339.50 USD.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=1.8,
    ),
    "task_003": TaskResult(
        task_id="task_003",
        agent_result=AgentResult(
            output={
                "all_in_stock": True,
                "reservations": [
                    {"sku": "WIDGET-A", "qty": 50, "reservation_id": "RSV-001"},
                    {"sku": "GADGET-B", "qty": 20, "reservation_id": "RSV-002"},
                ],
                "warehouse": "WH-WEST-01",
            },
            status="success",
            metadata={
                "model": "llama-3.3-70b-versatile",
                "tool_calls": [
                    {
                        "tool": "file_read",
                        "args": {"path": "inventory/current_stock.json"},
                        "result": '{"WIDGET-A": 120, "GADGET-B": 45, "GIZMO-C": 0}',
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=1.5,
    ),
    "task_004": TaskResult(
        task_id="task_004",
        agent_result=AgentResult(
            output={
                "shipping_label": "SHP-2025-XK42",
                "carrier": "FedEx",
                "tracking_number": "FX-7789012345",
                "warehouse_assigned": "WH-WEST-01",
                "estimated_ship_date": "2025-06-15",
            },
            status="success",
            metadata={
                "model": "llama-3.3-70b-versatile",
                "tool_calls": [
                    {
                        "tool": "file_write",
                        "args": {"path": "labels/SHP-2025-XK42.json", "content": '{"carrier":"FedEx","tracking":"FX-7789012345"}'},
                        "result": "Wrote 52 characters to labels/SHP-2025-XK42.json",
                    },
                    {
                        "tool": "file_write",
                        "args": {"path": "fulfillment/ORD-2025-8842.json", "content": '{"status":"processing","warehouse":"WH-WEST-01"}'},
                        "result": "Wrote 49 characters to fulfillment/ORD-2025-8842.json",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=3.1,
    ),
    "task_005": TaskResult(
        task_id="task_005",
        agent_result=AgentResult(
            output={
                "email_sent": True,
                "recipient": "orders@acmecorp.com",
                "tracking_number": "FX-7789012345",
                "confirmation_id": "CONF-5567",
            },
            status="success",
            metadata={
                "model": "llama-3.3-70b-versatile",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "FedEx tracking FX-7789012345 status"},
                        "result": "Package FX-7789012345: Label created. Estimated delivery: June 18, 2025.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=0.9,
    ),
    "task_006": TaskResult(
        task_id="task_006",
        agent_result=AgentResult(
            output={
                "metrics_recorded": False,
                "error": "Database connection timeout after 3 retries",
                "partial_data": {"revenue": 1339.50, "items_sold": 70},
            },
            status="failure",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "file_write",
                        "args": {"path": "analytics/daily_metrics.json", "content": '{"date":"2025-06-15","revenue":1339.50,"orders":1}'},
                        "result": "[ERROR] Connection timeout: analytics DB unreachable at 10.0.1.42:5432",
                    },
                ],
            },
        ),
        attempt=2,
        elapsed_seconds=8.4,
    ),
}


# ---------------------------------------------------------------------------
# Demo Critic Verdicts
# ---------------------------------------------------------------------------

DEMO_CRITIC_VERDICTS: list[dict] = [
    {
        "task_id": "task_001",
        "task_name": "Receive Order",
        "verdict": "PASS",
        "reasoning": "Order details include items list, shipping address, and payment method -- all required fields present.",
        "patch": None,
    },
    {
        "task_id": "task_002",
        "task_name": "Validate Payment",
        "verdict": "PASS",
        "reasoning": "Payment authorized with transaction ID TXN-9901-ABCD. Fraud score 0.02 is well below threshold.",
        "patch": None,
    },
    {
        "task_id": "task_003",
        "task_name": "Check Inventory",
        "verdict": "PASS",
        "reasoning": "All items confirmed in stock with valid reservation IDs. Warehouse assignment included.",
        "patch": None,
    },
    {
        "task_id": "task_004",
        "task_name": "Process Fulfillment",
        "verdict": "PASS",
        "reasoning": "Shipping label created (SHP-2025-XK42), carrier assigned (FedEx), warehouse confirmed. Two file_write tool calls completed successfully.",
        "patch": None,
    },
    {
        "task_id": "task_005",
        "task_name": "Send Confirmation",
        "verdict": "PASS",
        "reasoning": "Confirmation email sent to orders@acmecorp.com with tracking number FX-7789012345.",
        "patch": None,
    },
    {
        "task_id": "task_006",
        "task_name": "Update Analytics",
        "verdict": "FAIL",
        "reasoning": "metrics_recorded is False — database connection timed out. Partial data captured but analytics dashboard was not updated. Success criteria not met.",
        "patch": "retry task_006 with fallback: write metrics to local file analytics/daily_metrics_fallback.json instead of database",
    },
]


# ---------------------------------------------------------------------------
# Demo Compilation Log
# ---------------------------------------------------------------------------

DEMO_COMPILATION_LOG: list[str] = [
    "[0.00s] Compiler initialized -- provider: groq, model: llama-3.3-70b-versatile",
    "[0.01s] Building system prompt with 6 agent roles, 4 tools",
    "[0.02s] Sending goal to LLM for workflow generation...",
    "[1.24s] LLM response received (2,847 tokens)",
    "[1.25s] Parsing JSON response...",
    "[1.26s] JSON parsed successfully -- 6 tasks found",
    "[1.27s] Validating WorkflowSpec with Pydantic...",
    "[1.28s] Validation passed -- all task IDs unique, dependencies valid",
    "[1.29s] Building DAG from dependency graph...",
    "[1.30s] DAG validated -- no cycles detected (Kahn's algorithm)",
    "[1.31s] Topological order: task_001 -> task_002, task_003 -> task_004 -> task_005, task_006",
    "[1.32s] Compilation complete -- WorkflowSpec ready",
]


# ---------------------------------------------------------------------------
# Demo System Stats
# ---------------------------------------------------------------------------

DEMO_SYSTEM_STATS: dict = {
    "agents": {
        "registered": 6,
        "roles": ["planner", "pm", "breakdown", "critic", "user_proxy", "generic_llm"],
    },
    "providers": {
        "available": ["groq", "gemini", "openai", "anthropic"],
        "active": "groq",
        "model": "llama-3.3-70b-versatile",
    },
    "tools": {
        "registered": 4,
        "names": ["web_search", "file_read", "file_write", "shell_command"],
    },
    "schemas": {
        "count": 11,
        "names": [
            "WorkflowSpec", "TaskSpec", "AgentSpec", "DependencySpec",
            "AgentResult", "TaskResult", "TaskStatus", "AgentRole",
            "PatchAction", "PatchOperation", "WorkflowPatch",
        ],
    },
}
