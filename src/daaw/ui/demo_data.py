"""Pre-loaded demo data for the DAAW UI — works without any API keys.

Based on a real execution: Multi-Location Goa Trip Planner pipeline
compiled and executed via Gemini 2.5 Flash.
"""

from __future__ import annotations

from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec, WorkflowSpec
from daaw.schemas.results import AgentResult, TaskResult


# ---------------------------------------------------------------------------
# Demo Scenario: Multi-Location Goa Trip Planner
#
# DAG:
#   task_001 (Research Locations)
#       \--> task_002 (Propose Itinerary)
#               |-- task_003 (Accommodation Costs)
#               |-- task_004 (Transport & Activity Costs)
#               \-- task_005 (Food & Misc Costs)
#                      \-- (all 3) --> task_006 (Compile Trip Plan)
# ---------------------------------------------------------------------------

DEMO_WORKFLOW_SPEC = WorkflowSpec(
    id="7e72c38b-7013-4b81-95f8-b0b89fe24567",
    name="Multi-Location Goa Trip Planner",
    description=(
        "Plans a multi-location Goa trip including timeline, specific locations, "
        "accommodation options, and estimated costs for various categories."
    ),
    tasks=[
        TaskSpec(
            id="task_001",
            name="Research Goa Locations and Travel Times",
            description="Identify popular tourist areas in North and South Goa, their main attractions, and approximate travel times between them to inform itinerary planning.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[],
            success_criteria="A detailed list of at least 4 distinct popular Goa locations, their key characteristics, and estimated inter-location travel times.",
            timeout_seconds=300,
            max_retries=2,
        ),
        TaskSpec(
            id="task_002",
            name="Propose Multi-Location Itinerary Structure",
            description="Based on the researched locations, propose a logical multi-location itinerary for a typical duration (e.g., 5-7 days), specifying which locations to visit and for how long.",
            agent=AgentSpec(role="generic_llm", tools_allowed=[]),
            dependencies=[DependencySpec(task_id="task_001")],
            success_criteria="A proposed itinerary outline including 2-3 distinct locations, suggested duration for each, and a logical flow.",
            timeout_seconds=120,
            max_retries=2,
        ),
        TaskSpec(
            id="task_003",
            name="Research Accommodation Options and Costs",
            description="For each location identified in the proposed itinerary, research typical accommodation types (e.g., budget, mid-range, luxury hotels/guesthouses) and their estimated daily costs.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[DependencySpec(task_id="task_002")],
            success_criteria="A list of accommodation options and average daily costs (low, mid, high range) for each location in the proposed itinerary.",
            timeout_seconds=300,
            max_retries=2,
        ),
        TaskSpec(
            id="task_004",
            name="Estimate Transportation and Activity Costs",
            description="Research estimated costs for local transportation within Goa (e.g., scooter rental, taxis, buses) and common activities/excursions relevant to the proposed itinerary.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[DependencySpec(task_id="task_002")],
            success_criteria="Estimated daily costs for local transportation and a range of costs for typical tourist activities in Goa.",
            timeout_seconds=300,
            max_retries=2,
        ),
        TaskSpec(
            id="task_005",
            name="Estimate Food and Miscellaneous Costs",
            description="Research estimated daily costs for food (eating out at various types of establishments from shacks to mid-range restaurants) and suggest a buffer for miscellaneous expenses in Goa.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["web_search"]),
            dependencies=[DependencySpec(task_id="task_002")],
            success_criteria="Estimated daily costs for food (budget, mid-range, high) and a suggested miscellaneous buffer.",
            timeout_seconds=300,
            max_retries=2,
        ),
        TaskSpec(
            id="task_006",
            name="Compile Comprehensive Goa Trip Plan",
            description="Consolidate all gathered information into a detailed Goa trip plan, including a day-by-day timeline, specific locations, accommodation suggestions with costs, transportation costs, activity costs, food costs, and a total estimated budget.",
            agent=AgentSpec(role="generic_llm", tools_allowed=["file_write"]),
            dependencies=[
                DependencySpec(task_id="task_002"),
                DependencySpec(task_id="task_003"),
                DependencySpec(task_id="task_004"),
                DependencySpec(task_id="task_005"),
            ],
            success_criteria="A well-structured, comprehensive Goa trip plan document covering timeline, locations, accommodation, transportation, activities, food, and total estimated costs, written to a file.",
            timeout_seconds=300,
            max_retries=2,
        ),
    ],
    metadata={
        "compiled_by": "DAAW Compiler v1.0",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    },
)


# ---------------------------------------------------------------------------
# Demo Results -- from real Gemini 2.5 Flash execution
# ---------------------------------------------------------------------------

DEMO_RESULTS: dict[str, TaskResult] = {
    "task_001": TaskResult(
        task_id="task_001",
        agent_result=AgentResult(
            output=(
                "## Popular Tourist Areas in Goa\n\n"
                "### NORTH GOA\n\n"
                "#### 1. Calangute / Baga Beach Area\n"
                "- **Type:** High-energy beach hub\n"
                "- **Key Attractions:** Calangute Beach (\"Queen of Beaches\"), Baga Beach (water sports, nightlife), Tito's Lane\n"
                "- **Vibe:** Loud, touristy, commercial\n\n"
                "#### 2. Anjuna / Vagator\n"
                "- **Type:** Bohemian cultural hub\n"
                "- **Key Attractions:** Anjuna Flea Market, Chapora Fort, Vagator Beach, trance parties\n"
                "- **Vibe:** Artistic, alternative, backpacker-friendly\n\n"
                "### CENTRAL GOA\n\n"
                "#### 3. Panjim (Panaji)\n"
                "- **Type:** Capital city, heritage quarter\n"
                "- **Key Attractions:** Fontainhas Latin Quarter, Miramar Beach, Mandovi River cruises\n"
                "- **Vibe:** Cultured, walkable, Portuguese heritage\n\n"
                "### SOUTH GOA\n\n"
                "#### 4. Palolem / Agonda\n"
                "- **Type:** Serene beach paradise\n"
                "- **Key Attractions:** Palolem Beach (crescent bay), Agonda Beach (nesting turtles), Butterfly Beach\n"
                "- **Vibe:** Peaceful, scenic, nature-focused\n\n"
                "#### 5. Colva / Benaulim\n"
                "- **Type:** Quiet residential beach stretch\n"
                "- **Key Attractions:** Colva Beach, Benaulim Church, rice paddies\n"
                "- **Vibe:** Relaxed, local, less commercial\n\n"
                "#### 6. Old Goa\n"
                "- **Type:** UNESCO World Heritage site\n"
                "- **Key Attractions:** Basilica of Bom Jesus, Se Cathedral, Church of St. Cajetan\n"
                "- **Vibe:** Historical, spiritual, architectural\n\n"
                "## Inter-Location Travel Times\n\n"
                "| Route | Distance | Drive Time |\n"
                "|---|---|---|\n"
                "| Calangute <-> Anjuna | 10 km | 25-35 min |\n"
                "| Calangute <-> Panjim | 15 km | 30-45 min |\n"
                "| Anjuna <-> Panjim | 20 km | 35-50 min |\n"
                "| Panjim <-> Old Goa | 10 km | 20-30 min |\n"
                "| Panjim <-> Palolem | 70 km | 1.5-2 hrs |\n"
                "| Calangute <-> Palolem | 75 km | 1.5-2 hrs |\n"
                "| Anjuna <-> Palolem | 80 km | 2-2.5 hrs |\n"
                "| Palolem <-> Colva | 35 km | 45-60 min |\n"
                "| Colva <-> Panjim | 35 km | 50-70 min |\n"
                "| Old Goa <-> Palolem | 75 km | 1.5-2 hrs |\n\n"
                "> Note: Add 30-50% during peak season (Dec-Jan) due to traffic."
            ),
            status="success",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "popular tourist areas Goa North South attractions travel times"},
                        "result": "Found 6 distinct locations across North, Central, and South Goa with travel time data.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=86.56,
    ),
    "task_002": TaskResult(
        task_id="task_002",
        agent_result=AgentResult(
            output=(
                "# Goa Multi-Location Itinerary -- 6 Days / 5 Nights\n\n"
                "## Structure: 3 Bases, North -> Central -> South\n\n"
                "| Segment | Base Location | Duration | Vibe |\n"
                "|---|---|---|---|\n"
                "| **Leg 1** | Anjuna / Vagator | 2 nights | Bohemian, beaches, culture |\n"
                "| **Leg 2** | Panjim (Panaji) | 1 night | Heritage, city, history |\n"
                "| **Leg 3** | Palolem / Agonda | 2 nights | Peaceful, scenic, nature |\n\n"
                "## Day-by-Day Breakdown\n\n"
                "### Day 1 -- Arrive & Settle in Anjuna\n"
                "- Arrive Goa (Dabolim/Mopa airport), transfer to Anjuna (~45-60 min)\n"
                "- Check in, unwind at Vagator Beach\n"
                "- Evening: Explore Anjuna cafe scene\n\n"
                "### Day 2 -- North Goa Exploration\n"
                "- Morning: Chapora Fort sunrise, Vagator Beach\n"
                "- Afternoon: Anjuna Flea Market (Wed) or Mapusa Market\n"
                "- Evening: Beach shack sunset dinner at Thalassa or similar\n\n"
                "### Day 3 -- Transfer to Panjim\n"
                "- Morning: Checkout, ride to Panjim (35 min)\n"
                "- Afternoon: Walk Fontainhas Latin Quarter\n"
                "- Evening: Mandovi River sunset cruise\n\n"
                "### Day 4 -- Old Goa + Transfer South\n"
                "- Morning: Old Goa churches (Basilica of Bom Jesus, Se Cathedral)\n"
                "- Afternoon: Drive to Palolem (1.5-2 hrs)\n"
                "- Evening: Palolem Beach sunset, settle in\n\n"
                "### Day 5 -- South Goa Beach Day\n"
                "- Morning: Kayak to Butterfly Beach from Palolem\n"
                "- Afternoon: Agonda Beach (sea turtles season permitting)\n"
                "- Evening: Beach bonfire or silent disco at Palolem\n\n"
                "### Day 6 -- Departure\n"
                "- Morning: Final beach walk, breakfast\n"
                "- Transfer to airport (~1.5-2 hrs from Palolem)\n\n"
                "## Why This Flow Works\n"
                "- North -> Central -> South avoids backtracking\n"
                "- Starts high-energy (Anjuna), transitions through culture (Panjim), ends relaxed (Palolem)\n"
                "- Total inter-city transit: only 3 transfers over 6 days"
            ),
            status="success",
            metadata={"model": "gemini-2.5-flash"},
        ),
        attempt=1,
        elapsed_seconds=43.2,
    ),
    "task_003": TaskResult(
        task_id="task_003",
        agent_result=AgentResult(
            output=(
                "# Accommodation Research -- Goa 6-Day Itinerary\n\n"
                "All prices in INR per room/night, peak season (Nov-Feb).\n\n"
                "## Base 1: Anjuna / Vagator (2 nights)\n\n"
                "| Tier | Type | Price Range (INR/night) |\n"
                "|---|---|---|\n"
                "| Budget | Guesthouse / Hostel dorm | 800-1,500 |\n"
                "| Mid-Range | Boutique hotel / Beach villa | 2,500-5,000 |\n"
                "| Luxury | Resort / Pool villa | 8,000-15,000+ |\n\n"
                "## Base 2: Panjim (1 night)\n\n"
                "| Tier | Type | Price Range (INR/night) |\n"
                "|---|---|---|\n"
                "| Budget | Heritage hostel / Guesthouse | 1,000-2,000 |\n"
                "| Mid-Range | Boutique heritage hotel | 3,000-6,000 |\n"
                "| Luxury | 5-star / Heritage luxury | 10,000-20,000+ |\n\n"
                "## Base 3: Palolem / Agonda (2 nights)\n\n"
                "| Tier | Type | Price Range (INR/night) |\n"
                "|---|---|---|\n"
                "| Budget | Beach hut (basic) | 600-1,200 |\n"
                "| Mid-Range | AC beach hut / Cottage | 2,000-4,500 |\n"
                "| Luxury | Boutique resort / Pool cottage | 7,000-12,000+ |\n\n"
                "## 5-Night Trip Total Estimate\n\n"
                "| Tier | Total (5 nights) |\n"
                "|---|---|\n"
                "| Budget | INR 3,800-7,700 |\n"
                "| Mid-Range | INR 12,500-24,500 |\n"
                "| Luxury | INR 38,000-62,000+ |"
            ),
            status="success",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "Goa hotel accommodation prices Anjuna Panjim Palolem budget mid-range luxury 2025"},
                        "result": "Bot detection blocked. Used training knowledge for well-established Goa accommodation pricing.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=63.98,
    ),
    "task_004": TaskResult(
        task_id="task_004",
        agent_result=AgentResult(
            output=(
                "# Goa Local Transportation & Activity Costs\n\n"
                "## Transportation -- Daily Cost Estimates\n\n"
                "### Self-Drive (Recommended)\n\n"
                "| Mode | Cost | Notes |\n"
                "|---|---|---|\n"
                "| Scooter (Activa/Honda) | INR 300-450/day | Most popular; widely available |\n"
                "| Motorcycle (Royal Enfield) | INR 700-1,200/day | For longer South Goa rides |\n"
                "| Car (self-drive) | INR 1,500-3,000/day | Hatchback to sedan |\n\n"
                "### Hired Transport\n\n"
                "| Mode | Cost | Notes |\n"
                "|---|---|---|\n"
                "| Auto-rickshaw | INR 50-150/trip | Negotiate; no meters |\n"
                "| Taxi (local) | INR 200-500/trip | Airport transfer: INR 800-1,500 |\n"
                "| Pilot (bike taxi) | INR 100-300/trip | Common in tourist areas |\n"
                "| Bus (local Kadamba) | INR 15-50/trip | Cheapest; Panjim hub |\n\n"
                "### Inter-City Transfers\n\n"
                "| Route | Cost |\n"
                "|---|---|\n"
                "| Airport -> Anjuna | INR 800-1,500 |\n"
                "| Anjuna -> Panjim | INR 400-700 |\n"
                "| Panjim -> Palolem | INR 2,000-3,000 |\n"
                "| Palolem -> Airport | INR 2,000-3,500 |\n\n"
                "## Activity Costs\n\n"
                "| Activity | Cost/Person | Location |\n"
                "|---|---|---|\n"
                "| Parasailing | INR 800-1,500 | Calangute/Baga |\n"
                "| Jet Ski (15 min) | INR 500-1,000 | Baga Beach |\n"
                "| Kayaking | INR 500-800/hr | Palolem |\n"
                "| Scuba Intro Dive | INR 3,500-5,000 | Grande Island |\n"
                "| Dolphin Trip | INR 400-700 | Palolem/Sinquerim |\n"
                "| Spice Plantation Tour | INR 400-600 | Ponda |\n"
                "| Mandovi Cruise | INR 200-500 | Panjim |\n"
                "| Old Goa Churches | Free | Old Goa |"
            ),
            status="success",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "Goa scooter rental cost taxi prices activities excursion costs 2025"},
                        "result": "Bot detection blocked. Used training knowledge for Goa transport and activity pricing.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=136.05,
    ),
    "task_005": TaskResult(
        task_id="task_005",
        agent_result=AgentResult(
            output=(
                "# Goa Daily Food Costs & Miscellaneous Buffer\n\n"
                "Based on Numbeo Dec 2025 data + Goa market knowledge.\n\n"
                "## Food Cost Tiers -- Per Person Per Day\n\n"
                "### Budget Traveler (Beach Shacks + Street Food)\n"
                "| Meal | Venue | Est. Cost |\n"
                "|---|---|---|\n"
                "| Breakfast | Tea + banana pancake at local stall | INR 80-120 |\n"
                "| Lunch | Fish curry rice thali at local shack | INR 150-220 |\n"
                "| Dinner | Beach shack meal (prawn fry + rice + drink) | INR 250-400 |\n"
                "| Snacks/chai | Throughout the day | INR 50-100 |\n"
                "| **Daily Total** | | **INR 530-840** |\n\n"
                "### Mid-Range Traveler\n"
                "| Meal | Venue | Est. Cost |\n"
                "|---|---|---|\n"
                "| Breakfast | Cafe brunch (eggs, smoothie) | INR 200-350 |\n"
                "| Lunch | Restaurant (seafood, pasta) | INR 350-550 |\n"
                "| Dinner | Mid-range restaurant + drinks | INR 600-1,000 |\n"
                "| Snacks | Cafe snack + juice | INR 100-200 |\n"
                "| **Daily Total** | | **INR 1,250-2,100** |\n\n"
                "### Comfortable/High Traveler\n"
                "| Meal | Venue | Est. Cost |\n"
                "|---|---|---|\n"
                "| Breakfast | Hotel or premium cafe | INR 400-700 |\n"
                "| Lunch | Upscale restaurant | INR 700-1,200 |\n"
                "| Dinner | Fine dining + cocktails | INR 1,500-3,000 |\n"
                "| Snacks | Premium cafe/bar | INR 200-400 |\n"
                "| **Daily Total** | | **INR 2,800-5,300** |\n\n"
                "## Miscellaneous Buffer\n\n"
                "| Category | Daily Estimate |\n"
                "|---|---|\n"
                "| SIM card / data | INR 30-50 |\n"
                "| Sunscreen / toiletries | INR 50-100 |\n"
                "| Tips | INR 100-200 |\n"
                "| Emergency / shopping | INR 200-500 |\n"
                "| **Misc Total** | **INR 700-1,000/day** |"
            ),
            status="success",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "args": {"query": "Goa food prices daily cost eating out 2025 Numbeo"},
                        "result": "Retrieved Numbeo cost-of-living data for Goa restaurants and food.",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=80.86,
    ),
    "task_006": TaskResult(
        task_id="task_006",
        agent_result=AgentResult(
            output=(
                "The Goa trip plan has been written to `goa-trip-plan.md` in your workspace.\n\n"
                "**Document Structure:**\n\n"
                "1. **Trip Overview** -- 6 days/5 nights, 3-zone structure, season info\n"
                "2. **Day-by-Day Itinerary** -- All 6 days with specific locations, timings, and per-activity costs inline\n"
                "3. **Accommodation** -- Budget/Mid-Range/Luxury tiers for all 3 bases (Anjuna, Panjim, Palolem) with per-night and total costs\n"
                "4. **Transportation** -- Inter-city transfer costs, daily scooter/taxi costs, 6-day totals per person\n"
                "5. **Activities** -- Free must-dos + paid optional experiences with per-person costs\n"
                "6. **Food** -- 3-tier daily breakdowns with benchmark prices from Numbeo data\n"
                "7. **Total Budget Summary** -- Per-person totals for solo and couple across all tiers\n\n"
                "**Budget Range (6 days, per person):**\n"
                "- Budget: INR 15,000-22,000\n"
                "- Mid-Range: INR 35,000-55,000\n"
                "- Luxury: INR 80,000-1,20,000+"
            ),
            status="success",
            metadata={
                "model": "gemini-2.5-flash",
                "tool_calls": [
                    {
                        "tool": "file_write",
                        "args": {"path": "goa-trip-plan.md", "content": "(comprehensive trip plan document)"},
                        "result": "Wrote 8,432 characters to goa-trip-plan.md",
                    },
                ],
            },
        ),
        attempt=1,
        elapsed_seconds=118.97,
    ),
}


# ---------------------------------------------------------------------------
# Demo Critic Verdicts -- from real execution
# ---------------------------------------------------------------------------

DEMO_CRITIC_VERDICTS: list[dict] = [
    {
        "task_id": "task_001",
        "task_name": "Research Goa Locations and Travel Times",
        "verdict": "PASS",
        "reasoning": (
            "The output identifies 6 distinct popular Goa locations (well above the required 4), "
            "covering both North and South Goa. Each includes key characteristics, attractions, and vibe. "
            "Inter-location travel times are provided in a clear table covering 10 route pairs."
        ),
        "patch": None,
    },
    {
        "task_id": "task_002",
        "task_name": "Propose Multi-Location Itinerary Structure",
        "verdict": "PASS",
        "reasoning": (
            "Proposes 3 distinct locations (Anjuna/Vagator, Panjim, Palolem/Agonda) with clear durations "
            "(2 nights, 1 night, 2 nights). Fits the 5-7 day window at 6 days/5 nights. "
            "Logical north-to-south geographic flow with explicit rationale."
        ),
        "patch": None,
    },
    {
        "task_id": "task_003",
        "task_name": "Research Accommodation Options and Costs",
        "verdict": "PASS",
        "reasoning": (
            "Provides accommodation across three tiers (budget, mid-range, luxury) with INR costs "
            "for all three itinerary locations. Includes cumulative trip cost summary and seasonal notes. "
            "Web search was unavailable but training knowledge is acceptable for well-established market data."
        ),
        "patch": None,
    },
    {
        "task_id": "task_004",
        "task_name": "Estimate Transportation and Activity Costs",
        "verdict": "PASS",
        "reasoning": (
            "Provides daily transport costs (scooter, taxi, bus) broken down by budget tier, "
            "plus comprehensive activity costs (water sports, diving, cruises, cultural sites). "
            "Data is well-structured and tied to the proposed itinerary locations."
        ),
        "patch": None,
    },
    {
        "task_id": "task_005",
        "task_name": "Estimate Food and Miscellaneous Costs",
        "verdict": "PASS",
        "reasoning": (
            "Provides daily food costs across three tiers with itemized meal breakdowns and benchmark prices "
            "from Numbeo data. Miscellaneous buffer is well-categorized at INR 700-1,000/day. "
            "Summary table ties food and misc together across traveler types."
        ),
        "patch": None,
    },
    {
        "task_id": "task_006",
        "task_name": "Compile Comprehensive Goa Trip Plan",
        "verdict": "PASS",
        "reasoning": (
            "Comprehensive plan written to goa-trip-plan.md covering all required elements: "
            "day-by-day timeline, specific locations, accommodation with costs (3 tiers x 3 bases), "
            "transportation, activities, food, and total budget summary. Well-structured document."
        ),
        "patch": None,
    },
]


# ---------------------------------------------------------------------------
# Demo Compilation Log
# ---------------------------------------------------------------------------

DEMO_COMPILATION_LOG: list[str] = [
    "[0.00s] Compiler initialized -- provider: gemini, model: gemini-2.5-flash",
    "[0.01s] Building system prompt with 6 agent roles, 5 tools",
    "[0.02s] Sending goal to LLM for workflow generation...",
    "[2.41s] LLM response received (3,204 tokens)",
    "[2.42s] Parsing JSON response...",
    "[2.43s] JSON parsed successfully -- 6 tasks found",
    "[2.44s] Validating WorkflowSpec with Pydantic...",
    "[2.45s] Validation passed -- all task IDs unique, dependencies valid",
    "[2.46s] Building DAG from dependency graph...",
    "[2.47s] DAG validated -- no cycles detected (Kahn's algorithm)",
    "[2.48s] Topological order: task_001 -> task_002 -> task_003, task_004, task_005 (parallel) -> task_006",
    "[2.49s] Compilation complete -- WorkflowSpec ready",
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
        "active": "gemini",
        "model": "gemini-2.5-flash",
    },
    "tools": {
        "registered": 5,
        "names": ["web_search", "file_read", "file_write", "shell_command", "notify"],
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
