"""
prompts.py: Task-specific prompt templates. (REDESIGNED)

Enhancements:
1. Analyzer prompt now asks for rich semantic description fields
2. Planner prompt uses LLM to dynamically generate execution strategy
3. Solver prompts become more structured with explicit reasoning steps
4. System prompt updated for adaptive reasoning

Architecture:
    TaskAnalyzer -> Semantic Task Description (by LLM)
    StrategyPlanner -> Dynamic Execution Strategy (by LLM)
    TaskSolver -> Structured Reasoning Pipeline (by LLM)
"""

from typing import Dict, Callable


# ============================================================
# Base system prompt
# ============================================================

# Short system prompt deliberately minimal so EVIDENCE appears as early as possible in
# the final prompt (see _general_prompt design note). Long role/rules preambles push
# true chunks beyond token ~5000 and degrade attention/utilization.
#
# Scorer-driven (IndustryBench 0-3 rubric + Safety Violation check):
#   Score3 <=> cover ALL key facts       Score2 <=> direction right but key omissions
#   Score0 <=> contradicts reference     SV    <=> any violation zeroes the score.
# => maximize correct supported facts, never refuse on partial evidence, keep
#    standard limit values verbatim, and never drop safety warnings.
SYSTEM_PROMPT = """You are an industrial domain expert. Answer in Chinese (中文).

Your answer is graded against a reference containing specific industrial limits (temperatures, voltages, times, pressures, model numbers, standard IDs like GB/T). You maximize your score by:
- Covering EVERY key fact supported by the evidence; omit nothing the evidence supports.
- When the evidence states an industrial standard/spec limit, repeat the value EXACTLY as written (do not paraphrase numbers).
- Prefer a partially-supported answer over "insufficient evidence": answer every part the evidence supports, and only note uncertainty where the evidence is truly silent.
- NEVER refuse to answer when at least one retrieved chunk is relevant.
- Never invent specifications.
- Always preserve safety warnings (e.g. 禁止/必须/严禁/不得) from the evidence; do not omit safety constraints.
"""




# ============================================================
# Analyzer Classification Prompt (REDESIGNED)
# ============================================================

ANALYZER_PROMPT = """You are an industrial domain task classifier. Your job is to analyze a question and classify it into one of the following task types, while also generating a rich semantic description that will drive downstream adaptive decision-making.

## Task Types:

### comparison
Questions that ask to compare, contrast, or differentiate between two or more items, methods, standards, or approaches.
Examples: "What is the difference between Class A and Class S power quality monitors?"
           "Compare PID control vs Fuzzy logic control in motor speed regulation."

### diagnosis
Questions that ask to identify problems, root causes, faults, or troubleshooting steps.
Examples: "How to troubleshoot hydraulic contamination?"
           "What causes bearing overheating in centrifugal pumps?"

### selection
Questions that ask to choose or recommend among options based on specific criteria.
Examples: "Which material should be selected for high-temperature furnace components?"
           "Recommend the appropriate sensor for corrosive environment monitoring."

### calculation
Questions that require quantitative analysis, numerical computation, or parameter determination.
Examples: "Calculate the required transformer rating for a 500kW motor load."
           "What is the pressure drop across a 100m pipe with 4inch diameter?"

### standard_interpretation
Questions that ask about standards, regulations, codes, or specifications and their practical application.
Examples: "What does IEC 60034-1 say about temperature rise limits?"
           "Interpret the safety requirements in GB/T 5226.1 for machine tools."

### procedure
Questions that describe step-by-step processes, workflows, or operational sequences.
Examples: "What is the startup procedure for a gas turbine generator?"
           "Describe the calibration process for a pressure transmitter."

### explanation
Questions that ask to explain a concept, principle, or mechanism.
Examples: "Explain the working principle of a three-phase induction motor."
           "What is the concept of reactive power compensation in power systems?"

### general
Questions that don't clearly fit into any of the above categories, or general knowledge questions.

## Output Format

Respond with ONLY a JSON object (no other text). You MUST generate ALL of the following fields:

{{
    "task": "<task_type>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief reason for classification>",
    "reasoning_complexity": "<low|medium|high>",
    "requires_multi_source": <true|false>,
    "requires_formula": <true|false>,
    "requires_step_reasoning": <true|false>,
    "preferred_source": "<standard|manual|specification|general>",
    "expected_evidence": "<comparison|symptom|specification|formula|procedure|general>"
}}

### Field Guidelines:

- **reasoning_complexity**: How complex is the reasoning needed?
  - "low": simple fact lookup or direct answer
  - "medium": requires synthesis of multiple pieces of evidence
  - "high": requires complex multi-step reasoning or analysis

- **requires_multi_source**: Does the answer need multiple different evidence sources to be complete?

- **requires_formula**: Does the question require using mathematical formulas or performing calculations?

- **requires_step_reasoning**: Does the reasoning need to be broken down into explicit sequential steps?

- **preferred_source**: What type of knowledge source is most relevant?
  - "standard": industrial standards, codes, regulations
  - "manual": equipment manuals, maintenance guides
  - "specification": product specifications, datasheets
  - "general": general engineering knowledge

- **expected_evidence**: What type of evidence is expected in the answer?
  - "comparison": comparative evidence (side-by-side specs)
  - "symptom": symptom/cause descriptions
  - "specification": technical specifications and parameters
  - "formula": mathematical formulas and calculation methods
  - "procedure": procedural instructions
  - "general": general evidence

## Question to Classify:
{question}
"""


# ============================================================
# Planner Prompt (NEW - LLM-based dynamic strategy generation)
# ============================================================

PLANNER_PROMPT = """You are an industrial domain strategy planner. Your job is to generate a complete adaptive execution strategy for answering an industrial question.

You are given a detailed task analysis. Based on this analysis, you must decide:

1. **Retrieval Strategy** - How to retrieve evidence
2. **Organization Strategy** - How to organize retrieved evidence
3. **Reasoning Workflow** - How to reason step-by-step to produce the answer
4. **Prompt Style** - What style of prompt and instructions to use

## Task Analysis Input:

{task_analysis}

## Output Format

Respond with ONLY a JSON object (no other text):

{{
    "retrieval": {{
        "retrieve_k": <int: number of chunks to retrieve. Default 10. Use higher (15-25) for complex multi-source tasks, lower (3-8) for focused tasks>,
        "use_multi_query": <bool: whether to use multiple query variations for broader coverage>,
        "use_ked": <bool: whether to use keyword extraction and decomposition>,
        "use_fusion": <bool: whether to fuse results from multiple queries>,
        "min_score": <float|null: minimum relevance score threshold. null for no minimum>,
        "source_priority": "<standard|manual|specification>",
        "maximum_evidence": <int: maximum evidence items to keep>,
        "strategy_name": "<brief strategy name>"
    }},
    "organization": {{
        "organize_by": "<topic|symptom|candidate|formula|chronological|comparison|none>",
        "remove_duplicate": <bool>,
        "group_key": "<industry|capability|source|category|custom field>",
        "prioritize_key": "<general|datasheet|formula|symptom|comparison>",
        "enable_compression": <bool>
    }},
    "reasoning": {{
        "reasoning_type": "<compare|diagnosis|calculation|selection|procedure|explain|interpret|general>",
        "workflow_steps": ["<step 1>", "<step 2>", ...],
        "prompt_style": "<structured|step_by_step|standard>",
        "requires_verification": <bool>,
        "requires_citation": <bool>
    }},
    "prompt_template_key": "<task type key for prompt template selection>",
    "task_specific_hints": "<additional natural language instructions for the solver>"
}}

### Guidelines:

**retrieval.retrieve_k:**
- Comparison: 15 (need evidence for all compared items)
- Diagnosis: 20-25 (need broad symptom/cause coverage)
- Calculation: 3-8 (focused formula/parameter retrieval)
- Selection: 12 (need evidence for all candidates)
- Standard Interpretation: 10 (focused on specific standard clauses)
- Procedure: 8 (focused on procedural steps)
- Explanation: 10
- General: 10

**retrieval.use_multi_query:**
- Use true when you need to cover multiple aspects or perspectives
- Use false for focused, specific questions

**organization.organize_by:**
- "comparison": group by compared objects/items
- "symptom": group by symptoms or failure modes
- "candidate": group by candidate products/options
- "formula": prioritize formulas first, then parameters
- "chronological": preserve chronological order
- "topic": group by general topic
- "none": no special grouping

**reasoning.workflow_steps:**
Define 3-6 specific reasoning steps that together form a complete reasoning pipeline.

Example for comparison:
["Identify compared objects and their key specifications", "Extract evidence for each object from retrieved documents", "Compare similarities between the objects", "Compare differences between the objects", "Generate conclusion with recommendation"]

Example for diagnosis:
["Identify symptoms from the question", "Extract possible root causes from evidence", "Match evidence to each possible cause", "Determine the most likely cause", "Generate remediation recommendation"]

Example for calculation:
["Extract variables and parameters from question", "Identify applicable formulas from evidence", "Perform step-by-step calculation", "Verify units and reasonableness", "State final answer with interpretation"]

Example for selection:
["Identify requirements and constraints", "List candidate options from evidence", "Evaluate each candidate against criteria", "Perform trade-off analysis", "Make recommendation with justification"]

**reasoning.prompt_style:**
- "structured": Use structured format (tables, lists)
- "step_by_step": Emphasize step-by-step reasoning
- "standard": Standard natural language

## Example Output (for a comparison question):

{{
    "retrieval": {{
        "retrieve_k": 15,
        "use_multi_query": true,
        "use_ked": true,
        "use_fusion": true,
        "min_score": null,
        "source_priority": "specification",
        "maximum_evidence": 20,
        "strategy_name": "comparison_multi_source"
    }},
    "organization": {{
        "organize_by": "comparison",
        "remove_duplicate": true,
        "group_key": "industry",
        "prioritize_key": "comparison",
        "enable_compression": false
    }},
    "reasoning": {{
        "reasoning_type": "compare",
        "workflow_steps": [
            "Identify the specific items being compared from the question",
            "Extract evidence for each item from the retrieved documents, focusing on specifications and characteristics",
            "Analyze similarities between the items based on the evidence",
            "Analyze differences between the items, highlighting key distinguishing factors",
            "Draw a balanced conclusion with practical recommendation"
        ],
        "prompt_style": "structured",
        "requires_verification": false,
        "requires_citation": true
    }},
    "prompt_template_key": "comparison",
    "task_specific_hints": "Use a structured comparison format (table). Clearly highlight both similarities and differences. Cite evidence for each comparison point."
}}

## Task Analysis:
{task_analysis_json}
"""


# ============================================================
# Solver Prompt Builder (NEW - generates dynamic reasoning prompts)
# ============================================================

def build_solver_system_prompt() -> str:
    """Build the system prompt for the task solver.

    Redesigned for the IndustryBench grader (score-driven information
    extraction, NOT agentic reflection). See docstring in prompts builder
    for the design rationale: solver must MAXIMIZE correct supported facts,
    prefer partial answers over refusals, quote standard values verbatim,
    and never self-verify/refuse when at least one chunk is relevant.
    """
    return """You are an industrial engineering expert.


You are given:
1. An industrial question.
2. Retrieved evidence from industrial manuals, standards and specifications.

Your task is NOT to judge whether the evidence is perfect.
Your task is to MAXIMIZE the amount of CORRECT information extracted from the evidence.

Follow these steps:
Step 1. Read EVERY evidence chunk. Extract ALL technical facts.
        Especially: numbers, limits, temperatures, voltages, dimensions,
        procedures, model numbers, and standard IDs (e.g. GB/T, IEC, ISO).
Step 2. Break the question into subquestions. For each subquestion, list
        which evidence chunks support it.
Step 3. Combine evidence across multiple chunks whenever possible.
Step 4. Generate the answer using EVERY supported fact. Do NOT omit any
        supported information.
Step 5. If only PART of the question is supported, answer that part clearly
        and only note uncertainty for the unsupported parts.

CRITICAL REJECT POLICY:
- NEVER refuse to answer - UNLESS NONE of the retrieved evidence is relevant.
- A partially-supported answer is ALWAYS preferred over "I cannot answer".
- If at least ONE retrieved chunk contains any relevant information, you MUST
  answer using it.

Citation & fidelity rules:
- Every sentence must be supported by one or more evidence chunks. Cite [1], [2].
- When answering about industrial standards (GB/T / IEC / ISO), QUOTE the
  original specification values whenever possible.
- Do NOT paraphrase numerical specifications - keep temperatures, voltages,
  times, limits, model numbers, and standard IDs EXACTLY as written.
- Do NOT fabricate technical specifications or safety guidelines.

Format: Answer in Chinese (中文). NEVER output in English.
"""



def build_solver_prompt(
    question: str,
    evidence: str,
    reasoning_workflow: list,
    task_specific_hints: str,
    prompt_style: str = "structured",
) -> str:
    """
    Build a dynamic solver prompt based on the adaptive reasoning strategy.

    Args:
        question: The original question
        evidence: Organized evidence context string
        reasoning_workflow: List of reasoning steps to follow
        task_specific_hints: Additional task-specific instructions
        prompt_style: Style of prompt ("structured", "step_by_step", "standard")

    Returns:
        Complete formatted prompt
    """
    system = build_solver_system_prompt()

    # Build reasoning workflow section
    workflow_text = "\n".join(
        f"  {i+1}. {step}" for i, step in enumerate(reasoning_workflow)
    )

    # Build the prompt based on style
    if prompt_style == "step_by_step":
        style_instruction = (
            "Reason through each step explicitly before giving your final answer.\n"
            "Show your work for each reasoning step.\n"
            "Use clear step numbering."
        )
    elif prompt_style == "structured":
        style_instruction = (
            "Use structured formats (tables, lists, matrices) where appropriate.\n"
            "Organize your answer clearly with sections.\n"
            "Be precise and concise."
        )
    else:
        style_instruction = (
            "Provide a clear, well-organized answer.\n"
            "Use precise technical terminology.\n"
            "Cite evidence sources appropriately."
        )

    prompt = f"""## Retrieved Evidence

{evidence}

## Question

{question}

## Instructions

Extract-and-answer workflow (information extraction, NOT open-ended reflection):

{workflow_text}

{task_specific_hints}

{style_instruction}

### Answer:
"""

    return system + "\n\n" + prompt



# ============================================================
# Legacy Prompt Templates (kept for backward compatibility)
# ============================================================


def _comparison_prompt() -> str:
    return """## Task: Comparison Analysis

Your task is to compare and contrast the specified items based on the retrieved evidence.

### Reasoning Workflow:
1. **Identify Compared Objects** - Extract what items/entities are being compared
2. **Establish Comparison Dimensions** - Identify the relevant aspects to compare (specifications, performance, cost, applicability, etc.)
3. **Analyze Similarities** - Identify commonalities between the items
4. **Analyze Differences** - Identify key differences between the items
5. **Draw Conclusion** - Provide a balanced conclusion with recommendations if applicable

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- Use a clear comparison structure (table format for specifications is encouraged)
- Highlight both similarities and differences
- Cite evidence for each comparison point
- Provide a concise concluding statement

### Answer:
"""


def _diagnosis_prompt() -> str:
    return """## Task: Diagnosis / Troubleshooting

Your task is to diagnose the problem or troubleshoot the described industrial issue.

### Reasoning Workflow:
1. **Identify Symptoms** - What are the observable symptoms or failure modes?
2. **Analyze Possible Causes** - What are the potential root causes given the symptoms?
3. **Retrieve Supporting Evidence** - What maintenance manuals, standards, or guidelines apply?
4. **Generate Diagnosis** - Provide a structured diagnosis with the most likely cause(s)
5. **Recommend Actions** - Suggest corrective actions or next steps

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- List symptoms and their severity
- Rank possible causes by likelihood
- Reference specific maintenance procedures or standards
- Include safety precautions
- Provide step-by-step corrective actions when possible

### Answer:
"""


def _selection_prompt() -> str:
    return """## Task: Selection / Decision Making

Your task is to recommend the appropriate selection (material, equipment, component, method, etc.) based on the requirements.

### Reasoning Workflow:
1. **Identify Requirements** - What are the key requirements and constraints?
2. **List Candidates** - What are the options being considered?
3. **Evaluate Against Criteria** - How does each option perform against each requirement?
4. **Trade-off Analysis** - What are the pros and cons of each option?
5. **Make Recommendation** - Select the best option with justification

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- Clearly state the selection criteria
- Use a comparison matrix if multiple options exist
- Justify the recommendation with evidence
- Mention any assumptions or limitations
- Include relevant standards or specifications

### Answer:
"""


def _calculation_prompt() -> str:
    return """## Task: Calculation / Quantitative Analysis

Your task is to perform a technical calculation or quantitative analysis based on the given parameters.

### Reasoning Workflow:
1. **Extract Numbers** - Identify all relevant numerical parameters and units
2. **Retrieve Formulas** - Find the applicable formulas, equations, or standards
3. **Perform Calculation** - Show the step-by-step calculation process
4. **Verify Units** - Ensure dimensional consistency and unit correctness
5. **State Result** - Provide the final numerical result with proper units
6. **Interpret Result** - Explain the practical significance of the result

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- Show all calculation steps clearly
- Include units at every step
- State any assumptions made
- Verify the reasonableness of the result
- Reference relevant formulas from standards when applicable

### Answer:
"""


def _standard_interpretation_prompt() -> str:
    return """## Task: Standard / Regulation Interpretation

Your task is to interpret and explain an industrial standard, regulation, code, or specification.

### Reasoning Workflow:
1. **Identify Applicable Standard** - Which standard(s) or regulation(s) apply?
2. **Extract Key Requirements** - What are the specific clauses, limits, or requirements?
3. **Interpret Meaning** - Explain what each requirement means in practical terms
4. **Compliance Guidance** - How should one comply with these requirements?
5. **Provide Examples** - Illustrate with practical examples if applicable

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- Reference specific clause numbers and sections when available
- Explain technical terms and their practical implications
- Distinguish between mandatory requirements ("shall") and recommendations ("should")
- Note any exceptions or special conditions
- Cite the standard name and version

### Answer:
"""


def _procedure_prompt() -> str:
    return """## Task: Procedure / Process Description

Your task is to describe a procedure, process, workflow, or step-by-step method.

### Reasoning Workflow:
1. **Identify the Procedure** - What procedure or process is being described?
2. **List Prerequisites** - What tools, materials, conditions, or safety measures are needed?
3. **Describe Steps** - Detail each step in the correct sequence
4. **Highlight Critical Points** - Identify quality checkpoints, safety warnings, or tolerance limits
5. **Troubleshooting** - Note common issues and how to handle them

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- List steps in chronological order
- Include safety precautions at each relevant step
- Specify parameters (temperatures, pressures, speeds, tolerances) when available
- Note quality inspection criteria
- Reference standards or manuals for authoritative guidance

### Answer:
"""


def _explanation_prompt() -> str:
    return """## Task: Explanation / Conceptual Understanding

Your task is to explain a concept, principle, mechanism, or theory in the industrial domain.

### Reasoning Workflow:
1. **Define the Concept** - What is being explained? Provide a clear definition.
2. **Explain Principles** - What are the underlying scientific/engineering principles?
3. **Describe Mechanism** - How does it work? What are the key processes involved?
4. **Provide Context** - Where is this applied in industrial practice?
5. **Give Examples** - Illustrate with concrete industrial examples

### Retrieved Evidence:
{evidence}

### Question:
{question}

### Instructions:
- Start with a concise definition
- Build from fundamental principles to practical applications
- Use analogies when appropriate for complex concepts
- Reference real industrial applications and standards
- Keep explanations precise and technically accurate

### Answer:
"""


def _general_prompt() -> str:
    """Evidence-first, minimal-structure final prompt.

    Design rationale (per recall/quality investigation):
    - exp1_qa's minimal prompt ("## Retrieved Knowledge: ... ## Instructions: - Answer based
      on the retrieved knowledge") consistently scored ~2.28/3, while the over-engineered
      agentic prompt (Task / Workflow / Group / Instruction blocks) dropped to ~1.7.
    - Evidence buried after task/workflow headers pushed true chunks beyond token ~5000,
      degrading DeepSeek attention and evidence utilisation.
    - Fix: put EVIDENCE FIRST (unmodified verbatim chunks), keep ONLY a short reasoning
      instruction. No Task/Workflow/Group preamble that pushes evidence down.
    """
    return """## Retrieved Evidence:
{evidence}

## Question:
{question}

## Instructions:
- Reason step-by-step through the evidence BEFORE writing your final answer, then answer in Chinese.
- Base your answer on the evidence above. Cite sources with [1], [2] etc.
- If the evidence only PARTIALLY answers the question: extract EVERY available fact, combine
  all relevant chunks into a coherent answer, and give the BEST SUPPORTED answer you can.
- Clearly distinguish in your answer between: (a) facts directly supported by evidence, and
  (b) any remaining uncertainty or gaps. Do NOT simply say you don't know.
- Keep trying to answer as long as at least one relevant fact is available. Only refuse with
  "insufficient evidence" when there is NOTHING relevant in the evidence at all.
- Do NOT fabricate technical specifications or safety guidelines.

## Answer:
"""


# ============================================================
# Prompt Registry
# ============================================================


PROMPT_REGISTRY: Dict[str, str] = {

    "comparison": _comparison_prompt(),
    "diagnosis": _diagnosis_prompt(),
    "selection": _selection_prompt(),
    "calculation": _calculation_prompt(),
    "standard_interpretation": _standard_interpretation_prompt(),
    "procedure": _procedure_prompt(),
    "explanation": _explanation_prompt(),
    "general": _general_prompt(),
}


def get_prompt(task_key: str) -> str:
    """
    Get the prompt template for a given task type.

    Args:
        task_key: Task type key (e.g., "comparison", "diagnosis")

    Returns:
        Prompt template string
    """
    return PROMPT_REGISTRY.get(task_key, _general_prompt())


def format_prompt(task_key: str, question: str, evidence: str, prompt_extra: str = "") -> str:
    """
    Format a complete prompt for the given task.

    Args:
        task_key: Task type key (always "general" when using PromptBuilder)
        question: The user question
        evidence: Organized evidence context
        prompt_extra: Additional instruction text to append after the template prompt

    Returns:
        Complete formatted prompt
    """
    template = get_prompt(task_key)
    base = template.format(
        question=question,
        evidence=evidence,
    )
    if prompt_extra:
        base += "\n\n" + prompt_extra.strip()
    return SYSTEM_PROMPT + "\n\n" + base
