"""Prompt templates for EditReward scoring dimensions."""

INSTRUCTION_EDIT_FOLLOWING = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Instruction Following & Semantic Fidelity**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.
This dimension focuses on how accurately, completely, and exclusively the model executed the given text instruction.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Sub-Dimensions to Evaluate:**

- **Semantic Accuracy:**
  Assess whether the edited content accurately captures the semantics of the instruction. The edited result should precisely match the intended meaning. For example, if the instruction is "replace apples with oranges," the object must clearly be oranges, not other fruits.

- **Completeness of Editing:**
  Check whether **all parts** of the instruction are fully executed. For multi-step edits (e.g., "replace a red car with a blue bicycle"), both the color change and the object replacement must be done without omissions.

- **Exclusivity of Edit (No Over-Editing):**
  Ensure that only the requested parts are changed. The rest of the image (as seen in the source) should remain unaltered. For example, if the instruction only involves replacing an object, the background, lighting, and unrelated objects should not be unnecessarily modified.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly accurate, complete, and exclusive execution of the instruction.
- **3 (Relatively Good):** Largely correct, but with minor omissions or slight over-editing.
- **2 (Relatively Poor):** Major misinterpretation, incomplete edits, or noticeable unintended changes.
- **1 (Very Poor):** Instruction ignored or completely wrong execution.

Text instruction - {text_prompt}
"""

INSTRUCTION_EDIT_QUALITY = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Visual Quality & Realism**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.
This dimension focuses on how realistic, artifact-free, and aesthetically appealing the edited image is, while remaining consistent with the source image.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Sub-Dimensions to Evaluate:**

- **Plausibility & Physical Consistency:**
  Check whether the edit aligns with the laws of physics and the scene context. Lighting, shadows, reflections, perspective, size, and interactions with the environment should all appear natural compared to the source image.

- **Artifact-Free Quality:**
  Look for technical flaws such as blur, distortions, pixel misalignment, unnatural textures, or seams around edited regions. High-quality results should be free from such visible artifacts.

- **Aesthetic Quality:**
  Evaluate the overall harmony and visual appeal. The image should look natural, balanced, and pleasant. Colors, composition, and atmosphere should enhance the image rather than degrade it.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly realistic, artifact-free, seamless, and aesthetically pleasing.
- **3 (Relatively Good):** Mostly realistic and clean, with only minor flaws that do not significantly distract.
- **2 (Relatively Poor):** Noticeable physical inconsistencies or visible artifacts that make the edit unnatural.
- **1 (Very Poor):** Severe artifacts, incoherent composition, or visually unusable result.

Text instruction - {text_prompt}
"""

INSTRUCTION_EDIT_OVERALL = """
You are tasked with evaluating an edited image **in comparison with the original source image**, and assigning a score from 1 to 8, with 1 being the worst and 8 being the best.
This score should reflect **both how accurately the instruction was followed and the visual quality of the edited image**.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Sub-Dimensions to Evaluate:**

- **Instruction Following & Semantic Fidelity:**
  Assess whether the edited content accurately captures the semantics of the instruction, whether all parts of the instruction are fully executed, and whether only the requested parts are modified without unnecessary changes elsewhere.

- **Visual Quality & Realism:**
  Check whether the edit is realistic, artifact-free, and aesthetically pleasing. Consider physical plausibility, scene consistency, lighting, shadows, perspective, textures, and overall visual harmony.

**Scoring Criteria (1-8):**
- **8 (Very Good):** Perfect instruction following and flawless visual quality; edits are accurate, complete, exclusive, and visually seamless.
- **7 (Relatively Good):** Very good instruction following and high visual quality; minor, non-distracting flaws.
- **6 (Good):** Good instruction following or mostly good visual quality; some minor omissions or slight artifacts.
- **5 (Moderate):** Partially correct edits or moderate visual issues; noticeable flaws but understandable.
- **4 (Relatively Poor):** Significant misinterpretation or incomplete edits, or noticeable visual artifacts.
- **3 (Poor):** Major errors in instruction following and/or poor visual quality; hard to fully understand.
- **2 (Very Poor):** Very poor edits with large semantic errors and strong visual artifacts.
- **1 (Failed):** Completely wrong edits or visually unusable result.

Text instruction - {text_prompt}
"""

INSTRUCTION_EDIT_OVERALL_DETAILED = """
You are tasked with evaluating an edited image **in comparison with the original source image**, and assigning a score from 1 to 8, with 1 being the worst and 8 being the best.
This score should reflect **both how accurately the instruction was followed and the visual quality of the edited image**.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Dimension 1: Instruction Following & Semantic Fidelity**
Evaluate how well the edited image follows the given instruction. Consider the following sub-dimensions:
- **Semantic Accuracy:** Check if the edited content accurately captures the intended meaning of the instruction.
- **Completeness of Editing:** Verify that all aspects of the instruction are fully executed.
- **Exclusivity of Edit (No Over-Editing):** Ensure that only the requested changes are applied.

**Dimension 2: Visual Quality & Realism**
Evaluate the realism, technical quality, and aesthetic appeal of the edited image. Consider:
- **Plausibility & Physical Consistency**
- **Artifact-Free Quality**
- **Aesthetic Quality**

**Scoring Criteria (1-8):**
- **8 (Very Good):** Perfect instruction following and flawless visual quality.
- **7 (Relatively Good):** Very good with minor, non-distracting flaws.
- **6 (Good):** Good with minor omissions or slight artifacts.
- **5 (Moderate):** Partially correct or moderate visual issues.
- **4 (Relatively Poor):** Significant misinterpretation or noticeable artifacts.
- **3 (Poor):** Major errors and/or poor visual quality.
- **2 (Very Poor):** Large semantic errors and strong visual artifacts.
- **1 (Failed):** Completely wrong or visually unusable.

Text instruction - {text_prompt}
"""

# Prompt suffixes for special token / non-special token modes
PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

PROMPT_WITHOUT_SPECIAL_TOKEN = """
Please provide the overall ratings of this image:
"""
