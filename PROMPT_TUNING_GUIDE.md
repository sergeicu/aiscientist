# Prompt Tuning Guide

This guide explains how to customize and improve the classification prompts for better accuracy.

## Quick Start

The prompts are in `prompts/clinical_trial_classifier.yaml`. You can edit this file directly and test changes with:

```bash
python main.py test
```

## Prompt Structure

The YAML file has several sections:

### 1. System Prompt
```yaml
system_prompt: |
  You are an expert medical researcher specializing in clinical trial identification...
```

**Purpose**: Sets the role and overall instructions for the model.

**Tips**:
- Keep it focused and authoritative
- Emphasize JSON-only output
- Include key requirements

### 2. Task Description
```yaml
task_description: |
  Analyze the following article abstract and classify whether it describes a clinical trial.
```

**Purpose**: Explains the specific task.

**Tips**:
- Be specific about what constitutes a clinical trial
- List inclusion criteria
- List exclusion criteria
- Use bullet points for clarity

### 3. Output Format Instructions
```yaml
output_format_instructions: |
  Return your response as a JSON object with the following structure:
  {
    "is_clinical_trial": boolean,
    ...
  }
```

**Purpose**: Defines the exact JSON schema expected.

**Tips**:
- Show example JSON with comments
- Mark required vs optional fields
- Specify valid values for enums
- Remind: "Return ONLY JSON, no additional text"

### 4. Few-Shot Examples
```yaml
examples:
  - label: "Clinical Trial - Phase III Drug Study"
    abstract: |
      Background: Asthma is...
    expected_output:
      is_clinical_trial: true
      confidence: 0.95
```

**Purpose**: Show the model good examples of input→output.

**Tips**:
- Include diverse examples (different study types, phases)
- Include both positive and negative examples
- Show edge cases
- Keep examples concise but representative
- Usually 3-6 examples is optimal

### 5. Edge Case Guidance
```yaml
edge_case_guidance: |
  Challenging cases to consider:
  1. Pilot studies: Usually ARE clinical trials...
```

**Purpose**: Help model handle ambiguous cases.

## Iterative Tuning Process

### Step 1: Establish Baseline

Run initial test:
```bash
python main.py test > baseline_results.txt
```

Review results and identify errors.

### Step 2: Categorize Errors

Common error types:

**False Positives** (Non-trials classified as trials):
- Retrospective studies
- Meta-analyses
- Observational studies

**False Negatives** (Trials classified as non-trials):
- Pilot studies
- Small trials
- Non-drug interventions

**Field Extraction Errors**:
- Incorrect phase
- Missing sample size
- Wrong intervention type

### Step 3: Add Targeted Examples

For each error pattern, add 1-2 examples to the prompt:

```yaml
examples:
  # Add this if model misclassifies retrospective studies
  - label: "NOT Clinical Trial - Retrospective Study"
    abstract: |
      We conducted a retrospective chart review of 342 patients...
    expected_output:
      is_clinical_trial: false
      confidence: 0.98
      reasoning: "This is a retrospective chart review, not a prospective trial."
```

### Step 4: Adjust Instructions

Strengthen specific instructions:

```yaml
task_description: |
  ...

  IMPORTANT: Retrospective studies are NOT clinical trials.
  A clinical trial must:
  - Be prospective (not looking at past data)
  - Test an intervention
  - Have defined outcomes measured forward in time
```

### Step 5: Test and Compare

```bash
python main.py test > improved_results.txt
diff baseline_results.txt improved_results.txt
```

### Step 6: Validate on Larger Sample

Once test looks good, run on more articles:

```bash
python main.py process --batch-size 50 --no-resume
```

Review first 50 results for quality.

## Field-Specific Tuning

### Adding New Extraction Fields

1. **Update data model** (`src/models.py`):
```python
class ClassificationResult(BaseModel):
    ...
    # Add new field
    funding_source: Optional[str] = Field(
        default=None,
        description="Primary funding source (e.g., NIH, Industry)"
    )
```

2. **Update output format in prompt**:
```yaml
output_format_instructions: |
  {
    ...
    "funding_source": string  // optional - Primary funding source
  }
```

3. **Add examples showing the field**:
```yaml
examples:
  - label: "..."
    expected_output:
      ...
      funding_source: "NIH"
```

4. **Test**:
```bash
python main.py test
```

### Improving Field Accuracy

If a specific field (e.g., `trial_phase`) is often wrong:

1. **Add explicit instructions**:
```yaml
output_format_instructions: |
  ...

  Trial Phase Guidelines:
  - Phase I: First human testing, safety/dosage (usually <100 participants)
  - Phase II: Efficacy testing (usually 100-300 participants)
  - Phase III: Large confirmatory trials (usually >300 participants)
  - Phase IV: Post-market surveillance
```

2. **Add diverse examples**:
```yaml
examples:
  - label: "Phase I Safety Study"
    abstract: "First-in-human trial of XYZ drug in 24 healthy volunteers..."
    expected_output:
      trial_phase: "Phase I"

  - label: "Phase II Efficacy Study"
    abstract: "Multi-center study of 156 patients to evaluate efficacy..."
    expected_output:
      trial_phase: "Phase II"
```

## Model-Specific Tips

### Small Models (2B-3B parameters)

- Use **more examples** (5-8)
- Be **very explicit** in instructions
- Use **simpler vocabulary**
- Enable **both Outlines and json-repair**
- Lower temperature (0.05-0.1)

### Medium Models (7B-8B parameters)

- Use **3-5 examples**
- Balance specificity and brevity
- Standard temperature (0.1-0.2)

### Large Models (13B+ parameters)

- Use **2-3 examples**
- Can handle more nuance
- Can use chain-of-thought reasoning
- Higher temperature acceptable (0.1-0.3)

## Testing Strategies

### Test Specific Cases

Create a test CSV with known difficult cases:

```csv
pmid,title,abstract
TEST001,"Pilot Study","This pilot study of 20 patients..."
TEST002,"Retrospective Review","We reviewed charts from 2010-2020..."
TEST003,"Meta-Analysis","We pooled data from 42 trials..."
```

Then:
```bash
python main.py test --input-csv test_cases.csv
```

### Measure Accuracy

If you have ground truth labels:

```python
# In examples/measure_accuracy.py
from src import ArticleProcessor, Config

# Load results
true_labels = {...}  # Your ground truth
pred_labels = {...}  # Model predictions

# Calculate metrics
tp = sum(1 for pmid in true_labels if true_labels[pmid] and pred_labels[pmid])
fp = sum(1 for pmid in true_labels if not true_labels[pmid] and pred_labels[pmid])
fn = sum(1 for pmid in true_labels if true_labels[pmid] and not pred_labels[pmid])
tn = sum(1 for pmid in true_labels if not true_labels[pmid] and not pred_labels[pmid])

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

## Common Improvements

### Improve Confidence Calibration

If confidence scores don't match accuracy:

```yaml
output_format_instructions: |
  ...

  Confidence Guidelines:
  - 0.95-1.0: Extremely confident (clear RCT with all typical elements)
  - 0.80-0.94: Confident (clear trial but missing some details)
  - 0.60-0.79: Moderately confident (appears to be trial but some ambiguity)
  - 0.40-0.59: Uncertain (could go either way)
  - 0.00-0.39: Confident it's NOT a trial
```

### Improve Reasoning Quality

Add instruction:

```yaml
output_format_instructions: |
  ...

  Reasoning Format:
  1. First sentence: State your conclusion
  2. Second sentence: Provide key evidence
  3. Third sentence (optional): Note any caveats

  Example: "This is a randomized controlled trial. The abstract explicitly
  mentions randomization, blinding, and prospective design with 689 participants.
  All hallmarks of a Phase III trial are present."
```

### Handle Missing Information

```yaml
output_format_instructions: |
  ...

  For optional fields:
  - Use null if information is not mentioned in the abstract
  - Do NOT guess or infer beyond what's stated
  - "Unknown" values are only for when the abstract indicates uncertainty
```

## Version Control

Track prompt versions:

```bash
# Tag working versions
cp prompts/clinical_trial_classifier.yaml prompts/classifier_v1.0.yaml

# After improvements
cp prompts/clinical_trial_classifier.yaml prompts/classifier_v1.1.yaml

# Compare
diff prompts/classifier_v1.0.yaml prompts/classifier_v1.1.yaml
```

## Collaboration Workflow

When working together on prompts:

1. **Test current version**
   ```bash
   python main.py test > before.txt
   ```

2. **Make changes** to `prompts/clinical_trial_classifier.yaml`

3. **Test changes**
   ```bash
   python main.py test > after.txt
   ```

4. **Compare**
   ```bash
   diff before.txt after.txt
   # Or just visually compare
   ```

5. **Discuss**: Share observations about what improved/worsened

6. **Iterate**: Refine based on results

## Advanced Techniques

### Chain-of-Thought

For complex decisions, add reasoning step:

```yaml
system_prompt: |
  ...
  Before providing your final classification, think through:
  1. Is this prospective or retrospective?
  2. Is there an intervention being tested?
  3. Are there defined outcomes?
  4. Are there human participants?

  Then provide your JSON response.
```

### Dynamic Examples (Future)

Could implement example selection based on abstract similarity:
- Use embedding similarity to find most relevant examples
- Reduces context length
- Improves accuracy on edge cases

## Resources

- **Medical Trial Definitions**: [ClinicalTrials.gov Glossary](https://clinicaltrials.gov/ct2/about-studies/glossary)
- **Trial Phases**: [FDA Trial Phases](https://www.fda.gov/patients/clinical-trials-what-patients-need-know/what-are-clinical-trial-phases)
- **Study Designs**: [NIH Study Designs](https://www.nia.nih.gov/health/what-are-clinical-trials-and-studies)

## Getting Help

If you're stuck:
1. Review examples in this guide
2. Check if issue is model-specific (test different model)
3. Verify Ollama is working correctly
4. Check logs in `logs/` directory
5. Open an issue with example cases

Happy prompt tuning! 🎯
