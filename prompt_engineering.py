from transformers import pipeline
import csv
import os

# ============================================================
# PROMPT ENGINEERING - WEEK 6
# ============================================================

print("Loading model...")

# Load a pre-trained Hugging Face text generation model
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

# Avoid padding warnings
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

print("Model loaded successfully!\n")


# ============================================================
# FIVE REAL-WORLD PROBLEMS
# ============================================================

problems = {
    "Education": "How can a student improve their study habits?",
    
    "Healthcare": "What are some general ways to maintain a healthy lifestyle?",
    
    "Finance": "What are some simple ways to save money?",
    
    "Customer Support": "A customer received a damaged product. How should the support team respond?",
    
    "Programming": "How can a beginner find and fix an error in a Python program?"
}


# ============================================================
# FUNCTION TO GENERATE RESPONSE
# ============================================================

def generate_response(prompt):
    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )

    return result[0]["generated_text"].strip()


# ============================================================
# STORE ALL RESULTS
# ============================================================

results = []


# ============================================================
# RUN FIVE PROMPTING TECHNIQUES
# ============================================================

for domain, problem in problems.items():

    print("=" * 70)
    print("DOMAIN:", domain)
    print("PROBLEM:", problem)
    print("=" * 70)


    # --------------------------------------------------------
    # 1. ZERO-SHOT PROMPTING
    # --------------------------------------------------------

    zero_shot_prompt = f"""
Answer the following question clearly and briefly.

Question:
{problem}

Answer:
"""

    zero_response = generate_response(zero_shot_prompt)

    print("\nZERO-SHOT PROMPTING")
    print(zero_response)


    # --------------------------------------------------------
    # 2. ONE-SHOT PROMPTING
    # --------------------------------------------------------

    one_shot_prompt = f"""
Example:

Question:
How can I prepare for an exam?

Answer:
Create a study timetable, revise regularly, practice questions, and take short breaks.

Now answer this question:

Question:
{problem}

Answer:
"""

    one_response = generate_response(one_shot_prompt)

    print("\nONE-SHOT PROMPTING")
    print(one_response)


    # --------------------------------------------------------
    # 3. FEW-SHOT PROMPTING
    # --------------------------------------------------------

    few_shot_prompt = f"""
Example 1:

Question:
How can I prepare for an exam?

Answer:
Make a timetable and revise important topics regularly.

Example 2:

Question:
How can I improve my concentration?

Answer:
Remove distractions, take short breaks, and study in a quiet place.

Example 3:

Question:
How can I learn programming?

Answer:
Practice coding regularly and build small projects.

Now answer this question:

Question:
{problem}

Answer:
"""

    few_response = generate_response(few_shot_prompt)

    print("\nFEW-SHOT PROMPTING")
    print(few_response)


    # --------------------------------------------------------
    # 4. ROLE PROMPTING
    # --------------------------------------------------------

    role_prompt = f"""
You are an experienced professional.

Give a helpful, practical, simple, and easy-to-understand answer
to the following question.

Question:
{problem}

Answer:
"""

    role_response = generate_response(role_prompt)

    print("\nROLE PROMPTING")
    print(role_response)


    # --------------------------------------------------------
    # 5. CHAIN-OF-THOUGHT STYLE PROMPT
    # --------------------------------------------------------

    cot_prompt = f"""
Solve the following problem carefully.

Give a short explanation of the important steps and then provide
the final answer.

Question:
{problem}

Answer:
"""

    cot_response = generate_response(cot_prompt)

    print("\nCHAIN-OF-THOUGHT PROMPTING")
    print(cot_response)


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append([
        domain,
        problem,
        zero_response,
        one_response,
        few_response,
        role_response,
        cot_response
    ])


# ============================================================
# CREATE DATASET FOLDER
# ============================================================

os.makedirs("dataset", exist_ok=True)


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

csv_file = "dataset/prompt_results.csv"

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Domain",
        "Problem",
        "Zero-Shot",
        "One-Shot",
        "Few-Shot",
        "Role Prompting",
        "Chain-of-Thought"
    ])

    writer.writerows(results)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ALL PROMPT ENGINEERING EXPERIMENTS COMPLETED!")
print("=" * 70)

print("Results saved to:")
print(csv_file)