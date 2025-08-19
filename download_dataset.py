from datasets import load_dataset

# Load the dataset
print("🔄 Loading dataset...")
ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default")["train"]

# Define mapping from harm category name to S-tag
S_TAGS = {
    "Terrorism": "S2",
    "Religious Extremism": "S4",
    "Criminal Guidance": "S7",
    "Drugs": "S8",
}

target_categories = set(S_TAGS.keys())

def is_severe_with_target_category(example):
    # Check severity level
    if not (example["response_0_severity_level"] == 3 or example["response_1_severity_level"] == 3):
        return False

    # Check response_0 harm category
    for cat in target_categories:
        if example["response_0_harm_category"].get(cat, False):
            return True

    # Check response_1 harm category
    for cat in target_categories:
        if example["response_1_harm_category"].get(cat, False):
            return True

    return False

# Filter dataset
print("🔍 Filtering for severity level 3 and target categories...")
filtered_ds = ds.filter(is_severe_with_target_category)

# Print stats
print(f"\n📊 Found {len(filtered_ds)} prompts with severity level 3 and categories {target_categories}\n")

# Print examples
for i in range(min(5, len(filtered_ds))):
    ex = filtered_ds[i]
    categories_r0 = [k for k, v in ex["response_0_harm_category"].items() if v and k in target_categories]
    categories_r1 = [k for k, v in ex["response_1_harm_category"].items() if v and k in target_categories]
    sev_r0 = ex["response_0_severity_level"]
    sev_r1 = ex["response_1_severity_level"]

    print(f"[{i+1}] Prompt: {ex['prompt']}")
    print(f"    - response_0_severity_level: {sev_r0} | categories: {categories_r0}")
    print(f"    - response_1_severity_level: {sev_r1} | categories: {categories_r1}\n")

# Save prompts
with open("filtered_severe_prompts.txt", "w") as f:
    for row in filtered_ds:
        f.write(row["prompt"].strip() + "\n")


import json

with open("filtered_severe_prompts.jsonl", "w") as f:
    for row in filtered_ds:
        f.write(json.dumps({"prompt": row["prompt"]}) + "\n")




print("✅ Done! Prompts saved to: filtered_severe_prompts.txt and filtered_severe_prompts.jsonl")