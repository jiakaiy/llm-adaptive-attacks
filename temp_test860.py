import json
from main import main as run_one, get_default_args

prompt = None
with open('filtered_moderate_prompts_indexed.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except: 
            continue
        if obj.get('index') == 860:
            prompt = obj.get('prompt')
            break

if not prompt: 
    raise SystemExit('index 860 not found')

args = get_default_args()
args.goal = prompt
args.prompt_template = 'catnonsense'
args.target_model = 'deepseek-reasoner'
args.target_max_n_tokens = 350
args.judge_model = 'no-judge'
args.debug = True
args.n_iterations = 1
args.n_restarts = 1
args.n_chars_change_max = 0
args.n_tokens_change_max = 0
args.target_str = ''

noadv, orig, final = run_one(args)

rec = {
    'custom_id': 860,
    'index': 860,
    'prompt': prompt,
    'template': 'catnonsense',
    'target_model': args.target_model,
    'noadv': noadv,
    'orig': orig,
    'final': final
}

print(json.dumps(rec, ensure_ascii=False)) 