"""Validate the executed notebook."""
import nbformat as nbf
import re

nb = nbf.read('/home/zlac116/Code/learning/ml-revision/regression/regression.ipynb', as_version=4)
n_cells = len(nb.cells)
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f'total cells: {n_cells}')
print(f'  markdown : {n_md}')
print(f'  code     : {n_code}')

# count exercises per major section
section = None
counts = {}
order = []
for c in nb.cells:
    if c.cell_type != 'markdown':
        continue
    src = c.source
    m = re.match(r'^## (\d+)\.\s*(.+)', src.split('\n')[0])
    if m:
        section = f'{m.group(1)}. {m.group(2)[:50]}'
        if section not in counts:
            order.append(section)
        counts.setdefault(section, 0)
        continue
    if section and re.match(r'\*\*Exercise \d+\.\d+\*\*', src):
        counts[section] += 1

print()
for s in order:
    n = counts[s]
    flag = '' if n >= 3 or 'Setup' in s or 'Problem framing' in s or 'Caveats' in s else '   <-- BELOW 3'
    print(f'  {s:55s} -> {n} exercises{flag}')

# check non-scaffold code cells have outputs
n_with_output = 0
n_empty_output = 0
empty_examples = []
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    src_stripped = c.source.strip()
    if not src_stripped:
        continue
    if '# Your answer here' in src_stripped:
        continue
    if c.get('outputs'):
        n_with_output += 1
    else:
        n_empty_output += 1
        empty_examples.append(src_stripped[:80])

print()
print(f'non-scaffold code cells with output: {n_with_output}')
print(f'non-scaffold code cells without output: {n_empty_output}')
for e in empty_examples[:5]:
    print('  EMPTY:', e)

# now extract numeric results: train/val/test sizes, baselines, final test
print('\n=== EXECUTION RESULTS (parsed from outputs) ===')
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    src = c.source
    if 'train:' in src and 'val' in src and 'test' in src and 'features:' in src:
        for o in c.outputs:
            if 'text' in o:
                print(o['text'])
            elif o.get('output_type') == 'stream':
                print(o.get('text', ''))
        break

# print baseline table and final table
print('\n=== Baseline + Final tables ===')
for c in nb.cells:
    if c.cell_type != 'code':
        continue
    src = c.source
    if 'baseline_df' in src and 'historical_mean' in src:
        for o in c.outputs:
            if o.get('output_type') == 'execute_result':
                print('-- baselines (val):')
                print(o.get('data', {}).get('text/plain', ''))
    if 'final_table' in src and 'tuned_LGBM' in src:
        for o in c.outputs:
            if o.get('output_type') == 'execute_result':
                print('-- final (test):')
                print(o.get('data', {}).get('text/plain', ''))
