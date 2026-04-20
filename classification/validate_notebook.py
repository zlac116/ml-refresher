"""Validate the executed notebook."""
import nbformat as nbf
import re

nb = nbf.read('/home/zlac116/Code/learning/ml-revision/classification/classification.ipynb', as_version=4)
total = len(nb.cells)
md_cells = sum(1 for c in nb.cells if c.cell_type == 'markdown')
code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')

empty_code = [(i, c.source[:80].replace("\n", " ")) for i, c in enumerate(nb.cells)
              if c.cell_type == 'code' and not c.outputs]
err_code = []
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        for o in c.outputs:
            if o.get('output_type') == 'error':
                err_code.append((i, o.get('ename'), o.get('evalue')))

print(f'Total cells: {total}, markdown: {md_cells}, code: {code_cells}')
print(f'Code cells with NO outputs: {len(empty_code)}')
for i, src in empty_code:
    print(f'  cell {i}: {src!r}')
print(f'Code cells with ERRORS: {len(err_code)}')
for i, ename, eval_ in err_code:
    print(f'  cell {i}: {ename}: {eval_}')

# Count exercises per section
sections = []
current = None
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown':
        m = re.match(r'^## (\d+)\.\s+(.*?)$', c.source.strip().split("\n")[0])
        if m:
            current = (int(m.group(1)), m.group(2))
            sections.append({'idx': current[0], 'title': current[1], 'exercises': 0, 'cell': i})
        # detect exercise headers like "**Exercise X.Y"
        if c.source.strip().startswith("**Exercise ") and sections:
            sections[-1]['exercises'] += 1

print("\nExercises per section:")
for s in sections:
    print(f"  {s['idx']:2d}. {s['title']:<55} -> {s['exercises']} exercises")
print(f"Total exercises: {sum(s['exercises'] for s in sections)}")
