import nbformat
nb = nbformat.read('/home/zlac116/Code/learning/ml-revision/time-series/time_series.ipynb', as_version=4)
n_total = len(nb.cells)
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_with_output = sum(1 for c in nb.cells if c.cell_type == 'code' and c.outputs)
n_no_output = sum(1 for c in nb.cells if c.cell_type == 'code' and not c.outputs)
n_errors = 0
err_cells = []
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        for o in c.outputs:
            if o.get('output_type') == 'error':
                n_errors += 1
                err_cells.append((i, o.get('ename'), o.get('evalue')))
print(f'Total cells: {n_total}')
print(f'Code cells:  {n_code}')
print(f'MD cells:    {n_md}')
print(f'Code w/ output: {n_with_output}')
print(f'Code w/o output: {n_no_output}')
print(f'Errors: {n_errors}')
for e in err_cells[:20]:
    print('  ERR cell', e[0], '->', e[1], ':', e[2][:300])
# also count empty (scaffold) code cells: those whose source is just "# Your answer here"
n_scaffold = sum(1 for c in nb.cells if c.cell_type == 'code' and c.source.strip() == '# Your answer here')
print(f'Scaffold cells (expected empty): {n_scaffold}')
print(f'Non-scaffold code cells without output: {n_no_output - n_scaffold}')
