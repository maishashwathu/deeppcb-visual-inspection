"""Execute with an existing Python kernel; skip installation when dependencies exist."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ['MPLCONFIGDIR'] = str(root / '.mplconfig')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter' / 'runtime')
nb_path = root / 'DeepPCB_Inspection.ipynb'
nb = nbformat.read(nb_path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
def started(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Running cell {cell_index}: {cell.source.splitlines()[0][:85]}', flush=True)
client = NotebookClient(nb, km=km, timeout=1800, allow_errors=False,
                        skip_cells_with_tag='installation', on_cell_start=started)
try:
    client.execute()
finally:
    nbformat.write(nb, nb_path)
errors = [o for c in nb.cells for o in c.get('outputs', []) if o.output_type == 'error']
assert not errors, errors
print('Notebook execution completed without errors.', flush=True)
