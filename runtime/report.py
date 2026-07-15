"""Per-session report: PDF built from a run's logged CSVs — no re-simulation.

    python3 runtime/report.py run_20260715_101530
Output: reports/<run>_report.pdf  (+ plots in reports/<run>/)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    sys.exit('usage: python3 runtime/report.py <run_dir_name>')
run = sys.argv[1]
RUN = os.path.join(ROOT, 'logs', run)
if not os.path.isdir(RUN):
    sys.exit(f'no such run: {RUN}')

traj = np.genfromtxt(os.path.join(RUN, 'trajectory.csv'), delimiter=',',
                     names=True)
met = np.genfromtxt(os.path.join(RUN, 'metrics.csv'), delimiter=',',
                    names=True)
path = np.genfromtxt(os.path.join(RUN, 'path.csv'), delimiter=',', names=True)
meta = json.load(open(os.path.join(RUN, 'meta.json')))

out_dir = os.path.join(ROOT, 'reports', run)
os.makedirs(out_dir, exist_ok=True)

# ---- summary stats ----
settle = met['t'] > 0.5
stats = {
    'RMS cross-track [m]': float(np.sqrt(np.mean(met['e_ct'][settle] ** 2))),
    'Max |cross-track| [m]': float(np.max(np.abs(met['e_ct'][settle]))),
    'RMS heading err [rad]': float(np.sqrt(np.mean(met['e_psi'][settle] ** 2))),
    'Mean speed [m/s]': float(np.mean(met['v'][settle])),
    'Mean est. error [m]': float(np.mean(met['est_err'][settle])),
    'p99 solve [ms]': float(np.percentile(met['solve_ms'], 99)),
    'Duration [s]': float(traj['t'][-1]),
    'Goal reached': bool(meta.get('goal_reached', False)),
}

# ---- figures ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(path['x'], path['y'], 'k--', lw=1, alpha=.6, label='planned path')
ax.plot(traj['x'], traj['y'], color='tab:blue', lw=1.4, label='driven')
ax.set(aspect='equal', title='Trajectory'); ax.legend(fontsize=8)
fig.savefig(os.path.join(out_dir, 'trajectory.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(9, 6))
for a, (key, lab, col) in zip(axs.flat, [
        ('e_ct', 'cross-track [m]', 'tab:red'),
        ('v', 'speed [m/s]', 'tab:blue'),
        ('solve_ms', 'solve [ms]', 'tab:green'),
        ('est_err', 'estimation err [m]', 'tab:purple')]):
    a.plot(met['t'], met[key], color=col, lw=0.9)
    a.set_title(lab, fontsize=9); a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'timeseries.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

# ---- PDF ----
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle)
from PIL import Image as PILImage

styles = getSampleStyleSheet()
pdf_path = os.path.join(ROOT, 'reports', f'{run}_report.pdf')
doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=1.5 * cm)
S = [Paragraph(f'Session Report — {run}', styles['Title']),
     Paragraph(f"world={meta.get('world')} planner={meta.get('planner')} "
               f"controller={meta.get('controller')} "
               f"estimator={meta.get('estimator')} ai={meta.get('ai')}",
               styles['Heading3']),
     Spacer(1, 8)]

rows = [['Metric', 'Value']] + [[k, f'{v:.4f}' if isinstance(v, float)
                                 else str(v)] for k, v in stats.items()]
t = Table(rows, hAlign='LEFT')
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.grey)]))
S.append(t)
S.append(Spacer(1, 10))

for png in ('trajectory.png', 'timeseries.png'):
    p = os.path.join(out_dir, png)
    iw, ih = PILImage.open(p).size
    w = 15.5
    S.append(Image(p, width=w * cm, height=w * cm * ih / iw))
    S.append(Spacer(1, 8))

doc.build(S)
print(f'report: {pdf_path}')
for k, v in stats.items():
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
