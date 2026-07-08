"""Generate the full research report PDF from benchmark CSVs and plots."""
import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, 'benchmarks')
P = os.path.join(B, 'plots')

styles = getSampleStyleSheet()
H1 = styles['Heading1']
H2 = styles['Heading2']
N = ParagraphStyle('body', parent=styles['Normal'], fontSize=9.5, leading=13)
CAP = ParagraphStyle('cap', parent=styles['Normal'], fontSize=8,
                     textColor=colors.grey, spaceBefore=2, spaceAfter=10)


def df_table(csv, fmt='{:.4f}'):
    df = pd.read_csv(os.path.join(B, csv))
    data = [list(df.columns)] + [
        [v if isinstance(v, str) else fmt.format(v) for v in row]
        for row in df.itertuples(index=False)]
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#eef2f5')]),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    return t


def fig(png, w=16.5):
    from PIL import Image as PILImage
    p = os.path.join(P, png)
    iw, ih = PILImage.open(p).size
    return Image(p, width=w * cm, height=w * cm * ih / iw)


doc = SimpleDocTemplate(os.path.join(ROOT, 'paper_report.pdf'), pagesize=A4,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm)
S = []

S.append(Paragraph('Learning-Based Adaptive Navigation Controller for '
                   'Unmanned Ground Vehicles', styles['Title']))
S.append(Paragraph('A Benchmark Study of Estimation, Planning, Control, and '
                   'Learned Adaptation — Open-Source Research Layer',
                   styles['Heading3']))
S.append(Spacer(1, 8))

S.append(Paragraph('Abstract', H2))
S.append(Paragraph(
    'We present a fully reproducible, open-source research pipeline for UGV '
    'navigation that benchmarks three state estimators (EKF, UKF, sliding-'
    'window factor graph), five global planners (A*, Theta*, Hybrid A*, RRT*, '
    'MPPI), and five path-tracking controllers (PID, Pure Pursuit, Stanley, '
    'NMPC, and a learning-adapted NMPC) on a common kinematic bicycle model '
    'with simulated noisy GPS/IMU/encoder sensing. A learned speed-scheduling '
    'policy, trained in 88 s of CPU time with the Cross-Entropy Method, '
    'reduces RMS cross-track error by 12% and steering-rate variance by 40% '
    'relative to the fixed-parameter NMPC, confirming the adaptation '
    'hypothesis that motivates a full PPO/SAC meta-policy. A closed-loop '
    'robustness study with the EKF in the control loop reveals that all '
    'high-precision controllers converge to a common error floor of about '
    '0.065 m imposed by state-estimation accuracy, indicating that beyond '
    'Stanley-level precision, localization — not control — is the binding '
    'constraint. All code, seeds, and results are released.', N))
S.append(Spacer(1, 6))

S.append(Paragraph('1. Method Overview', H2))
S.append(Paragraph(
    'Pipeline: sensor simulation (GPS 5 Hz sigma = 0.15 m; gyro with bias '
    'random walk; encoder noise) feeds the state-estimation layer; planners '
    'operate on 6 x 4 m occupancy grids (0.1 m resolution); controllers track '
    'a figure-eight reference (lemniscate, 4 x 2 m) on a kinematic bicycle '
    '(L = 0.32 m) with rate-level inputs [a, ddelta] shared by ALL '
    'controllers for fair smoothness comparison; the learning layer adapts '
    'NMPC parameters online. The NMPC solves a 30-step, 50 ms multiple-'
    'shooting OCP (CasADi/ipopt) with Q = diag(10,10,5,1,0.1), '
    'R = diag(0.5,2.0), delta-u penalty, and warm starting — the identical '
    'OCP later compiled with ACADOS SQP-RTI for ROS2 deployment.', N))
S.append(Spacer(1, 6))

S.append(Paragraph('2. State Estimation Benchmark', H2))
S.append(df_table('estimation_results.csv'))
S.append(Paragraph('Table 1: 5 seeds, 40 s runs. UKF matches EKF once the '
                   'sigma-point mean uses a circular mean for heading; the '
                   'smoother trails in this GPS-rich scenario.', CAP))
S.append(fig('estimation_comparison.png'))
S.append(Paragraph('Figure 1: position error over time and RMS summary with '
                   'seed variance.', CAP))
S.append(PageBreak())

S.append(Paragraph('3. Planner Benchmark', H2))
S.append(df_table('planner_results.csv', '{:.2f}'))
S.append(Paragraph('Table 2: 3 maps x 3 seeds, 100% success for all '
                   'planners. Hybrid A* is smoothest and kinematically '
                   'feasible at 18 ms.', CAP))
S.append(fig('planner_comparison.png'))
S.append(Paragraph('Figure 2: planned paths on map 1; planning time (log) '
                   'and length/smoothness bars.', CAP))
S.append(Spacer(1, 6))

S.append(Paragraph('4. Controller Benchmark with Learned Adaptation', H2))
S.append(df_table('controller_results_full.csv'))
S.append(Paragraph('Table 3: full 40 s figure-eight. Adaptive-MPC: CEM-'
                   'learned curvature speed schedule v_ref = v_base / '
                   '(1 + k_curv |kappa|), converged to v_base = 1.89, '
                   'k_curv = 1.35.', CAP))
S.append(fig('full_controller_dashboard.png'))
S.append(Paragraph('Figure 3: trajectories, cross-track error, velocity '
                   '(with learned schedule), and metric bars.', CAP))
S.append(PageBreak())

S.append(Paragraph('5. Robustness: Estimation in the Control Loop', H2))
S.append(df_table('robustness_results.csv'))
S.append(Paragraph('Table 4: true tracking error when each controller acts '
                   'on the EKF estimate (3 noise seeds) vs ground-truth '
                   'state.', CAP))
S.append(fig('robustness_comparison.png'))
S.append(Paragraph('Figure 4: precision controllers collapse to a common '
                   '~0.065 m error floor set by estimation accuracy; Pure '
                   'Pursuit improves because its lookahead low-pass filters '
                   'estimation noise.', CAP))
S.append(Spacer(1, 6))

S.append(Paragraph('6. Discussion', H2))
S.append(Paragraph(
    'Three findings. (i) Adaptation works: even a 2-parameter learned '
    'schedule beats hand-tuned NMPC on error and steering smoothness, '
    'trading average speed — the trade-off a full PPO/SAC meta-policy will '
    'manage contextually. (ii) The estimation floor: below ~0.07 m of '
    'estimation error, controller precision is masked; investment should go '
    'to localization (or estimator-aware control, motivating solver/'
    'estimator-health signals in the RL observation). (iii) Benchmark '
    'integrity is fragile: we found and fixed three silent bugs — reference '
    'lobe-jumping at the path self-intersection, linear angle averaging in '
    'the UKF sigma-point mean, and closed-path index clamping that stopped '
    'the robot after one lap while INFLATING its error score. Each would '
    'have corrupted the conclusions without visible failure.', N))
S.append(Spacer(1, 6))

S.append(Paragraph('7. Limitations and Next Steps', H2))
S.append(Paragraph(
    'Kinematic model without tire dynamics; sensor simulation in place of '
    'Gazebo/Isaac physics; CEM policy has 2 parameters vs the specified '
    '58-dim PPO/SAC meta-policy; ipopt (10-14 ms) vs the ACADOS RTI target '
    '(<5 ms). Next: Gazebo sensor plugins replace simulation/sensors.py, '
    'PPO/SAC training (rl/ppo_training.py, environment smoke-tested) on GPU, '
    'ACADOS port with the <2 cm trajectory-match validation gate, and the '
    '2,400-episode ROS2 benchmark matrix.', N))
S.append(Spacer(1, 6))
S.append(Paragraph('Reproducibility: fixed seeds throughout; '
                   'tests/test_all_modules.py verifies all six modules; '
                   'results CSVs committed alongside plots.', CAP))

doc.build(S)
print('paper_report.pdf generated')
