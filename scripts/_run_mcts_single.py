"""Test MCTS 1 partita con seed da CLI."""
import sys, time, json, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa
import numpy as np
from risiko_env import encoding as e
e.STAGE_A_ATTIVO = False
from risiko_env import RisikoEnv
from mcts import MCTSPlanner

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_sim", type=int, default=20)
parser.add_argument("--out", default="/tmp/mcts_single.json")
args = parser.parse_args()

env = RisikoEnv(seed=args.seed, mode_1v1=True, reward_mode='margin')
obs, info = env.reset()
planner = MCTSPlanner(env, c_puct=1.4, rng_seed=args.seed)

t0 = time.time()
while True:
    if info is None: break
    mask = info['action_mask']
    legali = np.where(mask)[0]
    if len(legali) == 0: break
    if len(legali) <= 3:
        az = int(np.random.choice(legali))
    else:
        az = planner.scegli_azione(n_simulazioni=args.n_sim)
    obs, r, t, tr, info = env.step(int(az))
    if t or tr: break

elapsed = time.time() - t0
vinc = info.get('vincitore') if info else None
res = {
    'seed': args.seed, 'n_sim': args.n_sim, 'vinc': vinc,
    'reward': float(r), 'elapsed': elapsed,
    'round': env.stato.round_corrente
}

# Append a file collettivo
collettivo = []
out_collettivo = '/tmp/mcts_results.json'
if os.path.exists(out_collettivo):
    try:
        with open(out_collettivo) as f:
            collettivo = json.load(f)
    except Exception:
        pass
collettivo.append(res)
with open(out_collettivo, 'w') as f:
    json.dump(collettivo, f, indent=2)

print(f"seed={args.seed}: vinc={vinc} r={r:+.3f} elapsed={elapsed:.0f}s")
