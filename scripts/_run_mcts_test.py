"""Test MCTS 10 partite, salva risultati su file."""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers
import numpy as np
from risiko_env import encoding as e
e.STAGE_A_ATTIVO = False
from risiko_env import RisikoEnv
from mcts import MCTSPlanner

risultati = []
n_partite = 10
n_sim = 20

for seed in range(n_partite):
    env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode='margin')
    obs, info = env.reset()
    planner = MCTSPlanner(env, c_puct=1.4, rng_seed=seed)
    
    t0 = time.time()
    while True:
        if info is None: break
        mask = info['action_mask']
        legali = np.where(mask)[0]
        if len(legali) == 0: break
        if len(legali) <= 3:
            az = int(np.random.choice(legali))
        else:
            az = planner.scegli_azione(n_simulazioni=n_sim)
        obs, r, t, tr, info = env.step(int(az))
        if t or tr: break
        if time.time() - t0 > 120:
            break
    
    elapsed = time.time() - t0
    vinc = info.get('vincitore') if info else None
    risultati.append({'seed': seed, 'vinc': vinc, 'reward': float(r),
                      'elapsed': elapsed, 'round': env.stato.round_corrente})
    
    # Salva subito (per recupero parziale)
    with open('/tmp/mcts_results.json', 'w') as f:
        json.dump(risultati, f)

print(f"DONE: {len(risultati)} partite")
