"""SAC variant — thin wrapper over ppo_training.py's env and main()."""
import sys
from rl.ppo_training import main

if __name__ == '__main__':
    sys.argv += ['--algo', 'sac']
    main()
