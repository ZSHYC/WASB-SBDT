import os
import pandas as pd

clips = [
    'datasets/tennis_predict/game_1/Clip_1',
    'datasets/tennis_predict/game_2/Clip_1',
    'datasets/tennis_predict/game_3/Clip_1',
]
frames_in = 3

for c in clips:
    d = c
    files = [f for f in os.listdir(d) if f.lower().endswith('.jpg')]
    csv_path = os.path.join(d, 'Label.csv')
    csv_count = len(pd.read_csv(csv_path)) if os.path.exists(csv_path) else 0
    N = csv_count - frames_in + 1
    def seq_count(N, step):
        return ((N-1)//step)+1 if N>0 else 0
    print('Clip:', c)
    print('  images_count:', len(files))
    print('  csv_count:', csv_count)
    print('  frames_in:', frames_in)
    print('  windows (N):', N)
    print('  sequences step=1:', seq_count(N,1))
    print('  sequences step=3:', seq_count(N,3))
    print()
