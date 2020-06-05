import pandas as pd

a = train_df.groupby(['stage', 'home', 'capa', 'away']).mean().reset_index()[['stage', 'home', 'capa', 'away','y']]
b = train_df.groupby(['stage', 'home', 'capa']).mean().reset_index()[['stage', 'home', 'capa', 'y']]
c = pd.merge(a, b, how='left', on=['stage', 'home', 'capa'])
c = c.rename(columns={'y_x':'y', 'y_y':'y_mean'})
c['y_diff'] = c['y'] - c['y_mean']
c['y_%'] = c['y_diff'] / c['capa']
d = c.groupby(['stage', 'away']).mean().reset_index()[['stage', 'away', 'y_%']]
d[abs(d['y_%'] >= 0.05)].sort_values('y_%')

