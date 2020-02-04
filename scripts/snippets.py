# Cubehelix Paletter
import seaborn as sns

palette = sns.cubehelix_palette(start=1.9, hue=3, n_colors=18, rot=2.4, gamma=1.3)
sns.palplot(palette)
['#{:02x}{:02x}{:02x}'.format(int(255 * t[0]), int(255 * t[1]), int(255 * t[2])) for t in palette]
