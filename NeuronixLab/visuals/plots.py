import matplotlib.pyplot as plt
import seaborn as sns

def heatmap(df):
    fig, ax = plt.subplots(facecolor='none')
    ax.set_facecolor('none')
    
    # Premium light color palette
    cmap = sns.diverging_palette(250, 170, as_cmap=True, center="light")
    
    sns.heatmap(df.corr(), ax=ax, cmap=cmap, annot=True, fmt=".2f",
                cbar_kws={'shrink': .8}, linewidths=0.5, linecolor='#f1f5f9',
                annot_kws={"size": 10}, square=True)
                
    ax.tick_params(colors='#475569')
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(colors='#475569')
        
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    return fig