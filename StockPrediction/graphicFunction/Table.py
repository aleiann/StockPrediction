import pandas as pd
import matplotlib.pyplot as plt

def printTable(df, title):
    fig, ax = plt.subplots(figsize=(10, 14))  # Personalizza le dimensioni della figura se necessario

    # Nascondi l'asse
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.2, 1.2)

    plt.title(title)
    plt.show()