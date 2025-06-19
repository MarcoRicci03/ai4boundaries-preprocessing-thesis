import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

# Percorso al CSV
csv_path = 'PRIMAPROVA.csv'  # es: 'logs/metrics.csv'
output_path = 'metrics_plot.png'
refresh_interval = 100  # ogni 100 secondi salva il grafico

def plot_metrics(df):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    axs[0].plot(df['epoch'], df['tot_train_loss'], label='Train Loss')
    axs[0].plot(df['epoch'], df['tot_val_loss'], label='Val Loss')
    axs[0].set_title('Loss')
    axs[0].legend()

    axs[1].plot(df['epoch'], df['acc'], label='Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    axs[2].plot(df['epoch'], df['precision'], label='Precision')
    axs[2].plot(df['epoch'], df['recall'], label='Recall')
    axs[2].set_title('Precision & Recall')
    axs[2].legend()

    axs[3].plot(df['epoch'], df['kappa'], label='Kappa')
    axs[3].plot(df['epoch'], df['mcc'], label='MCC')
    axs[3].set_title('Kappa & MCC')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def countdown(seconds):
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\rProssimo aggiornamento tra {i} secondi...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rAggiorno i grafici...                      \n")

# Loop continuo
while True:
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plot_metrics(df)
        else:
            print("CSV non trovato, attendo...")
        countdown(refresh_interval)
    except Exception as e:
        print(f"Errore: {e}")
        time.sleep(refresh_interval)