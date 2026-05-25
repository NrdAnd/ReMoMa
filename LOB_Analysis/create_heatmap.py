import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_heatmap(df: pd.DataFrame, title: str, cmap: str = "YlGnBu", vmin=None, vmax=None, output_file=None):
    """
    Genera e mostra (o salva) una heatmap partendo da un DataFrame.
    """
    plt.figure(figsize=(20, 16))
    
    sns.heatmap(
        df,
        annot=False,
        cmap=cmap,
        cbar_kws={"label": title},
        linewidths=0,
        square=True,
        vmin=vmin,
        vmax=vmax
    )
    
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Se è stato specificato un nome file, salva l'immagine ad alta risoluzione
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Heatmap salvata con successo come: {output_file}")
        
    # Mostra l'immagine a schermo
    plt.show()

if __name__ == "__main__":
    file_input = './csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv'
    
    if os.path.exists(file_input):
        print(f"Caricamento di {file_input}...")
        # Carichiamo il DataFrame assicurandoci che la prima colonna sia l'indice
        df_media = pd.read_csv(file_input, index_col=0)
        
        print("Generazione della heatmap...")
        plot_heatmap(
            df=df_media, 
            title="NMI Media - Limit Order Book", 
            cmap="YlGnBu",  # Scala dal giallo (basso) al blu scuro (alto)
            vmin=0.0, 
            vmax=1.0,
            output_file="heatmap_nmi_ottimizzata.png"
        )
    else:
        print(f"Errore: Il file '{file_input}' non è stato trovato.")