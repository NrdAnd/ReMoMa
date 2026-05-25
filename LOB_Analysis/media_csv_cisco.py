import pandas as pd
import os

def calcola_media_matrici(file_list, output_file):
    print("Inizio il calcolo della media...")
    df_somma = None
    
    for file in file_list:
        if not os.path.exists(file):
            print(f"Errore: Il file '{file}' non è stato trovato nella cartella.")
            return
        
        # Leggiamo il CSV. index_col=0 è fondamentale perché 
        # la prima colonna contiene gli indici (ask_0_lag_0, ecc.)
        print(f"Caricamento di {file}...")
        df = pd.read_csv(file, index_col=0)
        
        if df_somma is None:
            df_somma = df
        else:
            # Assicuriamoci che le matrici abbiano la stessa forma
            if df_somma.shape != df.shape:
                print(f"Attenzione: Le dimensioni di {file} non corrispondono!")
            df_somma = df_somma.add(df, fill_value=0)
            
    # Calcolo della media matematica elemento per elemento
    df_media = df_somma / len(file_list)
    
    # Salvataggio del risultato
    df_media.to_csv(output_file)
    print(f"\nOperazione completata! Il nuovo CSV è stato salvato come: {output_file}")

if __name__ == "__main__":
    # 1. INSERISCI QUI I NOMI DEI TUOI QUATTRO FILE CSV
    file_input = [
        './csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000.csv', 
        './csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_24.csv', 
        './csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_25.csv', 
        './csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_28.csv'
    ]
    
    # 2. NOME DEL FILE CHE VERRA' CREATO
    file_output = 'lob_similarity_nmi_rellag_lag150_bins2000_mean.csv'
    
    calcola_media_matrici(file_input, file_output)