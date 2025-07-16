import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.colors import to_hex

class DendrogramaIris:
    def __init__(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.target_names = self.iris.target_names
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.linkage_matrix = None

        # Pré-divide os dados em 3 grupos sem sobreposição
        self.grupos = self._dividir_em_grupos()

    def _dividir_em_grupos(self):
        """
        Divide as amostras em 3 grupos de 45 exemplos cada
        (15 setosa, 15 versicolor, 15 virginica por grupo),
        sem repetição entre os grupos.
        """
        np.random.seed(42)
        grupos = {1: [], 2: [], 3: []}

        indices_por_classe = {i: np.where(self.y == i)[0] for i in range(3)}
        for classe in range(3):
            np.random.shuffle(indices_por_classe[classe])
            grupos[1].extend(indices_por_classe[classe][:15])
            grupos[2].extend(indices_por_classe[classe][15:30])
            grupos[3].extend(indices_por_classe[classe][30:45])
        return grupos

    def get_dados_grupo(self, grupo_id):
        """
        Retorna os dados e rótulos do grupo especificado.
        """
        indices = self.grupos[grupo_id]
        X_grupo = self.X[indices]
        y_grupo = self.y[indices]
        X_scaled = self.scaler.transform(X_grupo)
        return X_scaled, y_grupo

    def calcular_linkage(self, method='ward', metric='euclidean', grupo_id=1):
        """
        Calcula a matriz de linkage para o grupo especificado.
        """
        X_scaled, _ = self.get_dados_grupo(grupo_id)
        if method == 'ward':
            self.linkage_matrix = linkage(X_scaled, method=method)
        else:
            distances = pdist(X_scaled, metric=metric)
            self.linkage_matrix = linkage(distances, method=method)
        return self.linkage_matrix

    def plotar_dendrograma_com_cores(self, method='ward', n_clusters=3, grupo_id=1, figsize=(12, 6)):
        """
        Plota o dendrograma com as cores dos clusters para o grupo especificado.
        """
        self.calcular_linkage(method=method, grupo_id=grupo_id)
        threshold = self.linkage_matrix[-n_clusters + 1, 2]

        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)

        # Define paleta de cores
        palette = sns.color_palette("husl", n_clusters)
        hex_colors = [to_hex(color) for color in palette]
        set_link_color_palette(hex_colors)

        dendro = dendrogram(
            self.linkage_matrix,
            ax=ax,
            color_threshold=threshold,
            above_threshold_color='gray',
            leaf_rotation=90,
            leaf_font_size=9,
            distance_sort='descending',
            count_sort='ascending',
        )

        ax.set_title(
            f'Dendrograma Grupo {grupo_id} com {n_clusters} Clusters - Iris ({method.capitalize()})',
            fontsize=14,
            weight='bold'
        )
        ax.set_xlabel('Índice das Amostras', fontsize=12)
        ax.set_ylabel('Distância', fontsize=12)
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.2)
        ax.text(ax.get_xlim()[1] * 0.8, threshold + 0.1, f'Threshold: {threshold:.2f}',
                fontsize=10, ha='center')

        fig.tight_layout()
        return fig

    def aplicar_aglomerativo_sklearn(self, linkage='ward', n_clusters=3, grupo_id=1):
        """
        Roda o AgglomerativeClustering do sklearn para o grupo especificado.
        """
        X_scaled, _ = self.get_dados_grupo(grupo_id)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X_scaled)
        return labels

    def salvar_dendrograma(self, fig, filename='dendrograma_melhorado.png'):
        fig.savefig(filename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Algoritmos Aglomerativos - Dendrograma Iris")
    root.geometry("1000x700")

    aba_frame = ttk.LabelFrame(root, text="Algoritmos Aglomerativos")
    aba_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Widget de seleção de método e botão
    controle_frame = ttk.Frame(aba_frame)
    controle_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    ttk.Label(controle_frame, text="Método de Ligação:").pack(side=tk.LEFT, padx=5)
    metodo_var = tk.StringVar(value='ward')
    metodo_combo = ttk.Combobox(controle_frame, textvariable=metodo_var,
                                 values=['ward', 'complete', 'average', 'single', 'median'], width=15)
    metodo_combo.pack(side=tk.LEFT, padx=5)

    ttk.Label(controle_frame, text="Grupo:").pack(side=tk.LEFT, padx=5)
    grupo_var = tk.IntVar(value=1)
    grupo_combo = ttk.Combobox(controle_frame, textvariable=grupo_var, values=[1, 2, 3], width=5)
    grupo_combo.pack(side=tk.LEFT, padx=5)

    def gerar_dendrograma():
        metodo = metodo_var.get()
        grupo = grupo_var.get()
        dendrograma = DendrogramaIris()
        fig = dendrograma.plotar_dendrograma_com_cores(method=metodo, n_clusters=3, grupo_id=grupo)
        for widget in grafico_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=grafico_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    ttk.Button(controle_frame, text="Gerar Dendrograma", command=gerar_dendrograma).pack(side=tk.LEFT, padx=5)

    grafico_frame = ttk.Frame(aba_frame)
    grafico_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    root.mainloop()