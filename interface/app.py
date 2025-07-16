import tkinter as tk
from tkinter import ttk, messagebox
from classificadores.perceptron import Perceptron
from classificadores.perceptron_delta import PerceptronDelta
from classificadores.dist_max import MaxDistance
from classificadores.dist_min import MinDistance
from classificadores.bayes import BayesClassifier
from utils.metrics import calcular_metricas
from utils.testes import teste_significancia
from utils.treino import treinar_modelo, df
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.colors import ListedColormap
from classificadores.rede_binaria import RedeNeuralRasa
from classificadores.dendograma_iris import DendrogramaIris
import numpy as np

# Interface do Classificador
class App:
    
    def __init__(self, master):
        self.master = master
        master.title("Comparação de Classificadores com Métricas")
        master.geometry("1200x800")
        

        self.classificadores = {
            "Perceptron": Perceptron,
            "Perceptron Delta": PerceptronDelta,
            "Distância Mínima": MinDistance,
            "Distância Máxima": MaxDistance,
            "Bayes": BayesClassifier,
            "Rede Neural Rasa": RedeNeuralRasa,
        }

        self.classes = ["setosa", "versicolor", "virginica"]
        
        # Resultados dos classificadores
        self.model_results = {
            "clf1": {"y_test": None, "y_pred": None, "model": None},
            "clf2": {"y_test": None, "y_pred": None, "model": None}
        }

        # Container principal
        main_container = ttk.Frame(master)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Criando um canvas com scrollbar - mais compacto e otimizado
        self.canvas = tk.Canvas(main_container)
        self.scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Reduzindo o espaço lateral fazendo com que o canvas se expanda totalmente
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Garantir que a janela do canvas use toda a largura disponível
        self.canvas.bind('<Configure>', self.resize_frame)
        
        # Frames principais
        top_frame = ttk.Frame(self.scrollable_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        main_frame = ttk.Frame(self.scrollable_frame)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Divisão em dois lados (classificador 1 e 2)
        left_frame = ttk.LabelFrame(main_frame, text="Classificador 1")
        left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        right_frame = ttk.LabelFrame(main_frame, text="Classificador 2")
        right_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Bottom frame para teste de significância
        bottom_frame = ttk.Frame(self.scrollable_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Configuração do classificador 1
        clf1_config = ttk.Frame(left_frame)
        clf1_config.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(clf1_config, text="Classificador:").grid(row=0, column=0, padx=5)
        self.clf1_combo = ttk.Combobox(clf1_config, values=list(self.classificadores.keys()), width=15)
        self.clf1_combo.current(0)
        self.clf1_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(clf1_config, text="Classe 1:").grid(row=0, column=2, padx=5)
        self.clf1_class1 = ttk.Combobox(clf1_config, values=self.classes, width=10)
        self.clf1_class1.set("setosa")
        self.clf1_class1.grid(row=0, column=3, padx=5)
        
        ttk.Label(clf1_config, text="Classe 2:").grid(row=0, column=4, padx=5)
        self.clf1_class2 = ttk.Combobox(clf1_config, values=self.classes, width=10)
        self.clf1_class2.set("versicolor")
        self.clf1_class2.grid(row=0, column=5, padx=5)
        
        self.run_btn1 = ttk.Button(clf1_config, text="Executar", command=lambda: self.classificar("clf1"))
        self.run_btn1.grid(row=0, column=6, padx=5)

        # Configuração do classificador 2
        clf2_config = ttk.Frame(right_frame)
        clf2_config.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(clf2_config, text="Classificador:").grid(row=0, column=0, padx=5)
        self.clf2_combo = ttk.Combobox(clf2_config, values=list(self.classificadores.keys()), width=15)
        self.clf2_combo.current(1)
        self.clf2_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(clf2_config, text="Classe 1:").grid(row=0, column=2, padx=5)
        self.clf2_class1 = ttk.Combobox(clf2_config, values=self.classes, width=10)
        self.clf2_class1.set("setosa")
        self.clf2_class1.grid(row=0, column=3, padx=5)
        
        ttk.Label(clf2_config, text="Classe 2:").grid(row=0, column=4, padx=5)
        self.clf2_class2 = ttk.Combobox(clf2_config, values=self.classes, width=10)
        self.clf2_class2.set("versicolor")
        self.clf2_class2.grid(row=0, column=5, padx=5)
        
        self.run_btn2 = ttk.Button(clf2_config, text="Executar", command=lambda: self.classificar("clf2"))
        self.run_btn2.grid(row=0, column=6, padx=5)
        
        # Áreas de métricas e visualizações
        # Classificador 1
        clf1_results = ttk.Frame(left_frame)
        clf1_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text1 = tk.Text(clf1_results, height=10, width=40, font=("Courier New", 10))
        self.result_text1.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        clf1_buttons = ttk.Frame(clf1_results)
        clf1_buttons.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_mc1 = ttk.Button(clf1_buttons, text="Matriz Confusão", 
                                  command=lambda: self.plotar_matriz_confusao("clf1"))
        self.btn_mc1.grid(row=0, column=0, padx=5)
        
        self.btn_sd1 = ttk.Button(clf1_buttons, text="Superfície Decisão", 
                                  command=lambda: self.plotar_superficie_decisao("clf1"))
        self.btn_sd1.grid(row=0, column=1, padx=5)
        
        self.btn_erro1 = ttk.Button(clf1_buttons, text="Erro por Época", 
                                   command=lambda: self.plotar_erro_por_epoca("clf1"))
        self.btn_erro1.grid(row=0, column=2, padx=5)
        
        self.input_entry1 = ttk.Entry(clf1_results, width=40)
        self.input_entry1.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.btn_entrada1 = ttk.Button(clf1_results, text="Classificar Entrada", 
                                     command=lambda: self.classificar_entrada("clf1"))
        self.btn_entrada1.pack(side=tk.TOP)
        
        self.graph_frame1 = ttk.LabelFrame(clf1_results, text="Visualização")
        self.placeholder1 = ttk.Label(self.graph_frame1, text="Aguardando execução para exibir gráfico...")
        self.placeholder1.pack(expand=True, fill=tk.BOTH, pady=20)
        self.graph_frame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Classificador 2
        clf2_results = ttk.Frame(right_frame)
        clf2_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text2 = tk.Text(clf2_results, height=10, width=40, font=("Courier New", 10))
        self.result_text2.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        clf2_buttons = ttk.Frame(clf2_results)
        clf2_buttons.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_mc2 = ttk.Button(clf2_buttons, text="Matriz Confusão", 
                                  command=lambda: self.plotar_matriz_confusao("clf2"))
        self.btn_mc2.grid(row=0, column=0, padx=5)
        
        self.btn_sd2 = ttk.Button(clf2_buttons, text="Superfície Decisão", 
                                  command=lambda: self.plotar_superficie_decisao("clf2"))
        self.btn_sd2.grid(row=0, column=1, padx=5)
        
        self.btn_erro2 = ttk.Button(clf2_buttons, text="Erro por Época", 
                                   command=lambda: self.plotar_erro_por_epoca("clf2"))
        self.btn_erro2.grid(row=0, column=2, padx=5)
        
        self.input_entry2 = ttk.Entry(clf2_results, width=40)
        self.input_entry2.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.btn_entrada2 = ttk.Button(clf2_results, text="Classificar Entrada", 
                                     command=lambda: self.classificar_entrada("clf2"))
        self.btn_entrada2.pack(side=tk.TOP)
        
        self.graph_frame2 = ttk.LabelFrame(clf2_results, text="Visualização")
        self.placeholder2 = ttk.Label(self.graph_frame2, text="Aguardando execução para exibir gráfico...")
        self.placeholder2.pack(expand=True, fill=tk.BOTH, pady=20)
        self.graph_frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # =================== Área do Teste de Significância ========================
        significancia_frame = ttk.LabelFrame(self.scrollable_frame, text="Teste de Significância")
        significancia_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(significancia_frame, text="Comparar classificadores usando teste estatístico:", font=("Arial", 10)).pack(side=tk.TOP, padx=5, pady=2)

        btn_frame = ttk.Frame(significancia_frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        self.teste_btn = ttk.Button(btn_frame, text="Executar Teste de Significância", command=self.executar_teste_significancia)
        self.teste_btn.pack(side=tk.LEFT, padx=5)

        self.teste_result = tk.Text(significancia_frame, height=3, width=100, font=("Courier New", 10))
        self.teste_result.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)


        # =================== Área do Hopfield ========================
        hopfield_frame = ttk.LabelFrame(self.scrollable_frame, text="Rede de Hopfield - Memória Associativa")
        hopfield_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.hopfield_result = tk.Text(hopfield_frame, height=10, width=100, font=("Courier New", 10))
        self.hopfield_result.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_hopfield = ttk.Button(hopfield_frame, text="Executar Hopfield", command=self.executar_hopfield)
        self.btn_hopfield.pack(side=tk.LEFT, padx=5, pady=5)

        # =================== Área da Rede Neural Multiclasse ========================
        multiclasse_frame = ttk.LabelFrame(self.scrollable_frame, text="Rede Neural Multiclasse - Classificação com 3 classes")
        multiclasse_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Botão para treinar a rede
        self.btn_treinar_multiclasse = ttk.Button(multiclasse_frame, text="Treinar Rede Multiclasse", command=self.treinar_multiclasse)
        self.btn_treinar_multiclasse.pack(anchor='w', padx=10, pady=5)

        # Campo para mostrar métricas
        self.rede_multi_result = tk.Text(multiclasse_frame, height=10, width=100, font=("Courier New", 10))
        self.rede_multi_result.pack(anchor='w', fill=tk.X, padx=10, pady=5)


        # Entrada manual
        entry_frame = ttk.Frame(multiclasse_frame)
        entry_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.entrada_manual = []
        atributos = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        for i, nome in enumerate(atributos):
            lbl = ttk.Label(entry_frame, text=nome)
            lbl.grid(row=0, column=i)
            ent = ttk.Entry(entry_frame, width=10)
            ent.grid(row=1, column=i)
            self.entrada_manual.append(ent)

        # Botão para prever amostra
        self.btn_prever_amostra = ttk.Button(multiclasse_frame, text="Classificar Amostra", command=self.classificar_multiclasse)
        self.btn_prever_amostra.pack(side=tk.LEFT, padx=5, pady=5)

        # Botão para plotar erro
        self.btn_plotar_erro = ttk.Button(multiclasse_frame, text="Plotar Erro", command=self.plotar_erro_multiclasse)
        self.btn_plotar_erro.pack(side=tk.LEFT, padx=5, pady=5)

        # Botão para exibir arquitetura
        self.btn_mostrar_arquitetura = ttk.Button(multiclasse_frame, text="Mostrar Arquitetura", command=self.mostrar_arquitetura)
        self.btn_mostrar_arquitetura.pack(side=tk.LEFT, padx=5, pady=5)

        # Botão para exibir probabilidades
        self.btn_mostrar_probabilidades = ttk.Button(multiclasse_frame, text="Mostrar Probabilidades", command=self.mostrar_probabilidades)
        self.btn_mostrar_probabilidades.pack(side=tk.LEFT, padx=5, pady=5)

        # =================== Área dos Algoritmos Aglomerativos ========================
    
        aglomerativo_frame = ttk.LabelFrame(self.scrollable_frame, text="Algoritmos Aglomerativos")
        aglomerativo_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10, expand=True)

        controle_agl_frame = ttk.Frame(aglomerativo_frame)
        controle_agl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(controle_agl_frame, text="Método de Ligação:").pack(side=tk.LEFT, padx=5)
        self.metodo_agl_var = tk.StringVar(value='ward')
        self.metodo_agl_combo = ttk.Combobox(controle_agl_frame, textvariable=self.metodo_agl_var,
                                           values=['ward', 'complete', 'average', 'single', 'median', 'centroid'], width=15)
        self.metodo_agl_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controle_agl_frame, text="Grupo:").pack(side=tk.LEFT, padx=5)
        self.grupo_agl_var = tk.IntVar(value=1)
        self.grupo_agl_combo = ttk.Combobox(
            controle_agl_frame,
            textvariable=self.grupo_agl_var,
            values=[1, 2, 3],
            width=5
        )
        self.grupo_agl_combo.pack(side=tk.LEFT, padx=5)

        self.btn_gerar_dendrograma = ttk.Button(controle_agl_frame, text="Gerar Dendrograma",
                                                command=self.gerar_dendrograma_iris)
        self.btn_gerar_dendrograma.pack(side=tk.LEFT, padx=5)

        self.grafico_agl_frame = ttk.Frame(aglomerativo_frame)
        self.grafico_agl_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            

    def gerar_dendrograma_iris(self):
        metodo = self.metodo_agl_var.get()
        grupo = self.grupo_agl_var.get()

        dendrograma = DendrogramaIris()
        fig = dendrograma.plotar_dendrograma_com_cores(
            method=metodo,
            n_clusters=3,
            grupo_id=grupo
        )

        for widget in self.grafico_agl_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.grafico_agl_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_frame(self, event):
        # Redimensiona o conteúdo para preencher toda a largura disponível
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_frame, width=canvas_width)

    def _on_mousewheel(self, event):
        # Diferentes sistemas operacionais têm diferentes eventos de roda de mouse
        if hasattr(event, 'num'):  # Linux
            if event.num == 4:  # roda para cima
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # roda para baixo
                self.canvas.yview_scroll(1, "units")
        else:  # Windows
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def mostrar_plot_tk(self, fig, graph_frame):
        for widget in graph_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        # Atualizar a área de rolagem quando um novo gráfico é mostrado
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def mostrar_plot_tk(self, fig, graph_frame):
        for widget in graph_frame.winfo_children():
            widget.destroy()  # remove placeholder se existir
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_classifier_params(self, clf_id):
        if clf_id == "clf1":
            return {
                "modelo": self.classificadores[self.clf1_combo.get()],
                "class1": self.clf1_class1.get(),
                "class2": self.clf1_class2.get(),
                "result_text": self.result_text1,
                "graph_frame": self.graph_frame1,
                "info_text_widget": self.result_text1
            }
        else:
            return {
                "modelo": self.classificadores[self.clf2_combo.get()],
                "class1": self.clf2_class1.get(),
                "class2": self.clf2_class2.get(),
                "result_text": self.result_text2,
                "graph_frame": self.graph_frame2,
                "info_text_widget": self.result_text2 
            }

    def classificar(self, clf_id):
        params = self.get_classifier_params(clf_id)
        modelo = params["modelo"]
        class1, class2 = params["class1"], params["class2"]
        result_text = params["result_text"]
        
        # Treinar o modelo E obter a instância treinada
        df_filtered = df[df['Species'].isin([class1, class2])]
        X = df_filtered.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df_filtered['Species'].values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Criar e treinar o modelo
        model_instance = modelo()
        model_instance.fit(X_train_scaled, y_train)
        
        # Fazer predições
        y_pred = model_instance.predict(X_test_scaled)
        
        # Calcular métricas
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = calcular_metricas(conf_matrix)
        
        # Armazenar resultados (COM A INSTÂNCIA TREINADA)
        self.model_results[clf_id]["y_test"] = y_test
        self.model_results[clf_id]["y_pred"] = y_pred
        self.model_results[clf_id]["model"] = model_instance  # ← INSTÂNCIA, não classe
        
        # Exibir resultados básicos
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Classificador: {modelo.__name__}\n")
        result_text.insert(tk.END, f"Classes: {class1} vs {class2}\n\n")
        
        # Contagem das amostras
        total = len(y_test)
        classe0 = sum(y_test == 0)
        classe1_count = sum(y_test == 1)
        result_text.insert(tk.END, f"Total de amostras: {total}\n")
        result_text.insert(tk.END, f"Classe 0 ({class1}): {classe0}\n")
        result_text.insert(tk.END, f"Classe 1 ({class2}): {classe1_count}\n\n")
        
        result_text.insert(tk.END, f"Matriz de Confusão:\n{conf_matrix}\n\n")
        nomes = ["Acurácia", "Precisão", "Revocação", "Especificidade", "F1", "F2", "F3", "Kappa", "Matthews"]
        for nome, val in zip(nomes, metrics):
            result_text.insert(tk.END, f"{nome}: {val:.4f}\n")
        
        # Chamar exibir_pesos_rna se for RedeNeuralRasa
        if modelo.__name__ == "RedeNeuralRasa":
            self.exibir_pesos_rna(clf_id, model_instance)
        
        # Atualizar a área de rolagem após adicionar conteúdo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def exibir_pesos_rna(self, clf_id, modelo_treinado):
        """
        Exibe os pesos da Rede Neural Rasa após o treinamento
        """
        params = self.get_classifier_params(clf_id)
        result_text = params["result_text"]
        
        try:
            # Verifica se o modelo tem os atributos de pesos da rede neural
            if hasattr(modelo_treinado, 'rede') and modelo_treinado.rede:
                # Extrai os pesos e bias das camadas
                w1 = modelo_treinado.rede["w1"]
                b1 = modelo_treinado.rede["b1"]
                w2 = modelo_treinado.rede["w2"]
                b2 = modelo_treinado.rede["b2"]
                
                # Converte para arrays NumPy se necessário
                if not isinstance(w1, np.ndarray):
                    w1 = np.array(w1)
                if not isinstance(b1, np.ndarray):
                    b1 = np.array(b1)
                if not isinstance(w2, np.ndarray):
                    w2 = np.array(w2)
                if not isinstance(b2, np.ndarray):
                    b2 = np.array(b2)
                
                # Adiciona informações dos pesos ao texto de resultados
                result_text.insert(tk.END, f"\n{'='*50}\n")
                result_text.insert(tk.END, f"PARÂMETROS DA REDE NEURAL RASA\n")
                result_text.insert(tk.END, f"{'='*50}\n\n")
                
                # Exibe dimensões das matrizes
                result_text.insert(tk.END, f"Dimensões das matrizes:\n")
                result_text.insert(tk.END, f"W1 (entrada → oculta): {w1.shape}\n")
                result_text.insert(tk.END, f"B1 (bias oculta): {b1.shape}\n")
                result_text.insert(tk.END, f"W2 (oculta → saída): {w2.shape}\n")
                result_text.insert(tk.END, f"B2 (bias saída): {b2.shape}\n\n")
                
                # Exibe os pesos da primeira camada (entrada → oculta)
                result_text.insert(tk.END, f"Pesos W1 (entrada → oculta):\n")
                if w1.ndim == 2:
                    for i, linha in enumerate(w1):
                        result_text.insert(tk.END, f"  Neurônio {i+1}: {np.round(linha, 4)}\n")
                else:
                    result_text.insert(tk.END, f"  {np.round(w1, 4)}\n")
                
                # Exibe os bias da camada oculta
                result_text.insert(tk.END, f"\nBias B1 (camada oculta):\n")
                if b1.ndim == 1:
                    for i, bias in enumerate(b1):
                        result_text.insert(tk.END, f"  Neurônio {i+1}: {bias:.4f}\n")
                else:
                    result_text.insert(tk.END, f"  {np.round(b1, 4)}\n")
                
                # Exibe os pesos da segunda camada (oculta → saída)
                result_text.insert(tk.END, f"\nPesos W2 (oculta → saída):\n")
                if w2.ndim == 2:
                    for i, linha in enumerate(w2):
                        result_text.insert(tk.END, f"  Saída {i+1}: {np.round(linha, 4)}\n")
                else:
                    result_text.insert(tk.END, f"  {np.round(w2, 4)}\n")
                
                # Exibe os bias da camada de saída
                result_text.insert(tk.END, f"\nBias B2 (camada de saída):\n")
                if b2.ndim == 1:
                    for i, bias in enumerate(b2):
                        result_text.insert(tk.END, f"  Saída {i+1}: {bias:.4f}\n")
                else:
                    result_text.insert(tk.END, f"  {np.round(b2, 4)}\n")
                
                # Informações adicionais sobre a arquitetura
                result_text.insert(tk.END, f"\n{'='*50}\n")
                result_text.insert(tk.END, f"ARQUITETURA DA REDE:\n")
                result_text.insert(tk.END, f"{'='*50}\n")
                
                # Calcula as dimensões da arquitetura
                if w1.ndim == 2:
                    entrada_size = w1.shape[1]
                    oculta_size = w1.shape[0]
                else:
                    entrada_size = len(w1) if isinstance(w1, (list, tuple)) else 1
                    oculta_size = 1
                    
                if w2.ndim == 2:
                    saida_size = w2.shape[0]
                else:
                    saida_size = 1
                
                result_text.insert(tk.END, f"Camada de entrada: {entrada_size} neurônios\n")
                result_text.insert(tk.END, f"Camada oculta: {oculta_size} neurônios\n")
                result_text.insert(tk.END, f"Camada de saída: {saida_size} neurônios\n")
                
                # Se houver histórico de erro, exibe informações sobre convergência
                if hasattr(modelo_treinado, 'errors_') and modelo_treinado.errors_:
                    erro_inicial = modelo_treinado.errors_[0]
                    erro_final = modelo_treinado.errors_[-1]
                    result_text.insert(tk.END, f"\nCONVERGÊNCIA:\n")
                    result_text.insert(tk.END, f"Erro inicial: {erro_inicial:.6f}\n")
                    result_text.insert(tk.END, f"Erro final: {erro_final:.6f}\n")
                    result_text.insert(tk.END, f"Épocas treinadas: {len(modelo_treinado.errors_)}\n")
                    
                    if erro_inicial != 0:
                        reducao = ((erro_inicial - erro_final) / erro_inicial * 100)
                        result_text.insert(tk.END, f"Redução do erro: {reducao:.2f}%\n")
                    else:
                        result_text.insert(tk.END, f"Erro inicial era zero - não foi possível calcular redução\n")
                        
            else:
                result_text.insert(tk.END, f"\nAviso: Modelo não possui estrutura de rede neural esperada.\n")
                
        except Exception as e:
            result_text.insert(tk.END, f"\nErro ao exibir pesos da Rede Neural Rasa: {str(e)}\n")
            # Debug: mostrar mais informações sobre o erro
            import traceback
            result_text.insert(tk.END, f"Detalhes do erro:\n{traceback.format_exc()}\n")
        
        # Atualiza a área de rolagem
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def executar_teste_significancia(self):
        if (self.model_results["clf1"]["y_test"] is None or 
            self.model_results["clf2"]["y_test"] is None):
            messagebox.showinfo("Informação", "Execute ambos os classificadores primeiro.")
            return
            
        y_true = self.model_results["clf1"]["y_test"]  # Ambos devem ter os mesmos dados de teste
        y_pred1 = self.model_results["clf1"]["y_pred"]
        y_pred2 = self.model_results["clf2"]["y_pred"]
        
        resultado_z = teste_significancia(y_pred1, y_pred2, y_true)
        self.teste_result.delete(1.0, tk.END)
        self.teste_result.insert(tk.END, f"--- Teste de Significância ---\n{resultado_z}")
        
        # Atualizar a área de rolagem após adicionar conteúdo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def plotar_matriz_confusao(self, clf_id):
        if self.model_results[clf_id]["y_test"] is None:
            messagebox.showinfo("Informação", "Execute o classificador primeiro.")
            return
            
        params = self.get_classifier_params(clf_id)
        graph_frame = params["graph_frame"]
        class1, class2 = params["class1"], params["class2"]
        
        y_test = self.model_results[clf_id]["y_test"]
        y_pred = self.model_results[clf_id]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        fig = Figure(figsize=(4, 4), dpi=100) 
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + [class1, class2])
        ax.set_yticklabels([''] + [class1, class2])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        ax.set_title("Matriz de Confusão")
        
        
        # Adiciona os valores nas células da matriz
        for (i, j), val in np.ndenumerate(cm):
            cor_fundo = cax.get_array().reshape(cm.shape)[i, j]
            color = "white" if cor_fundo > cm.max() / 2 else "black"
            ax.text(j, i, f'{val}', ha='center', va='center', color=color, fontsize=12)
        
        self.mostrar_plot_tk(fig, graph_frame)
        
        # Atualizar a área de rolagem após adicionar um gráfico
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    
    def plotar_superficie_decisao(self, clf_id):
        """
        Plota a superfície de decisão e exibe no campo de texto a
        equação discriminante aprendida, com parâmetros reais do modelo.
        """
        # Obter parâmetros
        params = self.get_classifier_params(clf_id)
        modelo = params["modelo"]
        class1 = params["class1"]
        class2 = params["class2"]
        graph_frame = params["graph_frame"]
        result_text = params["result_text"]

        # Preparar dados
        df_filtered = df[df['Species'].isin([class1, class2])]
        X = df_filtered.iloc[:, :2].values
        y = LabelEncoder().fit_transform(df_filtered['Species'].values)
        X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Treinar modelo
        model = modelo()
        model.fit(X_train, y_train)

        # Grid para superfície
        h = 0.02
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # ======== PLOT BONITO ========
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        ax.contour(xx, yy, Z, colors='k', linewidths=0.01)

        for class_value in np.unique(y_train):
            idx = y_train == class_value
            ax.scatter(
                X_train[idx, 0], X_train[idx, 1],
                c=[cmap_bold(class_value)],
                label=f"{class1 if class_value==0 else class2}",
                edgecolor='k',
                s=50
            )

        ax.set_title(f"Superfície de Decisão - {modelo.__name__}", fontsize=12)
        ax.set_xlabel("x1 (característica 1)")
        ax.set_ylabel("x2 (característica 2)")
        ax.legend()
        ax.grid(alpha=0.3)

        # ======== TEXTO EXPLICATIVO ========
        explicacao = []
        barra = "=" * 40
        subbarra = "-" * 30

        explicacao.append(f"{barra}")
        explicacao.append(f"SUPERFÍCIE DE DECISÃO")
        explicacao.append(f"Classificador: {modelo.__name__}")
        explicacao.append(f"Classes: {class1} vs {class2}")
        explicacao.append(f"{barra}\n")

        if modelo.__name__ == "BayesClassifier":
            explicacao.append("=== FÓRMULA DISCRIMINANTE ===")
            explicacao.append(
                "d_i(x) = ln P(c_i) - 0.5 * ln|C_i| - 0.5 * (x - mu_i)^T * C_i^-1 * (x - mu_i)\n"
            )
            explicacao.append("--- PARÂMETROS APRENDIDOS ---")
            for cls in sorted(model.means.keys()):
                mu = model.means[cls]
                cov = model.covs[cls]
                cov_det = np.linalg.det(cov)
                cov_inv = np.linalg.inv(cov)
                prior = model.priors[cls]
                explicacao.append(f"\nClasse {cls}:")
                explicacao.append(f"    Prior          : {prior:.4f}")
                explicacao.append(f"    Média (mu)     : [{mu[0]:.4f}, {mu[1]:.4f}]")
                explicacao.append(f"    |C|            : {cov_det:.4f}")
                explicacao.append(f"    C^-1           :\n{np.array2string(cov_inv, precision=4)}")

        elif modelo.__name__ in ["Perceptron", "PerceptronDelta"]:
            explicacao.append("=== FÓRMULA DISCRIMINANTE ===")
            explicacao.append("g(x) = w^T * x + b = 0\n")
            w = model.weights if hasattr(model, 'weights') else model.w_
            explicacao.append("--- PARÂMETROS APRENDIDOS ---")
            if len(w) == 3:
                explicacao.append(f"    w = [{w[1]:.4f}, {w[2]:.4f}]")
                explicacao.append(f"    b = {w[0]:.4f}")
                explicacao.append(f"    Equação: {w[1]:.4f} * x1 + {w[2]:.4f} * x2 + {w[0]:.4f} = 0")
            elif len(w) == 2:
                explicacao.append(f"    w = [{w[0]:.4f}, {w[1]:.4f}]")

        elif modelo.__name__ == "MinDistance":
            explicacao.append("=== FÓRMULA DISCRIMINANTE ===")
            explicacao.append("d_i(x) = (x - m_i)^T * (x - m_i)\n")
            explicacao.append("--- PARÂMETROS APRENDIDOS ---")
            for cls in sorted(model.prototypes.keys()):
                m_i = model.prototypes[cls]
                explicacao.append(f"    Classe {cls} -> Protótipo m_i = [{m_i[0]:.4f}, {m_i[1]:.4f}]")

        elif modelo.__name__ == "MaxDistance":
            explicacao.append("=== FÓRMULA DISCRIMINANTE ===")
            explicacao.append("d_i(x) = x^T * m_i - 0.5 * m_i^T * m_i\n")
            explicacao.append("--- PARÂMETROS APRENDIDOS ---")
            for cls in sorted(model.mean_vectors.keys()):
                mu = model.mean_vectors[cls]
                mi_term = 0.5 * np.dot(mu, mu)
                explicacao.append(f"\nClasse {cls}:")
                explicacao.append(f"    Média (m_i)          : [{mu[0]:.4f}, {mu[1]:.4f}]")
                explicacao.append(f"    0.5 * m_i^T * m_i    : {mi_term:.4f}")

        else:
            explicacao.append("Classificador sem fórmula discriminante específica implementada.\n")

        # Limpar e inserir no Text
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "\n".join(explicacao))

        # Mostrar o gráfico
        self.mostrar_plot_tk(fig, graph_frame)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def plotar_erro_por_epoca(self, clf_id):
        params = self.get_classifier_params(clf_id)
        modelo = params["modelo"]
        class1, class2 = params["class1"], params["class2"]
        graph_frame = params["graph_frame"]
        
        df_filtered = df[df['Species'].isin([class1, class2])]
        X = df_filtered.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df_filtered['Species'].values)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = modelo()
        model.fit(X_scaled, y)

        if not hasattr(model, "errors_") or not model.errors_:
            messagebox.showinfo("Info", "Este classificador não possui gráfico de erro.")
            return

        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(model.errors_, marker='o')
        ax.set_title(f"Erro por Época - {modelo.__name__}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Erro Quadrático Médio")
        self.mostrar_plot_tk(fig, graph_frame)
        
        # Atualizar a área de rolagem após adicionar um gráfico
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def classificar_entrada(self, clf_id):
        if clf_id == "clf1":
            entrada = self.input_entry1.get()
            result_text = self.result_text1
        else:
            entrada = self.input_entry2.get()
            result_text = self.result_text2
            
        try:
            valores = np.array([float(v) for v in entrada.strip().split()])
            if len(valores) != 4:
                raise ValueError
        except Exception:
            messagebox.showerror("Erro", "Insira 4 valores numéricos separados por espaço.")
            return

        params = self.get_classifier_params(clf_id)
        modelo = params["modelo"]
        class1, class2 = params["class1"], params["class2"]
        
        df_filtered = df[df['Species'].isin([class1, class2])]
        X = df_filtered.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df_filtered['Species'].values)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = modelo()
        model.fit(X_scaled, y)

        entrada_transformada = scaler.transform([valores])
        pred = model.predict(entrada_transformada)[0]
        classe_predita = class1 if pred == 0 else class2
        result_text.insert(tk.END, f"\nEntrada: {entrada} → Predição: {classe_predita}\n")
        
        # Atualizar a área de rolagem após adicionar conteúdo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def executar_hopfield(self):
        from sklearn.datasets import load_iris
        import matplotlib.pyplot as plt
        import numpy as np
        from classificadores.hopfield import Hopfield

        # Carrega a base Iris
        iris = load_iris()
        X = iris.data

        # Cria e treina o modelo Hopfield
        hopfield = Hopfield()
        hopfield.fit(X)

        # Original
        original = hopfield.patterns[0]

        # Adiciona ruído
        noisy = hopfield.add_noise(original, n_bits=1)

        # Recupera
        recovered = hopfield.recall_async(noisy)

        # Resultado
        resultado = hopfield.check_match(recovered)

        # Exibe no Text da interface
        self.hopfield_result.delete(1.0, tk.END)
        self.hopfield_result.insert(tk.END, f"Original:   {original}\n")
        self.hopfield_result.insert(tk.END, f"Com ruído:  {noisy}\n")
        self.hopfield_result.insert(tk.END, f"Recuperado: {recovered}\n")
        self.hopfield_result.insert(tk.END, f"Resultado:  {resultado}\n")

        # Exibe o gráfico
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(original.reshape(2, 2), cmap="gray", aspect="auto")

        plt.subplot(1, 3, 2)
        plt.title("Com Ruído")
        plt.imshow(noisy.reshape(2, 2), cmap="gray", aspect="auto")

        plt.subplot(1, 3, 3)
        plt.title("Recuperado")
        plt.imshow(recovered.reshape(2, 2), cmap="gray", aspect="auto")

        plt.tight_layout()
        plt.show()


    def treinar_multiclasse(self):
        from classificadores.rede_multicamadas import RedeNeuralMulticlasse, one_hot_encode
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

        # Normalização manual para manter os valores na rede
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std

        y_train_encoded = one_hot_encode(y_train, num_classes=3)

        self.rede_multiclasse = RedeNeuralMulticlasse(input_size=4, hidden_size=6, output_size=3)
        self.rede_multiclasse.X_mean = X_mean
        self.rede_multiclasse.X_std = X_std

        self.rede_multiclasse.train(X_train_norm, y_train_encoded, epochs=1000, learning_rate=0.1)

        y_pred_probs = self.rede_multiclasse.forward(X_test_norm)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_test, y_pred, target_names=iris.target_names)
        matrix = confusion_matrix(y_test, y_pred)

        self.rede_multi_result.delete(1.0, tk.END)
        self.rede_multi_result.insert(tk.END, "Matriz de Confusão:\n")
        self.rede_multi_result.insert(tk.END, str(matrix) + "\n\n")
        self.rede_multi_result.insert(tk.END, "Relatório de Classificação:\n")
        self.rede_multi_result.insert(tk.END, report)


    def classificar_multiclasse(self):
        try:
            atributos = [float(e.get()) for e in self.entrada_manual]
            classe, saida = self.rede_multiclasse.prever_amostra_manual(atributos)
            resultado = f"Amostra classificada como: {classe}\nSaída da rede: {np.round(saida, 3)}\n"
        except Exception as e:
            resultado = f"Erro na classificação: {str(e)}"

        self.rede_multi_result.insert(tk.END, "\n" + resultado)


    def plotar_erro_multiclasse(self):
        try:
            self.rede_multiclasse.plotar_erro()
        except Exception as e:
            print("Erro ao plotar erro:", e)


    def mostrar_arquitetura(self):
        try:
            arquitetura = self.rede_multiclasse.obter_arquitetura()
            self.rede_multi_result.insert(tk.END, "\n" + arquitetura + "\n")
        except Exception as e:
            self.rede_multi_result.insert(tk.END, f"\nErro ao obter arquitetura: {str(e)}\n")
            

    def mostrar_probabilidades(self):
        try:
            import matplotlib.pyplot as plt

            atributos = [float(e.get()) for e in self.entrada_manual]
            probs = self.rede_multiclasse.obter_probabilidades(atributos)
            
            # Exibir no Text
            texto = "\nProbabilidades das classes:\n"
            for i, p in enumerate(probs):
                texto += f"Classe {i}: {p:.4f}\n"
            self.rede_multi_result.insert(tk.END, texto)

            # Exibir gráfico
            plt.figure(figsize=(6, 4))
            plt.bar(range(len(probs)), probs, tick_label=["Classe 0", "Classe 1", "Classe 2"], color='skyblue')
            plt.ylim(0, 1)
            plt.title("Probabilidades por Classe (Softmax)")
            plt.ylabel("Probabilidade")
            plt.xlabel("Classe")
            plt.grid(True, axis='y')
            plt.show()

        except Exception as e:
            self.rede_multi_result.insert(tk.END, f"\nErro ao obter probabilidades: {str(e)}\n")







