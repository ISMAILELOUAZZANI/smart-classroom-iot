import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from typing import Callable, List, Tuple

# Configuration de la page
st.set_page_config(
    page_title="MOPSO - Multi-Objective PSO",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MOPSO:
    def __init__(self, objective_func: Callable, dim: int, bounds: List[Tuple], 
                 num_particles: int = 50, iterations: int = 100, 
                 w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.archive = []
        self.convergence_history = []
        
    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Vérifie si la solution a domine la solution b"""
        return np.all(a <= b) and np.any(a < b)
    
    def non_dominated_sort(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """Tri non-dominé pour identifier le front de Pareto"""
        pareto_front = []
        for i, solution in enumerate(population):
            is_dominated = False
            for j, other in enumerate(population):
                if i != j and self.dominates(other, solution):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(solution)
        return pareto_front
    
    def calculate_hypervolume(self, front: List[np.ndarray], ref_point: np.ndarray = None) -> float:
        """Calcul simplifié de l'hypervolume (2D seulement)"""
        if len(front) == 0:
            return 0.0
        if len(front[0]) != 2:
            return len(front)  # Fallback pour non-2D
        
        if ref_point is None:
            ref_point = np.array([1.1, 1.1])  # Point de référence par défaut
        
        # Trier par première objective
        sorted_front = sorted(front, key=lambda x: x[0])
        volume = 0.0
        prev_x = 0.0
        
        for point in sorted_front:
            if point[0] > prev_x:
                width = point[0] - prev_x
                height = ref_point[1] - point[1]
                if height > 0:
                    volume += width * height
                prev_x = point[0]
        
        return max(0.0, volume)
    
    def run(self, progress_bar=None, status_text=None):
        """Exécute l'algorithme MOPSO"""
        # Initialisation des particules
        positions = np.random.uniform(
            [b[0] for b in self.bounds], 
            [b[1] for b in self.bounds], 
            (self.num_particles, self.dim)
        )
        velocities = np.zeros_like(positions)
        
        self.archive = []
        self.convergence_history = []
        
        for iteration in range(self.iterations):
            # Évaluation des objectifs
            objectives = [self.objective_func(pos) for pos in positions]
            
            # Mise à jour de l'archive
            combined_objectives = self.archive + objectives
            self.archive = self.non_dominated_sort(combined_objectives)
            
            # Limiter la taille de l'archive
            if len(self.archive) > 100:
                # Sélection aléatoire pour maintenir la diversité
                indices = np.random.choice(len(self.archive), 100, replace=False)
                self.archive = [self.archive[i] for i in indices]
            
            # Mise à jour des particules
            for i in range(self.num_particles):
                if len(self.archive) > 0:
                    # Sélection d'un leader aléatoire dans l'archive
                    leader_idx = np.random.randint(len(self.archive))
                    
                    # Générer une position cible aléatoire
                    target = np.random.uniform(
                        [b[0] for b in self.bounds], 
                        [b[1] for b in self.bounds]
                    )
                    
                    # Mise à jour de la vitesse
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (self.w * velocities[i] + 
                                   self.c1 * r1 * (target - positions[i]) +
                                   self.c2 * r2 * (target - positions[i]))
                    
                    # Mise à jour de la position
                    positions[i] += velocities[i]
                    
                    # Application des contraintes
                    for d in range(self.dim):
                        positions[i][d] = np.clip(positions[i][d], 
                                                self.bounds[d][0], 
                                                self.bounds[d][1])
            
            # Enregistrement de la convergence
            hypervolume = self.calculate_hypervolume(self.archive)
            self.convergence_history.append({
                'iteration': iteration + 1,
                'archive_size': len(self.archive),
                'hypervolume': hypervolume
            })
            
            # Mise à jour de la barre de progression
            if progress_bar:
                progress_bar.progress((iteration + 1) / self.iterations)
            if status_text:
                status_text.text(f'Itération {iteration + 1}/{self.iterations} - '
                               f'Archive: {len(self.archive)} solutions')
        
        return self.archive, self.convergence_history

# Fonctions objectifs prédéfinies
def get_test_problems():
    """Retourne un dictionnaire des problèmes de test"""
    
    def zdt1(x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    def zdt2(x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])
    
    def zdt3(x):
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.array([f1, f2])
    
    def simple_example(x):
        f1 = x[0]
        f2 = 1 - np.sqrt(x[0])
        return np.array([f1, f2])
    
    def binh_korn(x):
        f1 = 4 * x[0]**2 + 4 * x[1]**2
        f2 = (x[0] - 5)**2 + (x[1] - 5)**2
        return np.array([f1, f2])
    
    def schaffer_n1(x):
        f1 = x[0]**2
        f2 = (x[0] - 2)**2
        return np.array([f1, f2])
    
    return {
        'ZDT1 (Convex)': {
            'function': zdt1,
            'dim': 30,
            'bounds': [(0, 1)] * 30,
            'description': 'Front de Pareto convexe'
        },
        'ZDT2 (Non-convex)': {
            'function': zdt2,
            'dim': 30,
            'bounds': [(0, 1)] * 30,
            'description': 'Front de Pareto non-convexe'
        },
        'ZDT3 (Disconnected)': {
            'function': zdt3,
            'dim': 30,
            'bounds': [(0, 1)] * 30,
            'description': 'Front de Pareto discontinu'
        },
        'Simple Example': {
            'function': simple_example,
            'dim': 1,
            'bounds': [(0, 1)],
            'description': 'f1=x, f2=1-√x'
        },
        'Binh-Korn': {
            'function': binh_korn,
            'dim': 2,
            'bounds': [(0, 5), (0, 3)],
            'description': 'Fonction bi-objective classique'
        },
        'Schaffer N.1': {
            'function': schaffer_n1,
            'dim': 1,
            'bounds': [(-1000, 1000)],
            'description': 'f1=x², f2=(x-2)²'
        }
    }

def plot_pareto_front(archive, problem_name):
    """Créer un graphique du front de Pareto"""
    if len(archive) == 0:
        return None
    
    # Convertir en DataFrame pour faciliter la manipulation
    df = pd.DataFrame(archive, columns=['f1', 'f2'])
    
    # Créer le graphique avec Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['f1'],
        y=df['f2'],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            symbol='circle',
            line=dict(width=1, color='darkblue')
        ),
        name='Solutions Pareto',
        hovertemplate='f1: %{x:.4f}<br>f2: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Front de Pareto - {problem_name}',
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def plot_convergence(convergence_history):
    """Créer un graphique de convergence"""
    if not convergence_history:
        return None
    
    df = pd.DataFrame(convergence_history)
    
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Taille de l\'Archive', 'Hypervolume'),
        vertical_spacing=0.1
    )
    
    # Taille de l'archive
    fig.add_trace(
        go.Scatter(
            x=df['iteration'],
            y=df['archive_size'],
            mode='lines+markers',
            name='Taille Archive',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Hypervolume
    fig.add_trace(
        go.Scatter(
            x=df['iteration'],
            y=df['hypervolume'],
            mode='lines+markers',
            name='Hypervolume',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Convergence de l\'Algorithme',
        template='plotly_white',
        height=500
    )
    
    fig.update_xaxes(title_text="Itération", row=2, col=1)
    fig.update_yaxes(title_text="Nombre de Solutions", row=1, col=1)
    fig.update_yaxes(title_text="Hypervolume", row=2, col=1)
    
    return fig

# Interface Streamlit
def main():
    # En-tête
    st.markdown('<h1 class="main-header">🧬 MOPSO - Multi-Objective Particle Swarm Optimization</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Cette application permet de résoudre des **problèmes d'optimisation multi-objectifs** 
    en utilisant l'algorithme MOPSO (Multi-Objective Particle Swarm Optimization).
    """)
    
    # Sidebar pour la configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection du problème
    test_problems = get_test_problems()
    problem_name = st.sidebar.selectbox(
        "Problème de test:",
        options=list(test_problems.keys()),
        index=0
    )
    
    problem_info = test_problems[problem_name]
    
    # Affichage des informations du problème
    st.sidebar.markdown("### 📋 Info Problème")
    st.sidebar.write(f"**Nom:** {problem_name}")
    st.sidebar.write(f"**Dimensions:** {problem_info['dim']}")
    st.sidebar.write(f"**Description:** {problem_info['description']}")
    
    # Paramètres de l'algorithme
    st.sidebar.markdown("### 🔧 Paramètres MOPSO")
    
    num_particles = st.sidebar.slider(
        "Nombre de particules:",
        min_value=20, max_value=200, value=50, step=10
    )
    
    iterations = st.sidebar.slider(
        "Nombre d'itérations:",
        min_value=50, max_value=500, value=100, step=25
    )
    
    w = st.sidebar.slider(
        "Poids d'inertie (w):",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1
    )
    
    c1 = st.sidebar.slider(
        "Coefficient cognitif (c1):",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1
    )
    
    c2 = st.sidebar.slider(
        "Coefficient social (c2):",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1
    )
    
    # Bouton d'exécution
    if st.sidebar.button("🚀 Lancer MOPSO", type="primary"):
        
        # Initialisation de l'algorithme
        mopso = MOPSO(
            objective_func=problem_info['function'],
            dim=problem_info['dim'],
            bounds=problem_info['bounds'],
            num_particles=num_particles,
            iterations=iterations,
            w=w, c1=c1, c2=c2
        )
        
        # Interface de progression
        col1, col2 = st.columns(2)
        with col1:
            progress_bar = st.progress(0)
        with col2:
            status_text = st.empty()
        
        # Exécution de l'algorithme
        start_time = time.time()
        archive, convergence_history = mopso.run(progress_bar, status_text)
        execution_time = time.time() - start_time
        
        # Effacer la barre de progression
        progress_bar.empty()
        status_text.empty()
        
        # Stocker les résultats dans la session
        st.session_state.archive = archive
        st.session_state.convergence_history = convergence_history
        st.session_state.problem_name = problem_name
        st.session_state.execution_time = execution_time
        st.session_state.config = {
            'particles': num_particles,
            'iterations': iterations,
            'w': w, 'c1': c1, 'c2': c2
        }
    
    # Affichage des résultats
    if hasattr(st.session_state, 'archive') and st.session_state.archive:
        
        st.markdown("## 📊 Résultats")
        
        # Métriques de performance
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Solutions trouvées",
                len(st.session_state.archive)
            )
        
        with col2:
            final_hypervolume = st.session_state.convergence_history[-1]['hypervolume']
            st.metric(
                "Hypervolume final",
                f"{final_hypervolume:.4f}"
            )
        
        with col3:
            st.metric(
                "Temps d'exécution",
                f"{st.session_state.execution_time:.2f}s"
            )
        
        with col4:
            final_archive_size = st.session_state.convergence_history[-1]['archive_size']
            st.metric(
                "Taille finale archive",
                final_archive_size
            )
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Front de Pareto")
            pareto_fig = plot_pareto_front(st.session_state.archive, st.session_state.problem_name)
            if pareto_fig:
                st.plotly_chart(pareto_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Convergence")
            conv_fig = plot_convergence(st.session_state.convergence_history)
            if conv_fig:
                st.plotly_chart(conv_fig, use_container_width=True)
        
        # Tableau des solutions
        st.markdown("### 📋 Solutions du Front de Pareto")
        
        if len(st.session_state.archive) > 0:
            # Créer un DataFrame des solutions
            solutions_df = pd.DataFrame(
                st.session_state.archive,
                columns=[f'f{i+1}' for i in range(len(st.session_state.archive[0]))]
            )
            solutions_df.index.name = 'Solution'
            
            # Afficher le tableau avec possibilité de tri
            st.dataframe(
                solutions_df,
                use_container_width=True,
                height=300
            )
            
            # Bouton de téléchargement
            csv = solutions_df.to_csv()
            st.download_button(
                label="📥 Télécharger les solutions (CSV)",
                data=csv,
                file_name=f'pareto_front_{st.session_state.problem_name.lower().replace(" ", "_")}.csv',
                mime='text/csv'
            )
    
    else:
        # Message d'accueil
        st.markdown("""
        ## 🎯 Comment utiliser cette application
        
        1. **Sélectionnez un problème** dans la barre latérale
        2. **Ajustez les paramètres** de l'algorithme selon vos besoins
        3. **Cliquez sur "Lancer MOPSO"** pour démarrer l'optimisation
        4. **Analysez les résultats** : front de Pareto et convergence
        
        ### 📚 Problèmes disponibles
        """)
        
        # Afficher les problèmes disponibles
        for name, info in test_problems.items():
            with st.expander(f"🔍 {name}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Dimensions:** {info['dim']}")
                st.write(f"**Bornes:** {info['bounds']}")
    
    # Section d'information sur l'algorithme
    with st.expander("ℹ️ À propos de MOPSO"):
        st.markdown("""
        **Multi-Objective Particle Swarm Optimization (MOPSO)** est un algorithme évolutionnaire 
        qui étend le PSO traditionnel pour gérer plusieurs objectifs conflictuels simultanément.
        
        **Caractéristiques principales:**
        - 🎯 **Tri non-dominé** pour identifier les solutions Pareto-optimales
        - 📦 **Archive adaptatif** des meilleures solutions
        - 🎲 **Sélection de leaders** basée sur l'archive
        - 📈 **Préservation de la diversité** dans l'espace objectif
        
        **Paramètres:**
        - **w** : Poids d'inertie (contrôle l'influence de la vitesse précédente)
        - **c1** : Coefficient cognitif (attraction vers la meilleure solution personnelle)
        - **c2** : Coefficient social (attraction vers la meilleure solution globale)
        """)

if __name__ == "__main__":
    main()