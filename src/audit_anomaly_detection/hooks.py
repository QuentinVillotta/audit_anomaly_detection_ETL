from kedro.framework.hooks import hook_impl
import streamlit as st

class StreamlitProgressHook:
    def __init__(self):
        self.total_nodes = 0
        self.completed_nodes = 0
        self.progress_bar = None

    @hook_impl
    def before_pipeline_run(self, pipeline, run_params, catalog):
        # Initialiser la barre de progression
        self.total_nodes = len(pipeline.nodes)
        self.completed_nodes = 0
        self.progress_bar = st.progress(0)

    @hook_impl
    def after_node_run(self, node, catalog, inputs, is_async):
        # Incrémenter et mettre à jour la barre de progression après chaque nœud
        self.completed_nodes += 1
        progress = self.completed_nodes / self.total_nodes
        self.progress_bar.progress(progress)

    @hook_impl
    def after_pipeline_run(self, pipeline, run_params, catalog):
        # Fin de la progression
        self.progress_bar.empty()
