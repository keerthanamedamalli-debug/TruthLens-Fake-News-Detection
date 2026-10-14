"""
Source Credibility Analysis Module for TruthLens.
- Normalizes and extracts domains from input URLs or strings.
- Queries a curated credibility database of ~60 known media domains.
- Constructs a NetworkX graph with sources, categories, and relationship edges.
- Computes graph-propagated credibility scores using seed-weighted PageRank / neighbour averaging.
- Provides interactive visualizers using PyVis and Plotly.
"""

import sys
import re
import json
import logging
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import networkx as nx

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

logger = logging.getLogger("TruthLens-Credibility")

# Known aliases mapping common informal names to domains
NAME_TO_DOMAIN = {
    "reuters": "reuters.com",
    "ap": "apnews.com",
    "associated press": "apnews.com",
    "bbc": "bbc.com",
    "bbc news": "bbc.com",
    "the hindu": "thehindu.com",
    "hindu": "thehindu.com",
    "ndtv": "ndtv.com",
    "indian express": "indianexpress.com",
    "the indian express": "indianexpress.com",
    "alt news": "altnews.in",
    "altnews": "altnews.in",
    "boom": "boomlive.in",
    "boom live": "boomlive.in",
    "snopes": "snopes.com",
    "politifact": "politifact.com",
    "guardian": "theguardian.com",
    "the guardian": "theguardian.com",
    "nyt": "nytimes.com",
    "new york times": "nytimes.com",
    "nature": "nature.com",
    "who": "who.int",
    "cdc": "cdc.gov",
    "pib": "pib.gov.in",
    "wsj": "wsj.com",
    "bloomberg": "bloomberg.com",
    "times of india": "timesofindia.indiatimes.com",
    "hindustan times": "hindustantimes.com",
    "mint": "livemint.com",
    "livemint": "livemint.com",
    "daily mail": "dailymail.co.uk",
    "fox news": "foxnews.com",
    "opindia": "opindia.com",
    "the onion": "theonion.com",
    "onion": "theonion.com",
    "babylon bee": "babylonbee.com",
    "infowars": "infowars.com",
    "natural news": "naturalnews.com",
    "postcard news": "postcard.news",
    "greatgameindia": "greatgameindia.com"
}


def extract_domain(input_str: Optional[str]) -> str:
    """
    Extracts a clean registered domain or alias from a URL, hostname, or free text source name.
    Examples:
      - 'https://www.reuters.com/world/india/article-123' -> 'reuters.com'
      - 'bbc.com' -> 'bbc.com'
      - 'The Hindu' -> 'thehindu.com'
      - None or empty -> ''
    """
    if not input_str or not str(input_str).strip():
        return ""

    input_clean = str(input_str).strip().lower()

    # Check direct name alias first
    if input_clean in NAME_TO_DOMAIN:
        return NAME_TO_DOMAIN[input_clean]

    # If it contains a URL scheme or looks like a URL
    if not input_clean.startswith("http://") and not input_clean.startswith("https://"):
        if "/" in input_clean or "." in input_clean:
            candidate_url = "https://" + input_clean
        else:
            candidate_url = input_clean
    else:
        candidate_url = input_clean

    try:
        parsed = urlparse(candidate_url)
        netloc = parsed.netloc or parsed.path.split("/")[0]
        # Strip port if any
        netloc = netloc.split(":")[0]
        # Strip leading www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if "." in netloc:
            return netloc
    except Exception:
        pass

    # Secondary fuzzy alias check
    for alias, dom in NAME_TO_DOMAIN.items():
        if alias in input_clean:
            return dom

    return input_clean


class CredibilityManager:
    """
    Manages source credibility lookups, network graph creation, and PageRank propagation.
    """
    def __init__(self):
        self.sources_df = self._load_csv()
        self.sources_dict = self.sources_df.set_index("domain").to_dict(orient="index")
        self.graph, self.graph_scores = self._build_and_propagate_graph()

    def _load_csv(self) -> pd.DataFrame:
        if config.SOURCE_CREDIBILITY_PATH.exists():
            return pd.read_csv(config.SOURCE_CREDIBILITY_PATH)
        logger.warning("source_credibility.csv not found at %s. Returning fallback dataframe.", config.SOURCE_CREDIBILITY_PATH)
        return pd.DataFrame(columns=["domain", "name", "credibility_score", "category", "country", "notes"])

    def _build_and_propagate_graph(self) -> Tuple[nx.DiGraph, Dict[str, float]]:
        """
        Loads the credibility network graph, adds nodes and edges,
        and runs a personalized PageRank propagation from seed nodes.
        """
        G = nx.DiGraph()

        if config.SOURCE_GRAPH_PATH.exists():
            try:
                with open(config.SOURCE_GRAPH_PATH, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)

                for node in graph_data.get("nodes", []):
                    G.add_node(
                        node["id"],
                        label=node.get("label", node["id"]),
                        category=node.get("category", "Unknown"),
                        base_credibility=float(node.get("base_credibility", 50)),
                        node_type=node.get("type", "source")
                    )

                for edge in graph_data.get("edges", []):
                    G.add_edge(
                        edge["source"],
                        edge["target"],
                        relationship=edge.get("relationship", "cites"),
                        weight=float(edge.get("weight", 0.8))
                    )
            except Exception as e:
                logger.error("Error reading source_graph.json: %s", e)

        # Ensure all domains from CSV exist in the graph
        for domain, row in self.sources_dict.items():
            if domain not in G:
                G.add_node(
                    domain,
                    label=row.get("name", domain),
                    category=row.get("category", "Unknown"),
                    base_credibility=float(row.get("credibility_score", 50)),
                    node_type="source"
                )

        # Compute graph-propagated credibility via seed-weighted PageRank
        # Seed personalization: Trusted nodes get high weight, Unreliable/Satire get minimal
        personalization = {}
        for n, data in G.nodes(data=True):
            cat = data.get("category", "Unknown")
            base = data.get("base_credibility", 50)
            if cat == "Trusted":
                personalization[n] = base / 100.0
            elif cat == "Mixed":
                personalization[n] = base / 200.0
            elif cat == "Satire":
                personalization[n] = 0.05
            elif cat == "Unreliable":
                personalization[n] = 0.02
            else:
                personalization[n] = 0.1

        total_weight = sum(personalization.values())
        if total_weight > 0:
            for n in personalization:
                personalization[n] /= total_weight

        try:
            # Run PageRank propagation
            pr = nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=100)
            # Scale PageRank values so they map realistically to 0-100
            max_pr = max(pr.values()) if pr else 1.0
            min_pr = min(pr.values()) if pr else 0.0
            range_pr = max_pr - min_pr if max_pr > min_pr else 1.0

            graph_scores = {}
            for n, val in pr.items():
                normalized = (val - min_pr) / range_pr
                base = G.nodes[n].get("base_credibility", 50)
                # Combine 60% base score with 40% graph centrality/propagation
                blended = round(0.60 * base + 0.40 * (normalized * 100), 1)
                graph_scores[n] = min(max(blended, 5.0), 99.0)

        except Exception as e:
            logger.warning("PageRank calculation failed: %s. Using base scores.", e)
            graph_scores = {n: G.nodes[n].get("base_credibility", 50) for n in G.nodes}

        return G, graph_scores

    def evaluate_source(self, source_or_url: Optional[str]) -> Dict[str, Any]:
        """
        Evaluates the credibility of a given source or URL.
        Returns a rich metadata dictionary explaining the score origin.
        """
        raw_input = (source_or_url or "").strip()
        if not raw_input:
            return {
                "domain": "None Specified",
                "source_name": "No Source Provided",
                "credibility_score": 50,
                "category": "Unknown",
                "is_known": False,
                "graph_score": 50,
                "notes": "No source domain or publication specified. Neutral default score (50/100) applied.",
                "derivation": "Neutral default fallback: no source was provided by user."
            }

        domain = extract_domain(raw_input)

        # Check database
        if domain in self.sources_dict:
            entry = self.sources_dict[domain]
            base_score = float(entry.get("credibility_score", 50))
            graph_score = float(self.graph_scores.get(domain, base_score))
            # Blended score: 70% direct editorial rating + 30% network propagation
            combined_score = round(0.70 * base_score + 0.30 * graph_score)
            category = entry.get("category", "Unknown")
            name = entry.get("name", domain)
            notes = entry.get("notes", "")

            derivation = (
                f"Source verified in database as '{name}' ({category}). "
                f"Base editorial rating: {int(base_score)}/100. "
                f"Graph network propagation score: {round(graph_score, 1)}/100. "
                f"Final combined credibility: {combined_score}/100."
            )

            return {
                "domain": domain,
                "source_name": name,
                "credibility_score": combined_score,
                "category": category,
                "is_known": True,
                "graph_score": round(graph_score, 1),
                "notes": notes,
                "derivation": derivation
            }

        # Unknown domain
        return {
            "domain": domain,
            "source_name": domain.title() if domain else "Unknown Domain",
            "credibility_score": 50,
            "category": "Unknown",
            "is_known": False,
            "graph_score": 50,
            "notes": "Domain not present in verified index. Neutral baseline score (50/100) assigned.",
            "derivation": (
                f"Domain '{domain}' is not in the curated source index. "
                "Assigned neutral baseline score 50/100 with 'Unknown' flag."
            )
        }

    def render_pyvis_html(self, height: str = "550px") -> str:
        """
        Renders the credibility graph as an interactive HTML widget using PyVis.
        Uses dark navy background and category color coding.
        """
        try:
            from pyvis.network import Network
            net = Network(height=height, width="100%", bgcolor="#0b132b", font_color="#e0e6ed", directed=True)

            color_map = {
                "Trusted": "#00f0ff",      # Teal / Cyan
                "Mixed": "#ffb703",        # Amber
                "Satire": "#c77dff",       # Purple
                "Unreliable": "#ff0055",   # Crimson / Magenta
                "Topic": "#48cae4",        # Soft Blue
                "Unknown": "#8d99ae"       # Grey
            }

            for n, data in self.graph.nodes(data=True):
                cat = data.get("category", "Unknown")
                node_type = data.get("node_type", "source")
                base_cred = data.get("base_credibility", 50)
                graph_cred = self.graph_scores.get(n, base_cred)
                color = color_map.get(cat, "#8d99ae")

                # Size nodes proportionally to credibility
                size = 18 if node_type == "topic" else max(int(graph_cred * 0.35), 12)
                title_html = (
                    f"<b>{data.get('label', n)}</b><br>"
                    f"Category: {cat}<br>"
                    f"Base Credibility: {base_cred}<br>"
                    f"Network Credibility: {round(graph_cred, 1)}"
                )

                net.add_node(
                    n,
                    label=data.get("label", n),
                    color=color,
                    size=size,
                    title=title_html,
                    shape="diamond" if node_type == "topic" else "dot"
                )

            for u, v, data in self.graph.edges(data=True):
                rel = data.get("relationship", "")
                weight = data.get("weight", 0.5)
                net.add_edge(u, v, title=f"Rel: {rel} (weight: {weight})", width=max(weight * 2.5, 1.0), color="#3a506b")

            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 30,
                "solver": "forceAtlas2Based",
                "timestep": 0.35
              }
            }
            """)

            return net.generate_html()

        except Exception as e:
            logger.error("Failed to generate PyVis HTML: %s", e)
            return f"<div style='color: white; padding: 20px;'>Graph visualization rendering error: {e}</div>"

    def render_plotly_figure(self):
        """
        Renders the credibility graph using Plotly for native Streamlit embedding.
        """
        import plotly.graph_objects as go

        pos = nx.spring_layout(self.graph, seed=42, k=0.35)

        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#2c3e50"),
            hoverinfo="none",
            mode="lines"
        )

        color_map = {
            "Trusted": "#00f0ff",
            "Mixed": "#ffb703",
            "Satire": "#c77dff",
            "Unreliable": "#ff0055",
            "Topic": "#48cae4",
            "Unknown": "#8d99ae"
        }

        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            data = self.graph.nodes[node]
            cat = data.get("category", "Unknown")
            cred = self.graph_scores.get(node, data.get("base_credibility", 50))
            node_colors.append(color_map.get(cat, "#8d99ae"))
            node_sizes.append(26 if data.get("node_type") == "topic" else max(int(cred * 0.28), 12))
            node_text.append(f"{data.get('label', node)}<br>Category: {cat}<br>Credibility: {round(cred, 1)}/100")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[self.graph.nodes[n].get("label", n) for n in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=9, color="#e0e6ed"),
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=1.5, color="#0b132b")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="TruthLens Source Credibility & Influence Network", font=dict(color="#00f0ff", size=18)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=10, r=10, t=50),
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig


# Global singleton instance for easy import
_credibility_manager = None

def get_credibility_manager() -> CredibilityManager:
    global _credibility_manager
    if _credibility_manager is None:
        _credibility_manager = CredibilityManager()
    return _credibility_manager


def evaluate_source(source_or_url: Optional[str]) -> Dict[str, Any]:
    """Exposed module function for evaluating a source."""
    return get_credibility_manager().evaluate_source(source_or_url)
